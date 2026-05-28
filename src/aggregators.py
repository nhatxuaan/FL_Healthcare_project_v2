import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

logger = logging.getLogger(__name__)

# ==============================================================================
# CÁC HÀM TOÁN HỌC GỘP THÍCH ỨNG (Đặt trực tiếp ở đây để tránh lỗi Import)
# ==============================================================================
def compute_cosine_divergence(g_i: np.ndarray, g_j: np.ndarray) -> float:
    """Tính toán Công thức (8): Divergence dựa trên Cosine Similarity."""
    norm_i = np.linalg.norm(g_i)
    norm_j = np.linalg.norm(g_j)
    
    if norm_i == 0 or norm_j == 0:
        return 1.0  # Tránh lỗi chia cho 0
        
    cosine_sim = np.dot(g_i, g_j) / (norm_i * norm_j)
    return 1.0 - cosine_sim


def aggregate_adaptive_internal(
    client_params_list: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    tau: float,
    learning_rate: float,
    gamma: float = 1.0,
) -> Tuple[List[np.ndarray], str, float]:
    """
    Hàm gộp thích ứng chuẩn hóa theo bài báo (Chuyển đổi linh hoạt FedAvg / FedSGD)
    """
    if not client_params_list:
        raise ValueError("No clients to aggregate")

    num_clients = len(client_params_list)
    
    # Bước 1: Xấp xỉ Gradients (G_i = W_global - W_client)
    client_gradients_flat = []
    for client_params in client_params_list:
        g_parts = []
        for c_p, g_p in zip(client_params, global_params):
            g_parts.append((g_p - c_p).flatten())
        client_gradients_flat.append(np.concatenate(g_parts))
        
    # Bước 2: Tính toán độ phân kì trung bình của hệ thống (Divergence Metric)
    divergences = []
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            div_ij = compute_cosine_divergence(client_gradients_flat[i], client_gradients_flat[j])
            divergences.append(div_ij)
            
    current_round_divergence = np.mean(divergences) if divergences else 0.0

    # Bước 3: Tính toán Trọng số Thích ứng alpha_i cho từng Client dựa trên Công thức (7)
    avg_gradient = np.mean(client_gradients_flat, axis=0)
    alpha_list = []
    for idx in range(num_clients):
        div_i_to_avg = compute_cosine_divergence(client_gradients_flat[idx], avg_gradient)
        alpha_i = 1.0 / (1.0 + np.exp(-gamma * div_i_to_avg))
        alpha_list.append(alpha_i)
        
    total_alpha = sum(alpha_list)
    normalized_weights = [alpha / total_alpha for alpha in alpha_list]

    # Bước 4: Logic chuyển đổi chiến lược gộp dựa trên ngưỡng tau
    aggregated = [np.copy(param) for param in global_params]

    if current_round_divergence >= tau:
        # Tình huống: Các máy phân kì cao (Divergence >= tau) -> Dùng FedSGD để ổn định (Stability)
        algorithm = "FedSGD"
        for client_idx, client_params in enumerate(client_params_list):
            w_i = normalized_weights[client_idx]
            for i, (client_param, global_param) in enumerate(zip(client_params, global_params)):
                gradient_approx = global_param - client_param
                aggregated[i] -= (w_i * learning_rate * gradient_approx)
    else:
        # Tình huống: Các máy đồng consensus tốt (Divergence < tau) -> Dùng FedAvg để giảm phương sai (Variance)
        algorithm = "FedAvg"
        for i in range(len(aggregated)):
            aggregated[i] = np.zeros_like(aggregated[i])
            
        for client_idx, client_params in enumerate(client_params_list):
            w_i = normalized_weights[client_idx]
            for i, param in enumerate(client_params):
                aggregated[i] += param * w_i

    logger.info(
        f"Adaptive Aggregation: div={current_round_divergence:.4f} "
        f"{'>=' if current_round_divergence >= tau else '<'} τ={tau} -> Sử dụng {algorithm}"
    )
    
    return aggregated, algorithm, float(current_round_divergence)


# ==============================================================================
# CLASS STRATEGY CHÍNH CHO FLOWER SERVER
# ==============================================================================
class AdaptiveAggregationStrategy(fl.server.strategy.Strategy):
    """
    Chiến lược gộp thích ứng tự đóng gói không phụ thuộc file ngoài.
    """
    def __init__(
        self,
        tau: float = 0.1,
        learning_rate: float = 0.01,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.learning_rate = learning_rate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        # Lưu vết lịch sử huấn luyện
        self.loss_history: List[float] = []
        self.divergence_history: List[float] = []
        self.algorithm_history: List[str] = []
        self.round = 0
        self.global_params: Optional[NDArrays] = None
        self.final_parameters: Optional[Parameters] = None

    def __repr__(self) -> str:
        return (
            f"AdaptiveAggregationStrategy(tau={self.tau}, lr={self.learning_rate}, "
            f"min_fit_clients={self.min_fit_clients})"
        )

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.round = server_round
        
        if not results:
            logger.warning("No client results to aggregate")
            return None, {}
        
        # Trích xuất trọng số nhận được từ các client
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        client_weights = [w for w, _ in weights_results]
        sample_counts = [n for _, n in weights_results]
        
        if self.global_params is None:
            self.global_params = client_weights[0]
            logger.info("Global parameters initialized from first client")
        
        # Gọi trực tiếp hàm gộp nội bộ đã khai báo phía trên
        aggregated, algorithm_used, div = aggregate_adaptive_internal(
            client_params_list=client_weights,
            global_params=self.global_params,
            tau=self.tau,
            learning_rate=self.learning_rate,
            gamma=1.0
        )
        
        divergence = div
        self.divergence_history.append(divergence)
        self.algorithm_history.append(algorithm_used)
        
        self.global_params = aggregated
        
        # Tính toán Loss trung bình vòng này
        total_loss = sum(
            fit_res.metrics.get("loss", 0.0) * fit_res.num_examples
            for _, fit_res in results
        )
        avg_loss = total_loss / sum(sample_counts) if sample_counts else 0
        self.loss_history.append(avg_loss)
        
        # Thu thập thông số bảo mật Differential Privacy (Epsilon) nếu có
        privacy_metrics = {}
        epsilon_values = [
            fit_res.metrics.get("epsilon", float('inf'))
            for _, fit_res in results
        ]
        if any(e != float('inf') for e in epsilon_values):
            privacy_metrics["epsilon"] = min(epsilon_values)
        
        metrics = {
            "loss": avg_loss,
            "divergence": divergence,
            "algorithm": algorithm_used,
            "tau": self.tau,
            "num_clients": len(results),
            **privacy_metrics,
        }
        
        logger.info(
            f"Round {server_round}: loss={avg_loss:.4f}, div={divergence:.6f}, "
            f"algorithm={algorithm_used}, epsilon={metrics.get('epsilon', 'N/A')}"
        )
        
        self.final_parameters = ndarrays_to_parameters(aggregated)
        return self.final_parameters, metrics

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
            
        loss_aggregated = sum(res.loss * res.num_examples for _, res in results) / sum(
            res.num_examples for _, res in results
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None