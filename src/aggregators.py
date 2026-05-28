import logging
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

# Import các hàm gộp thích ứng từ file aggregators của bạn
from src.aggregators import aggregate_adaptive, compute_cosine_divergence

logger = logging.getLogger(__name__)

class AdaptiveAggregationStrategy(fl.server.strategy.Strategy):
    """
    Chiến lược gộp thích ứng (FedAvg <-> FedSGD) dựa trên độ phân kỳ Cosine
    được thiết kế theo cấu trúc hệ thống của Flower Framework.
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
        
        # Các biến lưu vết lịch sử huấn luyện
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
        """Khởi tạo tham số toàn cục ban đầu khi hệ thống FL bắt đầu chạy."""
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Lựa chọn các Client tham gia vào vòng huấn luyện hiện tại."""
        config = {}
        fit_ins = FitIns(parameters, config)
        
        # Lấy danh sách tất cả các client sẵn sàng trong hệ thống
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Lựa chọn các Client tham gia vào vòng đánh giá cục bộ (nếu có)."""
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
        """
        Gộp kết quả huấn luyện từ các client bằng thuật toán Thích ứng thông minh.
        """
        self.round = server_round
        
        if not results:
            logger.warning("No client results to aggregate")
            return None, {}
        
        # Trích xuất trọng số nhận được dạng NDArrays và số lượng mẫu của từng client
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        client_weights = [w for w, _ in weights_results]
        sample_counts = [n for _, n in weights_results]
        
        # Nếu chưa khởi tạo global_params, lấy trọng số của client đầu tiên làm mốc tham chiếu
        if self.global_params is None:
            self.global_params = client_weights[0]
            logger.info("Global parameters initialized from first client")
        
        # ----------------------------------------------------------------------
        # GỌI HÀM GỘP THÍCH ỨNG CHUẨN (Khớp hoàn toàn với file aggregators.py)
        # ----------------------------------------------------------------------
        aggregated, algorithm_used, div = aggregate_adaptive(
            client_params_list=client_weights,
            global_params=self.global_params,
            tau=self.tau,
            learning_rate=self.learning_rate,
            gamma=1.0  # Hệ số nhạy trong hàm Sigmoid của bài báo
        )
        
        # Lưu kết quả phân kỳ và thuật toán được chọn vào lịch sử lưu vết
        divergence = div
        self.divergence_history.append(divergence)
        self.algorithm_history.append(algorithm_used)
        
        # Cập nhật lại trọng số toàn cục mới nhất của server để chuẩn bị cho vòng sau
        self.global_params = aggregated
        
        # Tính toán giá trị Loss trung bình có trọng số của vòng này
        total_loss = sum(
            fit_res.metrics.get("loss", 0.0) * fit_res.num_examples
            for _, fit_res in results
        )
        avg_loss = total_loss / sum(sample_counts) if sample_counts else 0
        self.loss_history.append(avg_loss)
        
        # Thu thập các chỉ số bảo mật Differential Privacy (Epsilon) nếu có sử dụng Opacus
        privacy_metrics = {}
        epsilon_values = [
            fit_res.metrics.get("epsilon", float('inf'))
            for _, fit_res in results
        ]
        if any(e != float('inf') for e in epsilon_values):
            privacy_metrics["epsilon"] = min(epsilon_values)
        
        # Tạo từ điển tổng hợp các thông số trả về cho file log chính (simulation.py)
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
        
        # Chuyển đổi mảng NDArrays đã gộp thành định dạng Parameters của Flower để phân phát lại cho các client
        self.final_parameters = ndarrays_to_parameters(aggregated)
        
        return self.final_parameters, metrics

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Gộp kết quả đánh giá phân tán của các client (nếu có sử dụng thử nghiệm)."""
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
        """Hàm đánh giá tập trung tại Server (sử dụng khi Server có tập test riêng)."""
        # Trả về None để Flower biết ta thực hiện đánh giá độc lập sau khi kết thúc các round
        return None