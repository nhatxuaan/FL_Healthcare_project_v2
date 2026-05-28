from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

def flatten_weights(params: List[np.ndarray]) -> np.ndarray:
    """Duỗi thẳng toàn bộ các tầng trọng số thành 1 vector duy nhất để tính toán toán học."""
    return np.concatenate([p.flatten() for p in params])

def compute_cosine_divergence(g_i: np.ndarray, g_j: np.ndarray) -> float:
    """Tính toán Công thức (8): Divergence dựa trên Cosine Similarity."""
    norm_i = np.linalg.norm(g_i)
    norm_j = np.linalg.norm(g_j)
    
    if norm_i == 0 or norm_j == 0:
        return 1.0  # Tránh lỗi chia cho 0, nếu gradient bằng 0 coi như phân kì hoàn toàn
        
    cosine_sim = np.dot(g_i, g_j) / (norm_i * norm_j)
    return 1.0 - cosine_sim


def aggregate_adaptive(
    client_params_list: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    tau: float,
    learning_rate: float,
    gamma: float = 1.0,  # Tham số gamma trong công thức (7)
) -> Tuple[List[np.ndarray], str, float]:
    """
    Hàm gộp thích ứng chuẩn hóa theo bài báo:
    1. Xấp xỉ Gradient updates từ Parameter Deltas.
    2. Tính ma trận phân kì giữa các Client bằng Cosine Similarity.
    3. Tính trọng số alpha_i theo công thức tương tác cặp (hoặc so với trung bình hệ thống).
    4. Chuyển đổi linh hoạt giữa FedAvg và FedSGD dựa trên ngưỡng tau.
    """
    if not client_params_list:
        raise ValueError("No clients to aggregate")

    num_clients = len(client_params_list)
    
    # Bước 1: Xấp xỉ Gradients (G_i = W_global - W_client) cho từng client
    client_gradients_flat = []
    for client_params in client_params_list:
        g_parts = []
        for c_p, g_p in zip(client_params, global_params):
            g_parts.append((g_p - c_p).flatten()) # Gradient update hướng về phía client học
        client_gradients_flat.append(np.concatenate(g_parts))
        
    # Bước 2: Tính toán độ phân kì trung bình của hệ thống (Divergence Metric)
    # Ta tính toán độ phân kì trung bình giữa các cặp client liên tiếp để đại diện cho Delta(G_i, G_j)
    divergences = []
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            div_ij = compute_cosine_divergence(client_gradients_flat[i], client_gradients_flat[j])
            divergences.append(div_ij)
            
    # Độ phân kì đại diện cho vòng này (Lấy trung bình các cặp)
    current_round_divergence = np.mean(divergences) if divergences else 0.0

    # Bước 3: Tính toán Trọng số Thích ứng alpha_i cho từng Client dựa trên Công thức (7)
    # Ở đây bài báo so sánh cặp (G_i, G_j), để tổng quát hóa cho hệ thống, ta tính độ lệch của G_i so với G_trung_bình
    avg_gradient = np.mean(client_gradients_flat, axis=0)
    alpha_list = []
    for idx in range(num_clients):
        div_i_to_avg = compute_cosine_divergence(client_gradients_flat[idx], avg_gradient)
        # Công thức (7)
        alpha_i = 1.0 / (1.0 + np.exp(-gamma * div_i_to_avg))
        alpha_list.append(alpha_i)
        
    # Chuẩn hóa lại tổng alpha về 1 để không làm lệch scale của mô hình
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
                # Cập nhật dạng FedSGD: W_new = W_old - lr * tổng(alpha * Gradient)
                aggregated[i] -= (w_i * learning_rate * gradient_approx)
    else:
        # Tình huống: Các máy đồng thuận tốt (Divergence < tau) -> Dùng FedAvg để giảm phương sai (Variance)
        algorithm = "FedAvg"
        # Khởi tạo ma trận gộp bằng 0
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