import numpy as np
import matplotlib.pyplot as plt

def visualize_dataset_loss(label_actor_list, map_pred_actor_list, scenario_list, top_k=20):
    # 1. Tiền xử lý dữ liệu
    y_true = np.array(label_actor_list)
    y_pred = np.array(map_pred_actor_list).astype(np.float32)
    
    # Clip để tránh lỗi log(0)
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    # 2. Tính Binary Cross-Entropy Loss
    sample_losses = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    if sample_losses.ndim > 1:
        sample_losses = np.mean(sample_losses, axis=1)
        
    # 3. Trực quan hóa
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Biểu đồ 1: Histogram
    axes[0].hist(sample_losses, bins=50, color='skyblue', edgecolor='navy', alpha=0.7)
    axes[0].set_title('Distribution of Loss across Validation Set', fontsize=14)
    axes[0].set_xlabel('Loss Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Biểu đồ 2: Top K mẫu tệ nhất với tên ID CỤ THỂ
    worst_indices = np.argsort(sample_losses)[::-1][:top_k]
    worst_losses = sample_losses[worst_indices]
    
    # [QUAN TRỌNG] Lấy tên file cụ thể dựa vào index
    worst_scenarios = [scenario_list[i] for i in worst_indices]
    
    axes[1].bar(range(top_k), worst_losses, color='salmon', edgecolor='darkred')
    axes[1].set_title(f'Top {top_k} Worst Predictions', fontsize=14)
    axes[1].set_ylabel('Loss Value', fontsize=12)
    
    # Gắn nhãn trục x là TÊN FILE/SCENARIO
    axes[1].set_xticks(range(top_k))
    axes[1].set_xticklabels(worst_scenarios, rotation=45, ha='right', fontsize=9)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # Lưu file hoặc show tùy môi trường của bạn
    plt.savefig('worst_predictions_loss.png') 
    plt.show()
    
    return worst_scenarios, worst_losses