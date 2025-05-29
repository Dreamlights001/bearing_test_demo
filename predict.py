import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from data_loader import create_dataloaders, INV_FAULT_TYPE_MAP
from model import BearingCNN

# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, 'bearingset', 'test_set')
# TRAIN_DATA_DIR is needed by create_dataloaders, even if not used for training here
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'bearingset', 'train_set') 
MODEL_LOAD_PATH = os.path.join(BASE_DIR, 'bearing_cnn_model.pth')
RESULTS_DIR = os.path.join(BASE_DIR, 'results') # 目录用于存放结果图表
SAMPLE_PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'sample_predictions') # 目录用于存放每个样本的预测结果
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, 'classification_report.txt')

BATCH_SIZE = 64 # 应该与训练时相似，或者根据评估需求调整
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader, device, inv_fault_map):
    """在测试集上评估模型，并返回详细的预测结果。"""
    model.eval() # 设置模型为评估模式
    all_preds_indices = []
    all_labels_indices = []
    all_filenames = []
    all_pred_names = []

    print(f"\nStarting evaluation on {device}...")
    progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)

    with torch.no_grad(): # 在评估过程中不计算梯度
        for inputs, labels, filenames_batch in progress_bar: # 假设dataloader现在返回filenames
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted_indices = torch.max(outputs.data, 1)
            
            all_preds_indices.extend(predicted_indices.cpu().numpy())
            all_labels_indices.extend(labels.cpu().numpy())
            all_filenames.extend(filenames_batch)
            
            # 将预测索引转换为类别名称
            for pred_idx in predicted_indices.cpu().numpy():
                all_pred_names.append(inv_fault_map[pred_idx])
            
            # 计算当前批次的准确率以更新进度条 (可选)
            batch_acc = accuracy_score(labels.cpu().numpy(), predicted_indices.cpu().numpy())
            progress_bar.set_postfix(batch_acc=f"{batch_acc:.3f}")

    progress_bar.close()
    print("Evaluation finished.")
    return np.array(all_labels_indices), np.array(all_preds_indices), all_filenames, all_pred_names

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    # plt.show()

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    # 创建结果目录 (如果不存在)
    for dir_path in [RESULTS_DIR, SAMPLE_PREDICTIONS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    # 检查数据和模型文件是否存在
    if not os.path.isdir(TEST_DATA_DIR):
        print(f"Error: Test data directory not found: {TEST_DATA_DIR}")
        exit()
    if not os.path.isdir(TRAIN_DATA_DIR): # 需要它来初始化dataloader
        print(f"Error: Train data directory not found: {TRAIN_DATA_DIR}. It's needed by data_loader.")
        exit()
    if not os.path.exists(MODEL_LOAD_PATH):
        print(f"Error: Model file not found: {MODEL_LOAD_PATH}")
        print("Please run the training script (train.py) first to generate the model file.")
        exit()

    # 1. 加载数据
    try:
        # 注意：create_dataloaders 返回 train_loader, test_loader, num_classes, inv_map
        # 我们这里主要关心 test_loader 和 num_classes, inv_map
        _, test_loader, num_classes, inv_fault_map = create_dataloaders(
            train_data_dir=TRAIN_DATA_DIR, # 即使不用于训练，也需要提供给函数
            test_data_dir=TEST_DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=0
        )
        print("Test data loaded successfully.")
        print(f"Number of classes: {num_classes}")
        print(f"Number of test batches: {len(test_loader)}")
        print(f"Total test samples: {len(test_loader.dataset)}")
        class_names = [inv_fault_map[i] for i in range(num_classes)]
        print(f"Class names: {class_names}")

    except RuntimeError as e:
        print(f"Error loading data: {e}")
        exit()
    except FileNotFoundError as e:
        print(f"FileNotFoundError during data loading: {e}")
        exit()

    # 2. 初始化模型并加载训练好的权重
    model = BearingCNN(num_classes=num_classes).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
        print(f"Model loaded from {MODEL_LOAD_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_LOAD_PATH}. Train the model first.")
        exit()
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        exit()

    # 3. 评估模型
    try:
        true_labels, pred_labels_indices, pred_filenames, pred_class_names = evaluate_model(model, test_loader, DEVICE, inv_fault_map)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        exit()

    # 将每个样本的预测结果保存到单独的文件
    # 创建一个字典来收集每个原始CSV文件的预测结果
    predictions_by_file = {}
    for fname, p_name in zip(pred_filenames, pred_class_names):
        original_csv_name = fname.split('.')[0] # 获取文件名（不含扩展名）
        if original_csv_name not in predictions_by_file:
            predictions_by_file[original_csv_name] = []
        predictions_by_file[original_csv_name].append(p_name)

    # 将收集到的预测结果写入文件
    for original_csv_name, predictions in predictions_by_file.items():
        output_filepath = os.path.join(SAMPLE_PREDICTIONS_DIR, f"{original_csv_name}_predictions.txt")
        try:
            with open(output_filepath, 'w') as f:
                for pred_item in predictions:
                    f.write(f"{pred_item}\n")
            print(f"Saved predictions for {original_csv_name} to {output_filepath}")
        except Exception as e:
            print(f"Error saving predictions for {original_csv_name}: {e}")


    # 4. 计算并打印性能指标
    accuracy = accuracy_score(true_labels, pred_labels_indices)
    report = classification_report(true_labels, pred_labels_indices, target_names=class_names, digits=4)

    print("\n--- Model Performance ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # 保存分类报告到文件
    try:
        with open(CLASSIFICATION_REPORT_PATH, 'w') as f:
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Classification report saved to {CLASSIFICATION_REPORT_PATH}")
    except Exception as e:
        print(f"Error saving classification report: {e}")

    # 5. 绘制并保存混淆矩阵
    try:
        plot_confusion_matrix(true_labels, pred_labels_indices, class_names, CONFUSION_MATRIX_PATH)
    except Exception as e:
        print(f"Error plotting/saving confusion matrix: {e}")


    print("\nPrediction script finished.")

    # 提示用户如何查看结果
    print(f"\nResults have been saved in the '{RESULTS_DIR}' directory:")
    print(f"- Confusion Matrix: {CONFUSION_MATRIX_PATH}")
    print(f"- Classification Report: {CLASSIFICATION_REPORT_PATH}")
    print(f"- Individual Sample Predictions: {SAMPLE_PREDICTIONS_DIR}")