import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import create_dataloaders, NUM_CLASSES
from model import BearingCNN

# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'bearingset', 'train_set')
TEST_DATA_DIR = os.path.join(BASE_DIR, 'bearingset', 'test_set') # 虽然主要用于训练，但加载器会创建它
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'bearing_cnn_model.pth')
PLOT_SAVE_PATH = os.path.join(BASE_DIR, 'training_loss_accuracy.png')

# 训练参数
NUM_EPOCHS = 50  # 可以根据需要调整
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """训练模型"""
    model.train() # 设置模型为训练模式
    epoch_losses = []
    epoch_accuracies = []

    print(f"\nStarting training on {device}...")

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for inputs, labels, _ in progress_bar: # Modified to unpack filename
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # 更新进度条的后缀信息
            progress_bar.set_postfix(loss=loss.item(), acc=correct_predictions/total_samples if total_samples > 0 else 0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        progress_bar.close()

    print("Training finished.")
    return epoch_losses, epoch_accuracies

def plot_training_history(losses, accuracies, save_path):
    """绘制训练过程中的损失和准确率变化"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    # plt.show() # 如果希望在脚本执行时显示图像，取消注释此行

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")

    # 检查数据目录是否存在
    if not os.path.isdir(TRAIN_DATA_DIR):
        print(f"Error: Training data directory not found: {TRAIN_DATA_DIR}")
        print("Please ensure the 'bearingset/train_set' directory exists in the same directory as this script.")
        exit()
    if not os.path.isdir(TEST_DATA_DIR):
        print(f"Warning: Test data directory not found: {TEST_DATA_DIR}. This script primarily trains, but data_loader might expect it.")
        # 通常训练脚本也可能需要测试集进行验证，这里仅作提示

    # 1. 加载数据
    try:
        train_loader, _, num_classes_from_loader, _ = create_dataloaders(
            train_data_dir=TRAIN_DATA_DIR,
            test_data_dir=TEST_DATA_DIR, # 即使不直接用测试集，也传入以保持接口一致性
            batch_size=BATCH_SIZE,
            num_workers=0 # Windows下多进程DataLoader有时会有问题，设为0以避免
        )
        print(f"Data loaded successfully. Number of classes: {num_classes_from_loader}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Total training samples: {len(train_loader.dataset)}")

    except RuntimeError as e:
        print(f"Error loading data: {e}")
        print("Please check your data directories and CSV file formats.")
        exit()
    except FileNotFoundError as e:
        print(f"FileNotFoundError during data loading: {e}")
        exit()

    # 2. 初始化模型
    # 使用从data_loader中获取的类别数，而不是硬编码的NUM_CLASSES，以确保一致性
    model = BearingCNN(num_classes=num_classes_from_loader).to(DEVICE)
    print("Model initialized.")

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Loss function and optimizer defined.")

    # 4. 训练模型
    try:
        training_losses, training_accuracies = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, DEVICE)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        exit()

    # 5. 保存模型
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # 6. 绘制并保存训练历史
    try:
        plot_training_history(training_losses, training_accuracies, PLOT_SAVE_PATH)
    except Exception as e:
        print(f"Error plotting training history: {e}")

    print("\nTraining script finished.")