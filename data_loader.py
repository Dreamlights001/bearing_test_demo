import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# 定义故障类型到标签的映射
FAULT_TYPE_MAP = {
    'health': 0,
    'ball': 1,
    'inner': 2,
    'outer': 3,
    'comb': 4  # 组合故障，可以根据具体情况调整或细分
}

INV_FAULT_TYPE_MAP = {v: k for k, v in FAULT_TYPE_MAP.items()}
NUM_CLASSES = len(FAULT_TYPE_MAP)

def get_fault_type_from_filename(filename):
    """从文件名中提取故障类型"""
    for fault_key in FAULT_TYPE_MAP.keys():
        if fault_key in filename.lower():
            return fault_key
    raise ValueError(f"Unknown fault type in filename: {filename}")

class BearingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): 包含CSV文件的目录路径。
            transform (callable, optional): 应用于样本的可选转换。
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.filenames = []

        self._load_data()

    def _load_data(self):
        """加载所有CSV文件的数据和标签"""
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    fault_type_str = get_fault_type_from_filename(filename)
                    label = FAULT_TYPE_MAP[fault_type_str]
                except ValueError as e:
                    print(f"Skipping file {filename}: {e}")
                    continue

                # 读取CSV文件，每一列是一个样本
                # pandas默认将第一行作为header，如果数据从第一行开始，需要设置header=None
                # 假设数据从第一行开始，没有表头
                df = pd.read_csv(file_path, header=None)
                
                # 检查数据是否为1024行
                if df.shape[0] != 1024:
                    print(f"Skipping file {filename}: Expected 1024 rows, got {df.shape[0]}")
                    continue
                
                # 每一列是一个样本
                for col in df.columns:
                    sample_data = df[col].values.astype(np.float32) # (1024,)
                    self.samples.append(sample_data)
                    self.labels.append(label)
                    self.filenames.append(filename) # 主要用于调试或追溯
        
        if not self.samples:
            raise RuntimeError(f"No valid data found in {self.data_dir}")

        self.labels = np.array(self.labels, dtype=np.int64)
        # self.samples已经是list of numpy arrays

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        label = self.labels[idx]
        filename = self.filenames[idx] # 获取文件名

        # 将样本转换为 (1, 1024) 的形状，以适应1D CNN的输入 (channels, sequence_length)
        sample = sample.reshape(1, -1) 

        if self.transform:
            sample = self.transform(sample)
        
        # PyTorch期望标签是torch.long类型
        # 返回样本、标签和文件名
        return torch.from_numpy(sample.copy()), torch.tensor(label, dtype=torch.long), filename

def create_dataloaders(train_data_dir, test_data_dir, batch_size, num_workers=0):
    """创建训练和测试的DataLoader"""
    train_dataset = BearingDataset(data_dir=train_data_dir)
    test_dataset = BearingDataset(data_dir=test_data_dir)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, test_loader, NUM_CLASSES, INV_FAULT_TYPE_MAP

if __name__ == '__main__':
    # 示例用法和测试
    print(f"PyTorch version: {torch.__version__}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'bearingset', 'train_set')
    test_dir = os.path.join(base_dir, 'bearingset', 'test_set')

    print(f"Looking for training data in: {train_dir}")
    print(f"Looking for test data in: {test_dir}")

    # 检查目录是否存在
    if not os.path.isdir(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        exit()
    if not os.path.isdir(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        exit()

    try:
        train_loader, test_loader, num_classes, inv_map = create_dataloaders(
            train_data_dir=train_dir,
            test_data_dir=test_dir,
            batch_size=32
        )

        print(f"Number of classes: {num_classes}")
        print(f"Inverse fault map: {inv_map}")

        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of test samples: {len(test_loader.dataset)}")

        # 检查一个批次的数据
        print("\n--- Training Loader Sample Batch ---")
        for i, (data, labels, filenames_batch) in enumerate(train_loader):
            print(f"Batch {i+1}:")
            print(f"Data shape: {data.shape}")  # Expected: [batch_size, 1, 1024]
            print(f"Labels shape: {labels.shape}") # Expected: [batch_size]
            print(f"Filenames in batch (first 5): {filenames_batch[:5]}")
            print(f"Sample labels: {labels[:5]}")
            if i == 0: # 只检查第一个批次
                break
        
        print("\n--- Test Loader Sample Batch ---")
        for i, (data, labels, filenames_batch) in enumerate(test_loader):
            print(f"Batch {i+1}:")
            print(f"Data shape: {data.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Filenames in batch (first 5): {filenames_batch[:5]}")
            print(f"Sample labels: {labels[:5]}")
            if i == 0:
                break
        
        print("\nData loader created successfully.")

    except RuntimeError as e:
        print(f"RuntimeError during DataLoader creation or testing: {e}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")