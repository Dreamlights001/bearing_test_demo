import torch
import torch.nn as nn
import torch.nn.functional as F

class BearingCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1, sequence_length=1024):
        super(BearingCNN, self).__init__()
        """
        Args:
            num_classes (int): 输出类别的数量。
            input_channels (int): 输入数据的通道数 (对于原始振动信号通常是1)。
            sequence_length (int): 输入序列的长度。
        """
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.sequence_length = sequence_length

        # 卷积层
        # Conv1: (B, 1, 1024) -> (B, 16, 512) after conv and pool
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=64, stride=2, padding=31)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv2: (B, 16, 512) -> (B, 32, 256) after conv and pool
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=2, padding=15)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv3: (B, 32, 256) -> (B, 64, 128) after conv and pool
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=2, padding=7)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) 
        
        # Conv4: (B, 64, 128) -> (B, 128, 64) after conv and pool
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=3)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) # Output: (B, 128, 32)

        # 计算展平后的特征数量
        # 初始长度 1024
        # conv1: (1024 - 64 + 2*31) / 2 + 1 = 1022 / 2 + 1 = 512 (wrong, (L_in + 2*padding - kernel_size)/stride + 1)
        # (1024 + 2*31 - 64)/2 + 1 = (1024 + 62 - 64)/2 + 1 = 1022/2 + 1 = 511 + 1 = 512. Correct.
        # pool1: 512 / 2 = 256
        # conv2: (256 + 2*15 - 32)/2 + 1 = (256 + 30 - 32)/2 + 1 = 254/2 + 1 = 127 + 1 = 128
        # pool2: 128 / 2 = 64
        # conv3: (64 + 2*7 - 16)/2 + 1 = (64 + 14 - 16)/2 + 1 = 62/2 + 1 = 31 + 1 = 32
        # pool3: 32 / 2 = 16
        # conv4: (16 + 2*3 - 8)/2 + 1 = (16 + 6 - 8)/2 + 1 = 14/2 + 1 = 7 + 1 = 8
        # pool4: 8 / 2 = 4
        # 所以最终序列长度是 4
        self.flattened_features = 128 * 4 

        # 全连接层
        self.fc1 = nn.Linear(self.flattened_features, 100)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Dropout层，防止过拟合
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_channels, sequence_length)
        # e.g., (B, 1, 1024)
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        # print(f"After conv1 pool1: {x.shape}") # Expected: (B, 16, 256)
        
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # print(f"After conv2 pool2: {x.shape}") # Expected: (B, 32, 64)
        
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        # print(f"After conv3 pool3: {x.shape}") # Expected: (B, 64, 16)

        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        # print(f"After conv4 pool4: {x.shape}") # Expected: (B, 128, 4)
        
        # 展平操作
        x = x.view(-1, self.flattened_features)
        # print(f"After flatten: {x.shape}") # Expected: (B, 128*4)
        
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    # 测试模型
    num_classes_test = 5 # 假设有5个类别
    batch_size_test = 4
    seq_len_test = 1024
    input_channels_test = 1

    # 创建模型实例
    model = BearingCNN(num_classes=num_classes_test, input_channels=input_channels_test, sequence_length=seq_len_test)
    print(model)

    # 创建一个虚拟输入张量
    dummy_input = torch.randn(batch_size_test, input_channels_test, seq_len_test)
    print(f"\nInput shape: {dummy_input.shape}")

    # 前向传播
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}") # Expected: (batch_size_test, num_classes_test)
        assert output.shape == (batch_size_test, num_classes_test)
        print("Model forward pass test successful!")
    except Exception as e:
        print(f"Error during model forward pass: {e}")

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")