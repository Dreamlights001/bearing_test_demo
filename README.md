# 基于CNN的轴承故障诊断

本项目使用卷积神经网络（CNN）对轴承的振动数据进行故障诊断。数据集包含不同直径轴承在健康、滚珠故障、内圈故障、外圈故障和组合故障等情况下的振动信号。

## 项目结构

```
故障诊断/
├── bearingset/                  # 数据集根目录
│   ├── test_set/                # 测试数据集
│   │   ├── ball20_0_test.csv
│   │   ├── ... (其他测试csv文件)
│   └── train_set/               # 训练数据集
│       ├── ball20_0_train.csv
│       ├── ... (其他训练csv文件)
├── data_loader.py             # 数据加载和预处理模块
├── model.py                   # CNN模型定义模块
├── train.py                   # 模型训练脚本
├── predict.py                 # 模型预测与评估脚本
├── requirements.txt           # Python依赖包列表
├── bearing_cnn_model.pth      # (训练后生成) 训练好的模型权重
├── training_loss_accuracy.png # (训练后生成) 训练过程图表
├── results/                   # (预测后生成) 存放评估结果的目录
│   ├── sample_predictions/    # (预测后生成) 存放每个测试CSV文件各样本的预测结果
│   │   ├── ball20_0_test_predictions.txt
│   │   └── ... (其他CSV对应的预测结果txt文件)
│   ├── confusion_matrix.png   # (预测后生成) 混淆矩阵图
│   └── classification_report.txt # (预测后生成) 分类报告
└── README.md                  # 项目说明文件
```

## 数据说明

- 数据集位于 `bearingset` 文件夹下，分为 `train_set` 和 `test_set`。
- 每个CSV文件代表一种特定工况下的轴承数据。
- 文件名中包含了故障类型（如 `health`, `ball`, `inner`, `outer`, `comb`）和轴承直径（如 `20`, `30`）。
- 每个CSV文件包含100列数据，每列代表一个振动信号样本。
- 每个样本（即每列）包含1024个数据点，表示随时间变化的振幅。

## 环境要求

- Python 3.7+
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Tqdm (用于显示进度条)

## 安装依赖

在项目根目录下（即 `故障诊断` 目录），打开终端或命令行，运行以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 训练模型

在项目根目录下运行训练脚本：

```bash
python train.py
```

训练过程会在控制台输出每个epoch的损失和准确率。训练完成后，会执行以下操作：
- 保存训练好的模型权重到 `bearing_cnn_model.pth`。
- 保存训练过程中的损失和准确率变化图到 `training_loss_accuracy.png`。

**注意：**
- 训练脚本会自动检测可用的设备（CPU或CUDA GPU）。如果您的机器上有兼容的NVIDIA GPU并且正确安装了CUDA和PyTorch的GPU版本，训练将在GPU上进行，速度会更快。
- 训练参数（如epoch数量、学习率、批次大小）可以在 `train.py` 文件顶部进行修改。

### 2. 进行预测和评估

训练完成后，运行预测脚本：

```bash
python predict.py
```

该脚本会执行以下操作：
- 加载 `bearing_cnn_model.pth` 中保存的模型权重。
- 在 `bearingset/test_set` 中的测试数据上进行预测。
- 计算并打印模型的整体准确率和详细的分类报告（精确率、召回率、F1分数）。
- 生成并保存在 `results/` 目录下的评估结果：
    - `confusion_matrix.png`: 混淆矩阵图。
    - `classification_report.txt`: 详细的分类性能报告。
    - `sample_predictions/` 目录：包含对测试集中每个原始CSV文件的所有样本的预测结果。每个CSV文件会对应一个 `_predictions.txt` 文件，其中每行是该CSV文件中一个样本的预测故障类型。

## 文件说明

- `data_loader.py`: 负责从CSV文件中加载数据，进行必要的预处理，并创建PyTorch的 `DataLoader` 以便模型进行批处理训练和测试。
- `model.py`: 定义了1D卷积神经网络（CNN）的结构。网络包含多个卷积层、批归一化层、ReLU激活函数、池化层以及全连接层，用于从振动信号中提取特征并进行分类。
- `train.py`: 包含了模型训练的完整流程。它会加载训练数据，初始化模型、损失函数（交叉熵损失）和优化器（Adam），然后进行指定epoch数量的训练。训练过程中会记录损失和准确率，并在训练结束后保存模型权重和训练历史图。
- `predict.py`: 用于在测试集上评估已训练模型的性能。它加载保存的模型权重和测试数据，进行预测，然后计算并展示多种性能指标，包括准确率、分类报告（精确率、召回率、F1分数）和混淆矩阵。评估结果会保存到文件中。
- `requirements.txt`: 列出了项目运行所需的所有Python库及其版本（如果指定）。

## 注意事项

- 请确保数据集文件 (`.csv`) 存放在正确的路径下 (`bearingset/train_set` 和 `bearingset/test_set`)。
- 如果CSV文件的格式与描述不符（例如，数据不是1024行或列数不为100），`data_loader.py` 中的加载逻辑可能需要调整。
- 模型的超参数（如卷积核大小、层数、全连接层大小等）和训练参数（学习率、批大小、epoch数）可以根据实际情况在相应 `.py` 文件中进行调整以获得更好的性能。