import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """初始化判别器模型"""
        super(Discriminator, self).__init__()

        # 定义MLP层
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(hidden_dim, 1)  # 隐藏层到输出层

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.relu(self.fc1(x))  # 通过第一层并使用ReLU激活
        x = self.relu(self.fc2(x))  # 通过第二层并使用ReLU激活
        x = self.fc3(x)  # 通过输出层
        return self.sigmoid(x)  # 使用Sigmoid激活函数输出概率值
