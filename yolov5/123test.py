import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# 1. 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = self.fc2(x)  # 输出层
        return x


# 2. 数据准备（这里使用随机数据作为示例）
input_size = 20  # 输入特征的大小
hidden_size = 50  # 隐藏层的大小
output_size = 3  # 输出类别数量

# 生成随机数据作为输入和标签
X_train = np.random.rand(100, input_size).astype(np.float32)  # 100个样本，每个样本20个特征
y_train = np.random.randint(0, output_size, 100)  # 100个标签（假设有3个类别）

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 3. 初始化模型、损失函数和优化器
model = MLP(input_size, hidden_size, output_size)  # 创建模型
criterion = nn.CrossEntropyLoss()  # 分类问题使用交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

# 4. 训练模型
num_epochs = 10  # 训练轮数

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0

    # 迭代训练数据
    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

        running_loss += loss.item()

    # 打印每轮的平均损失
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 5. 测试模型（这里用训练数据做测试示例）
model.eval()  # 设置为评估模式
with torch.no_grad():  # 评估时不需要计算梯度
    outputs = model(X_train_tensor)
    _, predicted = torch.max(outputs, 1)  # 获取预测的类别
    accuracy = (predicted == y_train_tensor).sum().item() / len(y_train_tensor)  # 计算准确率
    print(f"Accuracy: {accuracy * 100:.2f}%")