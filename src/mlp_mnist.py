# 此爲DeepSeek的手筆
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 超参数设置
batch_size = 64
learning_rate = 0.01
epochs = 10

# 数据加载
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# 定义模型（三层感知机）
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512),    # 第一层：输入层→隐藏层1
    nn.ReLU(),                # 激活函数
    nn.Linear(512, 256),       # 第二层：隐藏层1→隐藏层2
    nn.ReLU(),                # 激活函数
    nn.Linear(256, 10)        # 第三层：隐藏层2→输出层
)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数总数: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(epochs):
    # 训练阶段
    model.train()
    print(f"开始第{epoch+1}轮训练")
    for images, labels in tqdm(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 测试阶段
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch [{epoch+1}/{epochs}], 准确率: {100*correct/total:.2f}%")
