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
epochs = 5

# 数据加载
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 16  # 初始通道数调整为更小的尺寸
        
        # 输入层（适应MNIST的1通道输入）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 28x28 -> 14x14
        
        # 残差块层
        self.layer1 = self._make_layer(16, 16, stride=1, blocks=2)
        self.layer2 = self._make_layer(16, 32, stride=2, blocks=2)
        self.layer3 = self._make_layer(32, 64, stride=2, blocks=2)
        self.layer4 = self._make_layer(64, 128, stride=2, blocks=2)
        
        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, stride, blocks):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)       # [B,1,28,28] -> [B,16,28,28]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B,16,14,14]
        
        x = self.layer1(x)      # [B,16,14,14]
        x = self.layer2(x)      # [B,32,7,7]
        x = self.layer3(x)      # [B,64,4,4]
        x = self.layer4(x)      # [B,128,2,2]
        
        x = self.avgpool(x)     # [B,128,1,1]
        x = torch.flatten(x, 1) # [B,128]
        x = self.fc(x)          # [B,10]
        return x

# 初始化模型
model = ResNet18()
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
