import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 下载莎士比亚数据集
if not os.path.exists('shakespeare.txt'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, 'shakespeare.txt')

# 读取数据
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"文本长度: {len(text)} 字符")
print(text[:500])  # 打印前500个字符预览

# 创建字符到索引和索引到字符的映射
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"共有 {vocab_size} 个唯一字符")
print("字符集:", ''.join(chars))

# 将整个文本转换为索引
data = [char_to_idx[ch] for ch in text]

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        # 嵌入层将字符索引转换为密集向量
        self.embedding = nn.Embedding(input_size, hidden_size)
        # RNN层
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, hidden):
        # 嵌入层
        x = self.embedding(x)
        # RNN层
        out, hidden = self.rnn(x, hidden)
        # 全连接层
        out = self.fc(out)
        return out, hidden
    def init_hidden(self, batch_size):
        # 初始化隐藏状态
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

# 超参数
hidden_size = 128
seq_length = 100  # 每个训练序列的长度
batch_size = 64
learning_rate = 0.005
num_epochs = 20
print_every = 100  # 每100步打印一次

# 创建模型实例
model = CharRNN(vocab_size, hidden_size, vocab_size).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建训练批次
def get_batch(data, batch_size, seq_length):
    # 随机选择起始点
    starts = torch.randint(0, len(data) - seq_length - 1, (batch_size,))
    
    # 创建输入和目标序列
    inputs = torch.zeros((batch_size, seq_length), dtype=torch.long)
    targets = torch.zeros((batch_size, seq_length), dtype=torch.long)
    
    for i, start in enumerate(starts):
        inputs[i] = torch.tensor(data[start:start+seq_length])
        targets[i] = torch.tensor(data[start+1:start+seq_length+1])
    
    return inputs.to(device), targets.to(device)

def generate_text(model, char_to_idx, idx_to_char, device, start_str, length=1000, temperature=0.8):
    # 初始化隐藏状态
    hidden = model.init_hidden(1)
    
    # 处理起始字符串
    chars = [ch for ch in start_str]
    for ch in chars[:-1]:
        # 将字符转换为索引
        char_tensor = torch.tensor([[char_to_idx[ch]]], dtype=torch.long).to(device)
        # 前向传播但不保存输出
        _, hidden = model(char_tensor, hidden)
    
    # 最后一个字符作为输入
    input_char = torch.tensor([[char_to_idx[chars[-1]]]], dtype=torch.long).to(device)
    
    # 开始生成文本
    for _ in range(length):
        # 前向传播
        output, hidden = model(input_char, hidden)
        
        # 获取预测概率
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # 获取预测字符并添加到结果中
        predicted_char = idx_to_char[top_i.item()]
        chars.append(predicted_char)
        
        # 下一个输入是当前预测的字符
        input_char = torch.tensor([[top_i]], dtype=torch.long).to(device)
    
    return ''.join(chars)

print(f"\nEpoch {0} 生成文本:")
print(generate_text(model, char_to_idx, idx_to_char, device, "\n", 100))
print("-" * 50)

# 训练循环
for epoch in range(num_epochs):
    # 初始化隐藏状态
    hidden = model.init_hidden(batch_size)
    
    for step in tqdm(range(0, len(data) // (batch_size * seq_length))):
        # 获取批次数据
        inputs, targets = get_batch(data, batch_size, seq_length)
        
        # 前向传播
        hidden = hidden.detach()  # 断开与之前计算图的连接
        outputs, hidden = model(inputs, hidden)
        
        # 计算损失
        loss = criterion(outputs.transpose(1, 2), targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印训练信息
        if step % print_every == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step}], Loss: {loss.item():.4f}')
    
    # 每个epoch结束后生成一些文本看看效果
    print(f"\nEpoch {epoch+1} 生成文本:")
    print(generate_text(model, char_to_idx, idx_to_char, device, "\n", 100))
    print("-" * 50)

# 生成文本示例
start_string = "ROMEO:"
generated_text = generate_text(model, char_to_idx, idx_to_char, device, start_string, 100)
print("\n最终生成文本:")
print(generated_text)
start_string = "The meaning of life is:"
generated_text = generate_text(model, char_to_idx, idx_to_char, device, start_string, 100)
print("\n最终生成文本:")
print(generated_text)
