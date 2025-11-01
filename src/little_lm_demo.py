import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

# ==================== 手写Transformer实现 ====================

class SelfAttention(nn.Module):
    """单头自注意力机制"""
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(1000, 1000)))  # 因果掩码
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 计算注意力分数
        wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 因果掩码
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        
        # 加权聚合值
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    def __init__(self, embed_size, num_heads):
        super().__init__()
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList([SelfAttention(embed_size, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)  # 投影层
        
    def forward(self, x):
        # 拼接所有头的输出
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),  # 扩展维度
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),   # 投影回原维度
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer块: 自注意力 + 前馈网络"""
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size, num_heads)
        self.ff = FeedForward(embed_size)
        
    def forward(self, x):
        # 残差连接 + 层归一化
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    """三层Decoder-only的Transformer语言模型"""
    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(1000, embed_size)  # 位置编码
        self.blocks = nn.Sequential(
            *[Block(embed_size, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size)  # 最终层归一化
        self.lm_head = nn.Linear(embed_size, vocab_size)  # 语言模型头
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 获取token和位置嵌入
        tok_emb = self.token_embedding(idx)  # (B, T, embed_size)
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T, embed_size)
        x = tok_emb + pos_emb  # (B, T, embed_size)
        
        # 通过Transformer块
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # 计算交叉熵损失
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """生成文本"""
        for _ in range(max_new_tokens):
            # 裁剪输入以避免超出位置编码范围
            idx_cond = idx[:, -1000:]
            # 前向传播
            logits, _ = self(idx_cond)
            # 聚焦最后时间步
            logits = logits[:, -1, :] / temperature
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将采样结果拼接到序列中
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==================== 训练设置 ====================

# 超参数
embed_size = 128  # 嵌入维度
num_heads = 8     # 注意力头数
num_layers = 3    # Transformer层数
batch_size = 64   # 批量大小
seq_length = 256  # 序列长度
learning_rate = 3e-4
num_epochs = 50
eval_interval = 500
eval_iters = 200

# 创建模型实例
model = TransformerLM(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    num_layers=num_layers
).to(device)

# 打印模型参数数量
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 创建训练和验证集
n = int(0.9 * len(data))  # 90%用于训练
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """获取一个小批量数据"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([data[i:i+seq_length] for i in ix])
    y = torch.stack([data[i+1:i+seq_length+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    """评估模型在训练集和验证集上的损失"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ==================== 训练循环 ====================

for epoch in range(num_epochs):
    # 每轮开始评估一次
    losses = estimate_loss()
    print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # 初始化进度条
    pbar = tqdm(range(0, len(train_data) // (batch_size * seq_length)))
    
    for step in pbar:
        # 获取一批数据
        xb, yb = get_batch('train')
        
        # 前向传播
        _, loss = model(xb, yb)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # 更新进度条
        pbar.set_description(f"loss: {loss.item():.4f}")
        
        # 定期评估
        if step % eval_interval == 0 or step == len(pbar) - 1:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # 生成样本
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=200, temperature=0.8)
            print('生成文本:')
            print(''.join([idx_to_char[i] for i in generated[0].tolist()]))
            print('-' * 50)

# ==================== 生成示例 ====================

# 生成一些文本示例
contexts = [
    "\n",  # 从新行开始
    "ROMEO:",  # 莎士比亚风格
    "The meaning of life is",  # 哲学问题
    "To be or not to be",  # 经典台词
]

for context in contexts:
    # 将上下文转换为token
    ctx = torch.tensor([char_to_idx[c] for c in context], dtype=torch.long, device=device).unsqueeze(0)
    
    # 生成文本
    generated = model.generate(ctx, max_new_tokens=500, temperature=0.8)
    generated_text = ''.join([idx_to_char[i] for i in generated[0].tolist()])
    
    print(f"\n生成文本 (上下文: '{context}'):")
    print(generated_text)
    print("-" * 80)
