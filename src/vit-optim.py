# vit_cifar100_optimizers_compare_tqdm.py
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============ Reproducibility ============
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============ Model: Tiny Vision Transformer ============
class PatchEmbed(nn.Module):
    """Split image into patches and project to embeddings using Conv2d."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=attn_drop, batch_first=True)
        self.drop1 = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, dim),
        )
        self.drop2 = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x_res + self.drop_path(self.drop1(attn_out))
        x_res = x
        x = self.norm2(x)
        x = x_res + self.drop_path(self.drop2(self.mlp(x)))
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x / keep_prob * random_tensor


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=192, depth=8, num_heads=3, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, drop_path_rate=0.0,
                 use_cls_token=True, global_pool="cls"):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.use_cls_token = use_cls_token
        self.global_pool = global_pool

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            seq_len = num_patches + 1
        else:
            self.cls_token = None
            seq_len = num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=proj_drop)

        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.use_cls_token:
            cls_tok = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tok, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feat = x[:, 0] if self.use_cls_token else x.mean(dim=1)
        return self.head(feat)


# ============ Data ============
def get_dataloaders(batch_size: int = 128, num_workers: int = 2):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=test_tf)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    return running_loss / len(loader.dataset)


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    num_workers: int = 2
    img_size: int = 32
    patch: int = 4
    embed_dim: int = 192
    depth: int = 8
    heads: int = 3
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    lr_sgd: float = 0.01
    lr_sgdm: float = 0.01
    lr_adamw: float = 3e-4
    weight_decay_sgd: float = 5e-4
    weight_decay_adamw: float = 0.05
    momentum: float = 0.9
    seed: int = 42


def make_model(cfg):
    return ViT(
        img_size=cfg.img_size, patch_size=cfg.patch, in_chans=3, num_classes=100,
        embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.heads,
        mlp_ratio=cfg.mlp_ratio, proj_drop=0.1, drop_path_rate=cfg.drop_path_rate,
        use_cls_token=True, global_pool="cls"
    )


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    train_loader, test_loader = get_dataloaders(cfg.batch_size, cfg.num_workers)

    setups = {
        "SGD": {
            "model": make_model(cfg).to(device),
            "optimizer": lambda params: torch.optim.SGD(params, lr=cfg.lr_sgd,
                                                        weight_decay=cfg.weight_decay_sgd, momentum=0.0),
        },
        "SGD+Momentum": {
            "model": make_model(cfg).to(device),
            "optimizer": lambda params: torch.optim.SGD(params, lr=cfg.lr_sgdm, momentum=cfg.momentum,
                                                        weight_decay=cfg.weight_decay_sgd),
        },
        "AdamW": {
            "model": make_model(cfg).to(device),
            "optimizer": lambda params: torch.optim.AdamW(params, lr=cfg.lr_adamw,
                                                          weight_decay=cfg.weight_decay_adamw),
        },
    }

    criterion = nn.CrossEntropyLoss()
    history = {k: [] for k in setups}

    for name, pack in setups.items():
        print(f"\n===== Training with {name} =====")
        model = pack["model"]
        optimizer = pack["optimizer"](model.parameters())

        accs = [evaluate(model, test_loader, device)]
        print(f"Epoch 0 | Test Acc: {accs[-1]:.2f}%")

        for epoch in range(1, cfg.epochs + 1):
            print(f"\nEpoch {epoch}/{cfg.epochs} ({name})")
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            acc = evaluate(model, test_loader, device)
            t1 = time.time()
            accs.append(acc)
            print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Test Acc: {acc:.2f}% | Time: {t1-t0:.1f}s")

        history[name] = accs

    # 绘图保存（不显示）
    epochs_axis = list(range(cfg.epochs + 1))
    plt.figure(figsize=(8, 5), dpi=140)
    for name, accs in history.items():
        plt.plot(epochs_axis, accs, marker="o", label=name)
    plt.title("CIFAR-100: ViT Test Accuracy vs Epochs (Optimizers)")
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("vit_cifar100_optim_compare.png")
    print("✅ Plot saved to vit_cifar100_optim_compare.png")


if __name__ == "__main__":
    main()
