import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class ToySeqDataset(Dataset):
    def __init__(self, num_samples=2000, seq_len=16, vocab_size=100):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = (self.data[:, 0] % 2 == 0).long()  # 첫 토큰이 짝수면 1

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_ff=256, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos_enc(self.embed(x))
        enc = self.encoder(x)
        pooled = enc[:, 0]  # 첫 토큰을 CLS처럼 사용
        return self.fc(pooled)

def train_one_epoch(model, loader, optim, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total / len(loader.dataset), correct / len(loader.dataset)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ToySeqDataset(num_samples=2000)
    val_ds = ToySeqDataset(num_samples=500)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = TransformerClassifier(vocab_size=100).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, 4):
        tr_loss = train_one_epoch(model, train_loader, optim, device)
        val_loss, val_acc = eval_model(model, val_loader, device)
        print(f"[{epoch}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
