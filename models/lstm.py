import os
import random

# CUDA requires this workspace configuration for deterministic matrix products.
# It must be present before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import numpy as np
from config import RANDOM_SEED
from core.logging_utils import log_step


class SeizureLSTM(nn.Module):
    """
    Per-epoch binary classifier.
    Input : (batch, seq_len, feat_dim)
    Output: (batch, seq_len)  -- logits, one per epoch
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)              # (B, T, 2H)
        logits = self.head(out).squeeze(-1)  # (B, T)
        return logits


class SeqDataset(Dataset):
    """Each item = one EDF file as one sequence."""
    def __init__(self, seqs, labels):
        self.seqs = seqs        # list of np.ndarray (T_i, F)
        self.labels = labels    # list of np.ndarray (T_i,)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.seqs[idx]).float(),
            torch.from_numpy(self.labels[idx]).float(),
        )


def collate_pad(batch):
    """Pad variable-length sequences; return seqs, labels, mask."""
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    feat_dim = seqs[0].shape[1]

    padded = torch.zeros(len(seqs), max_len, feat_dim)
    lab_pad = torch.zeros(len(seqs), max_len)
    mask = torch.zeros(len(seqs), max_len)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        t = s.shape[0]
        padded[i, :t] = s
        lab_pad[i, :t] = l
        mask[i, :t] = 1.0
    return padded, lab_pad, mask


def _set_lstm_seed(seed):
    """Configure all RNGs used by LSTM initialization and training."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _train_lstm_on_files(
    train_files, features, labels,
    hidden_dim=128, num_layers=2, dropout=0.3,
    epochs=15, lr=1e-3, batch_size=4,
    device=None, random_seed=RANDOM_SEED,
):
    """Train LSTM treating each file as one sequence."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    _set_lstm_seed(random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    seqs   = [np.asarray(features[f], dtype=np.float32) for f in train_files]
    labs   = [np.asarray(labels[f], dtype=np.float32)    for f in train_files]

    # Standardize using train-set global stats
    all_feat = np.concatenate(seqs, axis=0)
    mean = all_feat.mean(axis=0, keepdims=True)
    std  = all_feat.std(axis=0, keepdims=True) + 1e-6
    seqs = [(s - mean) / std for s in seqs]

    # Class weight for imbalance
    all_y = np.concatenate(labs)
    pos = float(all_y.sum())
    neg = float(len(all_y) - pos)
    raw_ratio = neg / max(pos, 1.0)
    clipped = float(np.clip(raw_ratio, 1.0, 20.0))   # 上限 20
    pos_weight = torch.tensor([clipped], device=device)
    log_step(f"[LSTM] pos_weight raw={raw_ratio:.1f} → clipped={clipped}")
    # pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    ds = SeqDataset(seqs, labs)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(random_seed))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_pad, drop_last=False, generator=loader_generator,
    )

    model = SeizureLSTM(
        input_dim=seqs[0].shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        total_loss, total_count = 0.0, 0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss_raw = criterion(logits, y)
            loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_count += mask.sum().item()
        if (ep + 1) % max(1, epochs // 3) == 0:
            log_step(f"[LSTM] epoch {ep+1}/{epochs} loss={total_loss/max(total_count,1):.4f}")

    return model, mean, std, device


def _predict_lstm_sequence(model, seq, mean, std, device):
    """Predict per-epoch probability for one sequence."""
    model.eval()
    x = (np.asarray(seq, dtype=np.float32) - mean) / std
    x = torch.from_numpy(x).unsqueeze(0).to(device)   # (1, T, F)
    with torch.no_grad():
        logits = model(x).squeeze(0)                  # (T,)
        probs = torch.sigmoid(logits).cpu().numpy()
    log_step(
        f"[LSTM] pred stats: min={probs.min():.3f} "
        f"max={probs.max():.3f} mean={probs.mean():.3f} "
        f"p>0.5={np.mean(probs > 0.5):.3f}"
    )
    return probs

def predict_lstm(
    train_files,
    test_file,
    features,
    labels,
    lstm_params=None,
    random_seed=RANDOM_SEED,
):
    if train_files is None or test_file is None or features is None or labels is None:
        raise ValueError("LSTM mode needs train_files, test_file, features, labels")
    params = lstm_params or {}
    model, mean, std, device = _train_lstm_on_files(
        train_files, features, labels, random_seed=random_seed, **params,
    )
    return _predict_lstm_sequence(model, features[test_file], mean, std, device)
