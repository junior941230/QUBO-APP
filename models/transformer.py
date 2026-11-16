"""Sliding-window Transformer baseline on per-epoch feature sequences.

Each EDF is one sequence of per-epoch feature vectors, as for the LSTM. Every
epoch attends only to the previous ``window`` epochs (or the surrounding ones
when ``causal=False``), so a 4-hour EDF never needs a full 14,400 x 14,400
attention matrix. A short depthwise convolution supplies local order instead of
an absolute positional encoding, so an epoch's output depends only on its
neighbourhood, not on where a training segment or inference chunk starts.
Training splits EDFs into fixed-length segments; inference runs in chunks with
enough surrounding context that the result equals a single full-sequence pass.
"""

import math

import numpy as np

from config import RANDOM_SEED
from core.logging_utils import log_step

# Importing the LSTM module also sets CUBLAS_WORKSPACE_CONFIG before CUDA starts.
from .lstm import TORCH_AVAILABLE, SeqDataset, _set_lstm_seed, collate_pad, nn, torch

INFERENCE_CHUNK = 2048
POS_WEIGHT_CAP = 20.0  # Same cap as the LSTM so the two baselines stay comparable.


def _blocked_attention(length, window, causal, valid=None, device=None):
    """Boolean mask, True where a query may not attend to a key.

    Shape (T, T), or (B, 1, T, T) when ``valid`` (B, T) marks padded epochs.
    Every epoch may always attend to itself, so no row is fully blocked even
    for padding.
    """
    index = torch.arange(length, device=device)
    offset = index[:, None] - index[None, :]  # query - key
    if causal:
        blocked = (offset < 0) | (offset >= window)
    else:
        blocked = offset.abs() >= window
    if valid is None:
        return blocked
    blocked = blocked[None, None] | ~valid.bool()[:, None, None, :]
    return blocked & ~torch.eye(length, dtype=torch.bool, device=device)


class _WindowedSelfAttention(nn.Module if TORCH_AVAILABLE else object):
    """Plain matmul attention, deterministic under torch.use_deterministic_algorithms."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, blocked):
        batch, length, d_model = x.shape
        q, k, v = (
            self.qkv(x)
            .view(batch, length, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores.masked_fill(blocked, float("-inf")), dim=-1)
        attended = self.dropout(weights) @ v
        return self.out(attended.transpose(1, 2).reshape(batch, length, d_model))


class _TransformerBlock(nn.Module if TORCH_AVAILABLE else object):
    """Pre-norm block: windowed self-attention then a feed-forward layer."""

    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = _WindowedSelfAttention(d_model, n_heads, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, blocked):
        x = x + self.dropout(self.attn(self.attn_norm(x), blocked))
        return x + self.dropout(self.ff(self.ff_norm(x)))


class SeizureTransformer(nn.Module if TORCH_AVAILABLE else object):
    """
    Per-epoch binary classifier.
    Input : (batch, seq_len, feat_dim)
    Output: (batch, seq_len)  -- logits, one per epoch
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.1,
        window=120,
        conv_kernel=5,
        causal=True,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not causal and conv_kernel % 2 == 0:
            raise ValueError("Non-causal mode needs an odd conv_kernel")
        self.window = int(window)
        self.conv_kernel = int(conv_kernel)
        self.num_layers = int(num_layers)
        self.causal = bool(causal)
        self.input = nn.Linear(input_dim, d_model)
        self.conv = nn.Conv1d(d_model, d_model, self.conv_kernel, groups=d_model)
        self.blocks = nn.ModuleList(
            _TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(self.num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _conv_padding(self):
        if self.causal:
            return self.conv_kernel - 1, 0
        half = (self.conv_kernel - 1) // 2
        return half, half

    @property
    def context(self):
        """Epochs (before, after) an output depends on: the exact chunking margin."""
        reach = self.num_layers * (self.window - 1)
        before, after = self._conv_padding()
        return before + reach, (after + reach) if not self.causal else 0

    def forward(self, x, valid=None):
        h = self.input(x)
        if valid is not None:
            # Padded epochs look like the zero padding at a real sequence edge.
            h = h * valid.unsqueeze(-1).to(h.dtype)
        h = h + self.conv(
            nn.functional.pad(h.transpose(1, 2), self._conv_padding())
        ).transpose(1, 2)
        blocked = _blocked_attention(
            x.shape[1], self.window, self.causal, valid=valid, device=x.device
        )
        for block in self.blocks:
            h = block(h, blocked)
        return self.head(self.norm(h)).squeeze(-1)


def _split_segments(seqs, labs, segment_length):
    """Cut every training EDF into consecutive segments (the last may be shorter)."""
    segments, segment_labels = [], []
    for seq, lab in zip(seqs, labs):
        for start in range(0, len(seq), segment_length):
            segments.append(seq[start:start + segment_length])
            segment_labels.append(lab[start:start + segment_length])
    return segments, segment_labels


def _train_transformer_on_files(
    train_files, features, labels,
    d_model=64, n_heads=4, num_layers=2, ff_dim=128, dropout=0.1,
    window_seconds=120, conv_kernel=5, causal=True, segment_seconds=600,
    epochs=20, lr=5e-4, weight_decay=0.01, batch_size=32,
    device=None, random_seed=RANDOM_SEED,
):
    """Train on fixed-length segments of the training EDFs (one epoch = one second)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")

    _set_lstm_seed(random_seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log_step(f"[Transformer] device={device}")

    seqs = [np.asarray(features[f], dtype=np.float32) for f in train_files]
    labs = [np.asarray(labels[f], dtype=np.float32) for f in train_files]

    # Standardize using train-set global stats, as the LSTM does.
    all_feat = np.concatenate(seqs, axis=0)
    mean = all_feat.mean(axis=0, keepdims=True)
    std = all_feat.std(axis=0, keepdims=True) + 1e-6
    seqs = [(s - mean) / std for s in seqs]

    all_y = np.concatenate(labs)
    pos = float(all_y.sum())
    raw_ratio = (len(all_y) - pos) / max(pos, 1.0)
    clipped = float(np.clip(raw_ratio, 1.0, POS_WEIGHT_CAP))
    pos_weight = torch.tensor([clipped], device=device)
    log_step(f"[Transformer] pos_weight raw={raw_ratio:.1f} -> clipped={clipped}")

    segments, segment_labels = _split_segments(seqs, labs, int(segment_seconds))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(random_seed))
    loader = torch.utils.data.DataLoader(
        SeqDataset(segments, segment_labels), batch_size=batch_size, shuffle=True,
        collate_fn=collate_pad, drop_last=False, generator=loader_generator,
    )

    model = SeizureTransformer(
        input_dim=seqs[0].shape[1],
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        window=int(window_seconds),
        conv_kernel=conv_kernel,
        causal=causal,
    ).to(device)
    log_step(
        f"[Transformer] segments={len(segments)} "
        f"params={sum(p.numel() for p in model.parameters())}"
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for ep in range(epochs):
        total_loss, total_count = 0.0, 0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x, valid=mask)
            loss = (criterion(logits, y) * mask).sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_count += mask.sum().item()
        if (ep + 1) % max(1, epochs // 3) == 0:
            log_step(
                f"[Transformer] epoch {ep+1}/{epochs} "
                f"loss={total_loss/max(total_count, 1):.4f}"
            )

    return model, mean, std, device


def _predict_transformer_sequence(
    model, seq, mean, std, device, chunk=INFERENCE_CHUNK
):
    """Per-epoch probabilities for one EDF, computed chunk by chunk.

    Each chunk carries ``model.context`` epochs of surrounding input, so the
    result equals one pass over the whole sequence while memory stays bounded.
    """
    model.eval()
    x = torch.from_numpy(
        np.asarray((np.asarray(seq, dtype=np.float32) - mean) / std, dtype=np.float32)
    ).to(device)
    length = x.shape[0]
    before, after = model.context
    logits = []
    with torch.no_grad():
        for start in range(0, length, chunk):
            stop = min(length, start + chunk)
            lo, hi = max(0, start - before), min(length, stop + after)
            out = model(x[lo:hi].unsqueeze(0)).squeeze(0)
            logits.append(out[start - lo:stop - lo])
    probs = torch.sigmoid(torch.cat(logits)).cpu().numpy()
    log_step(
        f"[Transformer] pred stats: min={probs.min():.3f} "
        f"max={probs.max():.3f} mean={probs.mean():.3f} "
        f"p>0.5={np.mean(probs > 0.5):.3f}"
    )
    return probs


def predict_transformer(
    train_files,
    test_file,
    features,
    labels,
    transformer_params=None,
    random_seed=RANDOM_SEED,
):
    if train_files is None or test_file is None or features is None or labels is None:
        raise ValueError(
            "Transformer mode needs train_files, test_file, features, labels"
        )
    model, mean, std, device = _train_transformer_on_files(
        train_files, features, labels, random_seed=random_seed,
        **(transformer_params or {}),
    )
    return _predict_transformer_sequence(model, features[test_file], mean, std, device)
