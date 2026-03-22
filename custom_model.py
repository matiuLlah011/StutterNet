"""
Custom — Model Architecture (1D CNN + BiGRU + Attention)
Designed specifically for stuttering detection on our small dataset.

Why this architecture:
  - 1D CNN detects LOCAL temporal patterns (repeated syllables/words)
  - BiGRU captures LONG-RANGE dependencies (pauses, blocks)
  - Attention focuses on the most informative time regions
  - Only ~85K params (vs 722K in FluentNet) — less overfitting risk

Input: (batch, 220, 120) — MFCC sequence features
Output: (batch, 3) — class logits

Architecture:
  Conv1D(120→64, kernel=5) → BN → ReLU → MaxPool
  Conv1D(64→128, kernel=3) → BN → ReLU → MaxPool
  BiGRU(128→64 per direction = 128 total)
  Attention pooling (128→1 weights)
  Dense(128→64) → Dropout → Dense(64→3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Simple attention layer — learns which time steps matter most."""

    def __init__(self, feature_dim):
        super().__init__()
        # Two-layer attention: project to scalar score per timestep
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, features)
        Returns: (batch, features) — weighted sum over time
        """
        # Compute attention scores
        scores = self.attn(x).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum of features across time
        context = (weights.unsqueeze(-1) * x).sum(dim=1)  # (batch, features)
        return context


class CustomStutterDetector(nn.Module):
    """
    1D CNN + BiGRU + Attention for stutter type classification.
    Small enough (~85K params) for our 543-sample dataset.
    """

    def __init__(self, input_features=120, num_classes=3, dropout=0.3):
        super().__init__()

        # CNN block 1: detect short-range patterns (kernel=5 ≈ 160ms)
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)  # halve sequence length

        # CNN block 2: detect medium-range patterns (kernel=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)  # halve again

        # BiGRU: capture long-range temporal dependencies
        # GRU is lighter than LSTM — better for small datasets
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,  # output: 128 (64 × 2)
        )

        # Attention: learn which time steps are most important
        self.attention = Attention(feature_dim=128)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, features) — MFCC sequence
        """
        # Transpose for Conv1d: (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        # CNN block 1: local pattern detection
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (batch, 64, seq/2)

        # CNN block 2: deeper pattern detection
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (batch, 128, seq/4)

        # Transpose back for GRU: (batch, seq/4, 128)
        x = x.permute(0, 2, 1)

        # BiGRU: sequential modeling
        x, _ = self.gru(x)  # (batch, seq/4, 128)

        # Attention: weighted pooling across time
        x = self.attention(x)  # (batch, 128)

        # Classification
        logits = self.classifier(x)  # (batch, 3)
        return logits


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = CustomStutterDetector(input_features=120, num_classes=3)
    total, trainable = count_parameters(model)
    print(f"Custom Stutter Detector")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Test forward pass
    dummy = torch.randn(4, 220, 120)  # batch=4, 220 frames, 120 features
    out = model(dummy)
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {out.shape}")
