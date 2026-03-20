"""
StutterNet+ — FluentNet architecture (SE-ResNet + BiLSTM + Attention)
for Urdu stuttering detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        red_ch = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, red_ch),
            nn.ReLU(inplace=True),
            nn.Linear(red_ch, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation."""

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.shortcut(x))


class SEResNetEncoder(nn.Module):
    """SE-ResNet encoder for spectrogram feature extraction."""

    def __init__(self, in_channels=1):
        super().__init__()
        # Stem: aggressive downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # SE-ResNet blocks
        self.block1 = SEResBlock(32, 64, stride=1)
        self.block2 = SEResBlock(64, 128, stride=2)
        self.block3 = SEResBlock(128, 128, stride=2)
        # Collapse frequency, keep time
        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        # x: (B, 1, 257, 701)
        x = self.stem(x)       # (B, 32, 65, 176)
        x = self.block1(x)     # (B, 64, 65, 176)
        x = self.block2(x)     # (B, 128, 33, 88)
        x = self.block3(x)     # (B, 128, 17, 44)
        x = self.pool(x)       # (B, 128, 1, 44)
        x = x.squeeze(2)       # (B, 128, 44)
        return x


class BiLSTMLayer(nn.Module):
    """Bidirectional LSTM for temporal modeling."""

    def __init__(self, input_size=128, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C) for LSTM
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)  # (B, T, 2*hidden)
        return out


class AttentionPooling(nn.Module):
    """Bahdanau-style attention over temporal sequence."""

    def __init__(self, feature_dim=128, attention_dim=64):
        super().__init__()
        self.W = nn.Linear(feature_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.last_attention_weights = None  # stored for visualization

    def forward(self, x):
        # x: (B, T, D)
        energy = torch.tanh(self.W(x))       # (B, T, attn_dim)
        scores = self.v(energy).squeeze(-1)   # (B, T)
        alpha = F.softmax(scores, dim=-1)     # (B, T)
        self.last_attention_weights = alpha.detach()
        context = (alpha.unsqueeze(-1) * x).sum(dim=1)  # (B, D)
        return context


class StutterNetPlus(nn.Module):
    """
    StutterNet+ — FluentNet architecture for Urdu stuttering detection.

    SE-ResNet encoder → BiLSTM → Attention Pooling → Classifier
    Input: (B, 1, 257, 701) spectrograms
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes=4, dropout_rate=0.5):
        super().__init__()
        self.encoder = SEResNetEncoder(in_channels=1)
        self.pre_lstm_dropout = nn.Dropout(0.3)
        self.bilstm = BiLSTMLayer(input_size=128, hidden_size=64, num_layers=1)
        self.attention = AttentionPooling(feature_dim=128, attention_dim=64)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 257, 701)
        features = self.encoder(x)           # (B, 128, 44)
        features = self.pre_lstm_dropout(features.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out = self.bilstm(features)     # (B, 44, 128)
        context = self.attention(lstm_out)   # (B, 128)
        logits = self.classifier(context)    # (B, num_classes)
        return logits


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = StutterNetPlus(num_classes=4)
    total, trainable = count_parameters(model)
    print(f"StutterNet+ Model")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Test forward pass
    dummy = torch.randn(2, 1, 257, 701)
    out = model(dummy)
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Attention weights shape: {model.attention.last_attention_weights.shape}")
