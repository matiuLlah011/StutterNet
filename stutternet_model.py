"""
StutterNet — Model Architecture (PyTorch)
Follows the 8-layer StutterNet architecture from Abubakar et al.

Architecture:
  Layer 1: Input layer (sequence input)
  Layer 2: BiLSTM (100 units, return sequences)
  Layer 3: BiLSTM (100 units, return sequences)
  Layer 4: Dropout (0.25)
  Layer 5: RNN (128 units)
  Layer 6: Dense (512 units)
  Layer 7: Dropout (0.25)
  Layer 8: Dense output (3 classes, softmax)

NOTE: Original paper uses 2 output units (stuttered vs fluent).
      Adapted to 3 classes since our dataset has only stuttered speech
      with 3 stutter types: syllable_repetition, word_repetition, block.

Input: (batch, 40) — mean MFCC vector, reshaped to (batch, 40, 1) sequence.
Output: (batch, 3) — class logits.
"""
import torch
import torch.nn as nn


class StutterNet(nn.Module):
    """
    StutterNet 8-layer architecture for stutter type classification.
    Input: 40-dimensional mean MFCC vector per audio file.
    """

    def __init__(self, input_dim=40, num_classes=3):
        super().__init__()

        # Layer 1: Input reshape — treat 40 MFCCs as a sequence of 40 steps, 1 feature each
        # This allows the BiLSTM to learn patterns across MFCC coefficients
        self.input_dim = input_dim

        # Layer 2: First BiLSTM (100 units per direction, return full sequence)
        self.bilstm1 = nn.LSTM(
            input_size=1,           # each timestep has 1 feature
            hidden_size=100,        # 100 units per direction
            batch_first=True,
            bidirectional=True,     # bidirectional → output is 200-dim
        )

        # Layer 3: Second BiLSTM (100 units per direction, return full sequence)
        self.bilstm2 = nn.LSTM(
            input_size=200,         # input from first BiLSTM (100*2)
            hidden_size=100,        # 100 units per direction
            batch_first=True,
            bidirectional=True,     # output is 200-dim
        )

        # Layer 4: Dropout after BiLSTM layers
        self.dropout1 = nn.Dropout(0.25)

        # Layer 5: Simple RNN (128 units, takes last output only)
        self.rnn = nn.RNN(
            input_size=200,         # from second BiLSTM
            hidden_size=128,        # 128 hidden units
            batch_first=True,
        )

        # Layer 6: Dense layer (fully connected, 512 units)
        self.fc1 = nn.Linear(128, 512)
        self.relu = nn.ReLU()

        # Layer 7: Dropout before output
        self.dropout2 = nn.Dropout(0.25)

        # Layer 8: Output layer (3 classes for stutter types)
        # Original paper: 2 units + sigmoid for binary
        # Adapted: 3 units for our 3 stutter types
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass.
        x: (batch, 40) — mean MFCC features
        """
        # Reshape input: (batch, 40) → (batch, 40, 1) as a sequence
        x = x.unsqueeze(-1)  # (batch, 40, 1)

        # Layer 2: First BiLSTM — learns patterns across MFCC coefficients
        x, _ = self.bilstm1(x)   # (batch, 40, 200)

        # Layer 3: Second BiLSTM — deeper temporal modeling
        x, _ = self.bilstm2(x)   # (batch, 40, 200)

        # Layer 4: Dropout for regularization
        x = self.dropout1(x)     # (batch, 40, 200)

        # Layer 5: RNN — final sequential processing, take last hidden state
        _, h_n = self.rnn(x)     # h_n: (1, batch, 128)
        x = h_n.squeeze(0)       # (batch, 128)

        # Layer 6: Dense layer with ReLU activation
        x = self.relu(self.fc1(x))  # (batch, 512)

        # Layer 7: Dropout for regularization
        x = self.dropout2(x)     # (batch, 512)

        # Layer 8: Output layer — class logits
        x = self.fc2(x)          # (batch, 3)

        return x


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Quick test of the model
    model = StutterNet(input_dim=40, num_classes=3)
    total, trainable = count_parameters(model)
    print(f"StutterNet Model")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Test forward pass with dummy input
    dummy = torch.randn(4, 40)  # batch of 4, each with 40 MFCCs
    out = model(dummy)
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {out.shape}")
