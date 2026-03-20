"""
StutterNet+ — Dataset and augmentation for Urdu stuttering detection.
"""
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SpecAugment:
    """SpecAugment-style augmentation for spectrograms (CHW format)."""

    def __init__(self, freq_mask_param=30, time_mask_param=70,
                 num_freq_masks=2, num_time_masks=2,
                 noise_std=0.02, prob=0.5):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.noise_std = noise_std
        self.prob = prob

    def __call__(self, spec):
        """Apply augmentations to spectrogram tensor (1, F, T)."""
        spec = spec.clone()
        _, F, T = spec.shape

        # Frequency masking
        if random.random() < self.prob:
            for _ in range(self.num_freq_masks):
                f = random.randint(0, min(self.freq_mask_param, F - 1))
                f0 = random.randint(0, F - f)
                spec[:, f0:f0 + f, :] = 0.0

        # Time masking
        if random.random() < self.prob:
            for _ in range(self.num_time_masks):
                t = random.randint(0, min(self.time_mask_param, T - 1))
                t0 = random.randint(0, T - t)
                spec[:, :, t0:t0 + t] = 0.0

        # Gaussian noise
        if random.random() < self.prob:
            noise = torch.randn_like(spec) * self.noise_std
            spec = (spec + noise).clamp(0.0, 1.0)

        # Random gain
        if random.random() < self.prob:
            gain = random.uniform(0.8, 1.2)
            spec = (spec * gain).clamp(0.0, 1.0)

        # Time reversal
        if random.random() < 0.3:
            spec = spec.flip(dims=[2])

        return spec


class StutterNetDataset(Dataset):
    """PyTorch Dataset for StutterNet+ spectrograms."""

    def __init__(self, samples, base_dir, transform=None, oversample=False):
        self.base_dir = base_dir
        self.transform = transform
        self.samples = samples

        # Build index list (with optional oversampling)
        self.indices = list(range(len(samples)))
        if oversample and len(samples) > 0:
            self.indices = self._oversample(samples)

    def _oversample(self, samples):
        """Duplicate minority class indices so each class has equal count."""
        label_to_indices = {}
        for i, s in enumerate(samples):
            lab = s["label"]
            label_to_indices.setdefault(lab, []).append(i)

        max_count = max(len(v) for v in label_to_indices.values())
        balanced = []
        for lab, idxs in label_to_indices.items():
            if len(idxs) == 0:
                continue
            repeated = idxs * (max_count // len(idxs)) + idxs[:max_count % len(idxs)]
            balanced.extend(repeated)
        random.shuffle(balanced)
        return balanced

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.samples[actual_idx]

        spec_path = os.path.join(self.base_dir, sample["spectrogram_file"])
        spec = np.load(spec_path)  # (257, 701, 1)
        spec = torch.from_numpy(spec).permute(2, 0, 1).float()  # (1, 257, 701)

        label = torch.tensor(sample["label"], dtype=torch.long)

        if self.transform:
            spec = self.transform(spec)

        return spec, label


def create_dataloaders(annotations_path, base_dir, batch_size=4,
                       val_split=0.2, seed=42):
    """Create train/val dataloaders with stratified split."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]

    # Stratified split
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    label_to_samples = {}
    for s in samples:
        label_to_samples.setdefault(s["label"], []).append(s)

    train_samples, val_samples = [], []
    for lab, group in label_to_samples.items():
        random.shuffle(group)
        n_val = max(1, int(len(group) * val_split))
        val_samples.extend(group[:n_val])
        train_samples.extend(group[n_val:])

    print(f"Dataset split: {len(train_samples)} train, {len(val_samples)} val")
    train_labels = Counter(s["label"] for s in train_samples)
    val_labels = Counter(s["label"] for s in val_samples)
    print(f"  Train labels: {dict(train_labels)}")
    print(f"  Val labels:   {dict(val_labels)}")

    augment = SpecAugment()
    train_ds = StutterNetDataset(train_samples, base_dir, transform=augment, oversample=True)
    val_ds = StutterNetDataset(val_samples, base_dir, transform=None, oversample=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_samples, val_samples
