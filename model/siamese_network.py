import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# ==== 1. Siamese Network Definition ====
class SiameseNetwork(nn.Module):
    def __init__(self, input_size):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),      # Dropout layer added here
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3)       # Dropout layer added here
        )
        # Output layer without Sigmoid
        self.out = nn.Linear(32, 1)

    def forward_once(self, x):
        return self.fc(x)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        diff = torch.abs(out1 - out2)
        return self.out(diff)  # raw logits returned here

# ==== 2. Dataset for Pairs ====

class SignatureDataset(Dataset):
    def __init__(self, genuine_vectors, forged_vectors):
        self.pairs = []
        self.labels = []
        self.generate_pairs(genuine_vectors, forged_vectors)

    def generate_pairs(self, genuine, forged):
        # Positive pairs: pairs between different genuine signatures
        for i in range(len(genuine)):
            for j in range(i + 1, len(genuine)):
                self.pairs.append((genuine[i], genuine[j]))
                self.labels.append(1)

        # Negative pairs: pairs between genuine and forged signatures
        for g in genuine:
            for f in forged:
                self.pairs.append((g, f))
                self.labels.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        y = self.labels[idx]
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )





