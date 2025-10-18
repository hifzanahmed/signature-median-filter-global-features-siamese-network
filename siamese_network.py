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
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.fc(x)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        diff = torch.abs(out1 - out2)
        return self.out(diff)

# ==== 2. Dataset for Pairs ====

class SignatureDataset(Dataset):
    def __init__(self, genuine_vectors):
        self.pairs = []
        self.labels = []
        self.generate_pairs(genuine_vectors)

    def generate_pairs(self, genuine):
        # Positive pairs
        for i in range(len(genuine)):
            for j in range(i + 1, len(genuine)):
                self.pairs.append((genuine[i], genuine[j]))
                self.labels.append(1)

        # Negative pairs: forged = random noise
        for _ in range(len(self.pairs)):
            i = random.randint(0, len(genuine) - 1)
            fake = np.random.rand(len(genuine[0]))  # Forged vector
            self.pairs.append((genuine[i], fake))
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




