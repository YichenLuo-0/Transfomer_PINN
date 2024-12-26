import torch
from torch import nn


# Patch embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4):
        super(PatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.seq_len = 0
        self.pad_len = 0

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len
        self.pad_len = self.patch_size - (seq_len % self.patch_size)

    def forward(self, x):
        # Padding the input to make the sequence length can be divided by patch size
        batch_size, _, d_emb = x.shape
        if self.pad_len != 0 and self.pad_len != self.patch_size:
            last_input = x[:, -1, :].reshape(batch_size, 1, -1).expand(-1, self.pad_len, -1)
            x = torch.cat([x, last_input], dim=1)

        # Reshape the input to patches
        x = x.reshape(batch_size, -1, d_emb * self.patch_size)
        return x
