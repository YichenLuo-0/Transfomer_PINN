import torch
from torch import nn

from model.activation_func import WaveAct


class CNN(nn.Module):
    def __init__(self, d_in, d_out, kernel_size):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(d_in, d_in, kernel_size, padding=kernel_size // 2, groups=d_in)
        self.sequence = nn.Sequential(*[
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = x + self.conv(x)
        x = x.transpose(1, 2)

        x = self.sequence(x)
        return x


class AdaptiveFullConnected(nn.Module):
    def __init__(self, dims, num_heads):
        super(AdaptiveFullConnected, self).__init__()
        self.dims = dims
        self.num_heads = num_heads

        self.weight = CNN(2, dims * num_heads, 5)
        self.activation = WaveAct()

    def forward(self, x, coords, indices, patch_seq_len):
        x = x.repeat(1, 1, self.num_heads)

        # Use MLP to encode weights
        weight = self.weight(coords)

        # Pass through the learned weights
        x = x * weight

        # Calculate the mean value of all the patches
        mean = []
        for i in range(patch_seq_len):
            # Get the mask for the i-th patch
            mask = indices == i
            mask = mask.expand(-1, -1, x.size(2))

            # Calculate the mean value of the i-th patch
            mean_i = (x * mask).sum(dim=1) / mask.sum(dim=1)
            mean.append(mean_i)
        x = torch.stack(mean, dim=1)

        # Pass through the activation function
        x = self.activation(x)
        return x


# Patch layer
class Patching(nn.Module):
    def __init__(self, d_bc, d_emb, patch_seq_len, depth, num_expand, d_output):
        super(Patching, self).__init__()
        self.num_expand = num_expand
        self.d_model = d_emb * num_expand
        self.patch_seq_len = patch_seq_len
        self.window_size = 2 ** (depth * 2) // patch_seq_len

        self.real_seq_len = None
        self.indices = None

        # Define the layers
        self.adaptive_fcn = AdaptiveFullConnected(d_emb, num_expand)
        self.mlp_out = nn.Sequential(*[
            nn.Linear(self.d_model + 2 + d_bc, self.d_model),
            WaveAct(),
            nn.Linear(self.d_model, self.d_model),
            WaveAct(),
            nn.Linear(self.d_model, d_output),
        ])

    def patch(self, x, coords, code):
        self.indices = torch.zeros_like(code)
        self.real_seq_len = 0
        for i in range(self.patch_seq_len):
            # calculate the beginning and end index of the i-th patch
            begin = i * self.window_size
            end = (i + 1) * self.window_size
            indices_i = (code >= begin) & (code < end)

            # if the i-th patch is empty, skip it
            if indices_i.sum() == 0:
                continue

            # update the indices tensor
            self.indices += indices_i * self.real_seq_len
            self.real_seq_len += 1

        # Pass through the pooling layer
        x = self.adaptive_fcn(x, coords, self.indices, self.real_seq_len)
        return x

    def linear_out(self, coord, bc, encode):
        batch_size, seq_len, _ = coord.size()
        indices = self.indices.expand(-1, -1, self.d_model)
        encode = torch.gather(encode, 1, indices)
        coord = torch.cat([coord, bc, encode], dim=-1)
        return self.mlp_out(coord)
