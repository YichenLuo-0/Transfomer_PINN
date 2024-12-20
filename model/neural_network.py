import torch
from torch import nn

from model.activation_func import WaveAct
from model.embedding import GBE
from model.patching import Patching
from model.serialization import serialization, sort_tensor, desort_tensor


class MLP(nn.Module):
    def __init__(self, d_input, d_output, d_ff=256, num_layers=3):
        super(MLP, self).__init__()
        if num_layers < 3:
            raise ValueError("The number of layers must be greater than 2.")

        # Define the layers
        layers = [nn.Linear(d_input, d_ff), WaveAct()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(d_ff, d_ff))
            layers.append(WaveAct())
        if num_layers > 2:
            layers.append(nn.Linear(d_ff, d_output))
            layers.append(WaveAct())

        # Sequentially stack the layers
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ff = MLP(d_model, d_model)

    def forward(self, x_in, x):
        x = x + self.attn(x, x, x)[0]
        x = self.layer_norm_1(x)

        x = x + self.ff(x)
        x = self.layer_norm_2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(Decoder, self).__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x):
        x_in = x

        for i in range(self.num_layers):
            # Pass through the encoder layer
            x = self.layers[i](x_in, x)
        return x


class PinnsFormer(nn.Module):
    def __init__(
            self,
            bc_dims,
            d_output,
            d_emb,
            num_expand,
            patch_seq_len,
            depth,
            num_layers,
            num_heads
    ):
        super(PinnsFormer, self).__init__()
        self.depth = depth
        self.d_model = d_emb * num_expand

        # Gated Boundary Embedding layer
        self.gbe = GBE(bc_dims, 2, d_emb, d_emb)
        self.positional_encoding = MLP(2, d_emb, d_emb, 4)
        self.patching = Patching(sum(bc_dims) + 1, d_emb, patch_seq_len, depth, num_expand, d_output)

        # Encoder only Transformer model
        self.encoder = Decoder(self.d_model, num_layers, num_heads)

    def forward(self, x, y, bc):
        coords = torch.cat([x, y], dim=-1)
        emb = self.gbe(coords, bc)

        # Sort the input tensor based on the code
        indices, code = serialization(coords, bc, depth=self.depth)
        coords = sort_tensor(coords, indices)
        emb = sort_tensor(emb, indices)
        code = sort_tensor(code, indices)

        # Patching the input
        out = self.patching.patch(emb, coords, code)

        # Pass through the encoder layers
        out = self.encoder(out)

        # Desort the output tensor based on the indices
        out = self.patching.linear_out(coords, bc, out)
        out = desort_tensor(out, indices)
        return out
