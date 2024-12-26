import torch
from torch import nn

from model.activation_func import WaveAct
from model.embedding import GBE
from model.patching import PatchEmbedding
from model.serialization import serialization, sort_tensor


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


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()

        # Multi-head self-attention layer
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        # Feed-forward network
        self.ff = MLP(d_model, d_model)

        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.attn(x, x, x)[0]
        x = residual + x
        x = self.layer_norm_1(x)

        residual = x
        x = self.ff(x)
        x = residual + x
        x = self.layer_norm_2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(Encoder, self).__init__()
        self.num_layers = num_layers

        # Stack multiple encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x):
        for i in range(self.num_layers):
            # Pass through each encoder layer
            x = self.layers[i](x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()

        # Multi-head cross-attention layer
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        # Feed-forward network
        self.ff = MLP(d_model, d_model)

        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, encoder_out, x):
        residual = x
        x = self.attn(x, encoder_out, encoder_out)[0]
        x = residual + x
        x = self.layer_norm_1(x)

        residual = x
        x = self.ff(x)
        x = residual + x
        x = self.layer_norm_2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers, num_heads):
        super(Decoder, self).__init__()
        self.num_layers = num_layers

        # Stack multiple decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, encoder_out, x):
        for i in range(self.num_layers):
            # Pass through the decoder layer
            x = self.layers[i](encoder_out, x)
        return x


class URT(nn.Module):
    def __init__(
            self,
            d_input,
            d_bc,
            d_output,
            d_emb,
            d_model,
            patch_size,
            depth,
            num_layers_encoder,
            num_layers_decoder,
            num_heads
    ):
        super(URT, self).__init__()
        self.depth = depth

        # Gated Boundary Embedding layer
        self.embedding = GBE(d_bc, d_hidden=d_emb, d_out=d_emb)
        self.positional_encoding = MLP(d_input, d_emb, d_emb)
        self.patching = PatchEmbedding(patch_size)

        # Encoder and decoder Transformer layers
        self.encoder = Encoder(d_model, num_layers_encoder, num_heads)
        self.decoder = Decoder(d_model, num_layers_decoder, num_heads)

        # Define the layers
        self.mlp_gird = nn.Linear(d_emb * patch_size, d_model)
        self.mlp_query = MLP(d_input + 1, d_model, d_model)
        self.mlp_out = nn.Sequential(
            *[
                MLP(d_model, d_model, d_model),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_output)
            ]
        )

    def forward(self, coords_gird, bc_gird, coords_query, t_query):
        batch_size, seq_len, _ = coords_gird.size()

        # Embedding and positional encoding, and encode the boundary conditions
        emb = self.embedding(bc_gird)
        pe = self.positional_encoding(coords_gird)
        gird = emb + pe

        # Sort the input tensor based on the code
        indices, _ = serialization(coords_gird, depth=self.depth)
        gird = sort_tensor(gird, indices)

        # Patching the input
        self.patching.set_seq_len(seq_len)
        query_patched = self.patching(gird)

        # Pass through the encoder layers
        encoder_in = self.mlp_gird(query_patched)
        encoder_out = self.encoder(encoder_in)

        # Pass through the decoder layers
        coords_t = torch.cat([coords_query, t_query], dim=-1)
        decoder_in = self.mlp_query(coords_t)
        decoder_out = self.decoder(encoder_out, decoder_in)

        out = self.mlp_out(decoder_out)
        return out
