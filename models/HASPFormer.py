import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import RobustDataEmbedding, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer

from layers.SelfAttention_Family import FullAttention, AttentionLayer  # From iTransformer


class HybridSTAR(nn.Module):
    def __init__(self, d_series, d_core, n_heads=2, dropout_rate=0.1, attention_dropout_rate=0.1,
                 initial_threshold=0.5):
        super(HybridSTAR, self).__init__()
        """
        HybridSTAR module
        """
        self.positional_embedding = PositionalEmbedding(d_series)

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

        self.attention_layer = AttentionLayer(
            attention=FullAttention(attention_dropout=attention_dropout_rate),
            d_model=d_core,
            n_heads=n_heads,
        )

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Initialize learnable threshold with an initial value
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))

        self.activation = F.gelu

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        input = self.positional_embedding(input)

        # set FFN
        combined_mean = self.activation(self.gen1(input))
        combined_mean = self.dropout1(combined_mean)
        combined_mean = self.gen2(combined_mean)

        # Apply attention if not in training mode, or randomly if in training mode
        if not self.training:
            apply_attention = True  # Always apply attention during evaluation/inference
        else:
            apply_attention = torch.rand(1).item() > self.threshold.item()  # Use learnable threshold during training

        if apply_attention:
            # Attention layer replacing stochastic pooling
            attn_output, attn_weights = self.attention_layer(
                queries=combined_mean,
                keys=combined_mean,
                values=combined_mean,
                attn_mask=None  # by default, it will be Triangular Causal Mask
            )

            combined_mean = attn_output
        else:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)

            attn_weights = None

        combined_mean = self.dropout2(combined_mean)  # Still apply dropout, but no attention modification

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = self.activation(self.gen3(combined_mean_cat))
        combined_mean_cat = self.dropout3(combined_mean_cat)  # Apply dropout
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat

        return output, attn_weights


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Embedding
        self.enc_embedding = RobustDataEmbedding(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    HybridSTAR(configs.d_model, configs.d_core,
                               n_heads=configs.lf_n_heads,
                               dropout_rate=configs.lf_dropout,
                               attention_dropout_rate=configs.lf_attention_dropout,
                               initial_threshold=configs.lf_initial_threshold),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
        )

        # Decoder/projection
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
