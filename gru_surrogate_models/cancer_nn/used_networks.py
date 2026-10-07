"""Script for torch neural networks"""
import math

import torch
import torch.nn as nn

from cancer_nn.nn_helpers import (
    attention_module,
    base_conv1_layer,
    calculate_matching_padding,
    transconv_matching_padding,
)

HORIZON = 72000.0  # last possible dose time in the data; readout happens here
DOSE_SCALE = 4.0  # max single dose in the data (Gy)

SUPPORTED_MODES = ("cnn_lstm", "gru", "cnn_att_gru", "event_gru", "ct_transformer")


def build_subnetworks(configs_list, mode):
    """Build one subnetwork per config, according to mode"""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported mode {mode!r}, expected one of {SUPPORTED_MODES}")
    builders = {
        "cnn_lstm": CNNAttLSTM,
        "gru": GRURegressor,
        "cnn_att_gru": CNNAttGRU,
        "event_gru": EventRecurrentRegressor,
        "ct_transformer": ContinuousTimeTransformer,
    }
    subnetwork_cls = builders[mode]
    return [subnetwork_cls(**config) for config in configs_list]


##################################################################
"""Multi head regressor"""
##################################################################
class MultiTaskRegressor(nn.Module):
    def __init__(self, configs_list, mode="cnn_lstm"):
        super().__init__()
        self.num_regr = len(configs_list)
        layers = build_subnetworks(configs_list, mode)
        self.layers = nn.ModuleList(layers)
        regr_heads = [nn.Sequential(nn.Linear(1, 1)) for _ in range(self.num_regr)]
        self.regr_heads = nn.ModuleList(regr_heads)
        self.last_layer = nn.Linear(self.num_regr * 1, 1)

    def forward(self, batch):
        outputs = []
        for i in range(self.num_regr):
            outputs.append(self.layers[i](batch))
        org_x = torch.cat(outputs, 1)
        x = torch.mean(org_x, 1)
        return torch.stack([x.unsqueeze(1)] + outputs, 1)


class MultiHeadTaskRegressor(nn.Module):
    def __init__(self, configs_list, mode="cnn_lstm"):
        super().__init__()
        self.num_regr = len(configs_list)
        layers = build_subnetworks(configs_list, mode)
        self.layers = nn.ModuleList(layers)
        regr_heads = [nn.Sequential(nn.Identity()) for _ in range(self.num_regr)]
        self.regr_heads = nn.ModuleList(regr_heads)
        self.last_layer = nn.Linear(self.num_regr * 1, 1)

    def forward(self, batch):
        sample_batch = (batch[0], batch[1])
        other_sample_batch = (batch[2], batch[3])
        outputs = []
        other_outputs = []
        for i in range(self.num_regr):
            output = self.layers[i](sample_batch)
            outputs.append(output)

        org_x = torch.cat(outputs, 1)
        x = torch.mean(org_x, 1)

        for i in range(self.num_regr):
            other_output = self.layers[i](other_sample_batch)
            other_outputs.append(other_output)

        other_org_x = torch.cat(other_outputs, 1)
        other_x = torch.mean(other_org_x, 1)
        return torch.stack([x.unsqueeze(1)] + outputs + [other_x.unsqueeze(1)], 1)


##############################
"""UNET CNN 1D"""#############
##############################


class BaseDecBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        maxpool_size=2,
        batchnorm="default",
        down="convolution1d",
        sh=128,
        dilation=1,
        *args,
        **kwargs
    ):
        modes = nn.ModuleDict(
            [
                ["maxpooling", nn.MaxPool1d(maxpool_size)],
                [
                    "convolution1d",
                    base_conv1_layer(
                        channels[-1],
                        channels[-1],
                        kernel_size=maxpool_size,
                        dilation=dilation,
                    ),
                ],
            ]
        )
        super(BaseDecBlock, self).__init__()
        dilations = [1 if i % 2 == 0 else 2 for i in range(len(channels[1:]))]
        conv_blocks = [
            base_conv1_layer(
                in_f,
                out_f,
                kernel_size=kernel_size,
                padding=calculate_matching_padding(sh, sh, kernel_size, dilation, 1),
                dilation=dilation,
            )
            for in_f, out_f, dilation in zip(channels, channels[1:], dilations)
        ]

        conv_blocks.append(modes[down])
        self.block = nn.Sequential(*conv_blocks)

    def forward(self, x):
        return self.block(x)


class BaseEncBlock(nn.Module):
    def __init__(
        self,
        channels,
        ch_out=3,
        kernel_size=3,
        upsl_size=2,
        batchnorm="default",
        sh=128,
        out_sh=256,
        trans_conv=False,
        *args,
        **kwargs
    ):
        super(BaseEncBlock, self).__init__()

        modes = nn.ModuleDict(
            [
                [
                    "convolution1d",
                    nn.Sequential(
                        nn.ConvTranspose1d(
                            channels[-1],
                            ch_out,
                            kernel_size,
                            stride=2,
                            padding=1,
                            output_padding=transconv_matching_padding(
                                out_sh, sh, 2, 1, kernel_size
                            ),
                        ),
                        nn.ReLU(),
                    ),
                ]
            ]
        )
        conv_blocks = [
            base_conv1_layer(
                in_f,
                out_f,
                kernel_size=kernel_size,
                padding=calculate_matching_padding(sh, sh, kernel_size, 1, 1),
            )
            for in_f, out_f in zip(channels, channels[1:])
        ]
        if trans_conv:
            conv_blocks.append(modes["convolution1d"])
        self.block = nn.Sequential(*conv_blocks)

    def forward(self, x):
        x = self.block(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        channels,
        shapes=[20, 10, 5],
        channel_len=2,
        kernels=[3, 3, 3, 3],
        maxpool_size=2,
        batchnorm="default",
        down="maxpooling",
    ):
        super(Decoder, self).__init__()
        dec_blocks = [
            BaseDecBlock(
                [c_in if i == 0 else c_out for i in range(channel_len)],
                kernel_size=kernel,
                maxpool_size=maxpool_size,
                batchnorm=batchnorm,
                down=down,
                sh=sh,
            )
            for c_in, c_out, kernel, sh in zip(channels, channels[1:], kernels, shapes)
        ]

        self.dec = nn.ModuleList(dec_blocks)

    def forward(self, x):
        outputs_to_save = [x]
        for layer in self.dec:
            x = layer(x)
            outputs_to_save.append(x)
        return outputs_to_save


def middle_layer(in_c, out_c, kernel_size, sh, out_sh):
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_c,
            out_c,
            kernel_size,
            stride=2,
            padding=1,
            output_padding=transconv_matching_padding(sh, out_sh, 2, 1, kernel_size),
        ),
        nn.ReLU(),
    )


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        channel_len=2,
        kernel_size=3,
        upsl_size=2,
        batchnorm="default",
        dosum="True",
        shapes=[20, 10, 5],
    ):
        super(Encoder, self).__init__()
        self.dosum = dosum
        if self.dosum:
            n = 1
        else:
            n = 2
        channels = channels[::-1][1:]
        sum_channels = channels[0]
        enc_channels = []
        for ch in channels:
            sum_channels = sum_channels + ch
            enc_channels.append(sum_channels)
        self.last_channels_num = enc_channels[-1]
        enc_blocks = [
            BaseEncBlock(
                [enc_channels[i] for j in range(channel_len)],
                ch_out=enc_channels[i],
                kernel_size=kernel_size,
                upsl_size=upsl_size,
                batchnorm=batchnorm,
                sh=shapes[i],
                out_sh=shapes[i + 1],
                trans_conv=True if i < len(enc_channels) - 1 else False,
            )
            for i in range(len(enc_channels))
        ]

        self.enc = nn.ModuleList(enc_blocks)

    def forward(self, x, saved_outputs):
        for layer, y in zip(self.enc, saved_outputs[::-1][1:]):
            if not self.dosum:
                x = torch.cat([y, x], dim=1)
            else:
                x = x + y
            x = layer(x)
        return x


#################### LSTM + CNN ##################


class CNNContext(nn.Module):
    def __init__(
        self,
        channels=[3, 8, 16],
        channel_len=4,
        kernels=[5, 3, 3, 3, 3],
        updown_size=2,
        batchnorm="default",
        dosum=False,
        down="maxpooling",
        shapes=[20, 10, 5],
        output_las_layer=True,
        with_att=True,
        output=64,
    ):
        super(CNNContext, self).__init__()

        self.dec = Decoder(
            channels,
            channel_len=channel_len,
            kernels=kernels,
            shapes=shapes,
            maxpool_size=updown_size,
            batchnorm=batchnorm,
            down=down,
        )
        self.middle_layer = middle_layer(
            channels[-1], channels[-2], 3, shapes[-1], shapes[-2]
        )
        self.enc = Encoder(
            channels,
            channel_len=channel_len,
            kernel_size=3,
            shapes=shapes,
            upsl_size=updown_size,
            batchnorm="default",
            dosum=dosum,
        )

        self.final_conv_layer = base_conv1_layer(
            self.enc.last_channels_num,
            self.enc.last_channels_num,
            5,
            padding=calculate_matching_padding(shapes[0], shapes[0], 5, 1, 1),
        )
        self.self_att_layer = attention_module(embed_dim=self.enc.last_channels_num)
        self.activation_layer = nn.GELU()
        self.last_layer = nn.Linear(self.enc.last_channels_num * 20, output)
        self.with_att = with_att

    def forward(self, x):
        x = x[0]
        x = x.permute(0, 2, 1)
        dec_outs = self.dec(x)
        x = dec_outs[-1]
        x = self.middle_layer(x)
        x = self.enc(x, dec_outs)
        x = self.final_conv_layer(x)
        x = x.permute(0, 2, 1)
        if self.with_att:
            x, _ = self.self_att_layer(x, x, x)
        x = self.activation_layer(x)
        x = torch.flatten(x, start_dim=1)
        return self.last_layer(x)


class CNNAttLSTM(torch.nn.Module):
    def __init__(self, n_h=16, n_l=4):
        super(CNNAttLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            3, n_h, n_l, bidirectional=True, batch_first=True, dropout=0.2
        )
        self.att = attention_module(embed_dim=n_h * 2)
        self.att1 = attention_module(embed_dim=n_h * 2)
        self.context = CNNContext(output=n_h * 2)
        self.activation_layer = nn.GELU()
        self.last_layer = nn.Linear(1 * n_h * 2, 1)

    def forward(self, x):
        query = self.context(x)
        query = query.unsqueeze(1)
        x = x[0]
        x, _ = self.lstm(x)
        x, _ = self.att(x, x, x)
        x = self.activation_layer(x)
        x, _ = self.att1(query, x, x)
        x = torch.flatten(x, start_dim=1)
        return self.last_layer(x)


class GRURegressor(nn.Module):
    """GRU over the (time, dose, time_gap) series, outcome read off the final hidden state."""

    def __init__(
        self,
        hidden_size: int = 16,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        n_features: int = 3,
        readout: str = "linear",
    ):
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.readout_kind = readout
        # the linear readout keeps the parameter names of older checkpoints
        if readout == "linear":
            self.last_layer = nn.Linear(out_size, 1)
        else:
            self.unscaler = FeatureUnscaler(n_features)
            self.readout = Readout(out_size, readout)

    def set_feature_scaling(self, scalers):
        if hasattr(self, "unscaler"):
            self.unscaler.set_from_scalers(scalers)

    def forward(self, x):
        x = x[0]
        states, h_n = self.gru(x)
        if self.gru.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]
        if self.readout_kind == "linear":
            return self.last_layer(last)
        mask = self.unscaler.event_mask(x)
        return self.readout(last, states, mask)


class CNNAttGRU(nn.Module):
    """CNNAttLSTM with a GRU core instead of an LSTM."""

    def __init__(
        self,
        hidden_size: int = 16,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        n_features: int = 3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.att = attention_module(embed_dim=out_size)
        self.att1 = attention_module(embed_dim=out_size)
        self.context = CNNContext(output=out_size)
        self.activation_layer = nn.GELU()
        self.last_layer = nn.Linear(out_size, 1)

    def forward(self, x):
        query = self.context(x)
        query = query.unsqueeze(1)
        x = x[0]
        x, _ = self.gru(x)
        x, _ = self.att(x, x, x)
        x = self.activation_layer(x)
        x, _ = self.att1(query, x, x)
        x = torch.flatten(x, start_dim=1)
        return self.last_layer(x)


class FeatureUnscaler(nn.Module):
    """Undoes the dataset's StandardScaler so a network can tell real doses from padding."""

    def __init__(self, n_features: int = 3):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def set_from_scalers(self, scalers, columns=("time", "dose", "time_gap")):
        mean = torch.tensor([float(scalers[c].mean_[0]) for c in columns])
        std = torch.tensor([float(scalers[c].scale_[0]) for c in columns])
        self.mean.copy_(mean.to(self.mean.device))
        self.std.copy_(std.to(self.std.device))

    def forward(self, x):
        return x * self.std + self.mean

    def event_mask(self, x):
        return (self(x)[..., 1] > 1e-3).float()


def gap_features(dt, horizon=HORIZON):
    """Linear and log features of a raw time gap, both bounded by 1."""
    dt = dt.clamp(min=0.0)
    return torch.stack(
        [dt / horizon, torch.log1p(dt / 100.0) / math.log1p(horizon / 100.0)], dim=-1
    )


class Readout(nn.Module):
    """Maps a final state (and optionally per-event states) to the scaled prediction.

    "linear" is Linear(final). "additive_log" is exp(log y0 + Linear(final) + masked sum of
    per-event contributions), a product of per-dose factors; y0 is the mean scaled target.
    """

    KINDS = ("linear", "additive_log")

    def __init__(self, hidden: int, kind: str = "linear", y0: float = 0.85):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"unknown readout {kind!r}, expected one of {self.KINDS}")
        self.kind = kind
        self.log_y0 = math.log(y0)
        self.final = nn.Linear(hidden, 1)
        if kind != "linear":
            self.contrib = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
            )
            nn.init.zeros_(self.contrib[-1].weight)
            nn.init.zeros_(self.contrib[-1].bias)
            # start exactly at y0
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)

    def forward(self, final, states=None, mask=None):
        if self.kind == "linear":
            return self.final(final)
        c = self.contrib(states).squeeze(-1)
        z = self.final(final) + self.log_y0 + (c * mask).sum(dim=1, keepdim=True)
        return torch.exp(z.clamp(-3.0, 3.0))


class EventRecurrentRegressor(nn.Module):
    """Event-based recurrence (mode ``event_gru``).

    The state is evolved through each inter-dose gap by a gap-conditioned residual MLP,
    updated by a GRUCell at each real dose, evolved to the horizon after the last dose and
    then read out. Padded rows are masked out.
    """

    def __init__(
        self,
        hidden_size: int = 96,
        flow_hidden: int = 192,
        flow_steps: int = 2,
        dropout: float = 0.1,
        readout: str = "linear",
        horizon: float = HORIZON,
        n_features: int = 3,
    ):
        super().__init__()
        self.horizon = horizon
        self.flow_steps = flow_steps
        self.unscaler = FeatureUnscaler(n_features)
        self.h0 = nn.Parameter(torch.zeros(1, hidden_size))
        self.dose_embed = nn.Sequential(nn.Linear(3, hidden_size), nn.GELU())
        self.jump = nn.GRUCell(hidden_size, hidden_size)
        self.flow_norm = nn.LayerNorm(hidden_size)
        self.flow = nn.Sequential(
            nn.Linear(hidden_size + 2, flow_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flow_hidden, hidden_size),
        )
        self.readout = Readout(hidden_size, readout)

    def set_feature_scaling(self, scalers):
        self.unscaler.set_from_scalers(scalers)

    def evolve(self, h, dt):
        feats = gap_features(dt / self.flow_steps, self.horizon)
        for _ in range(self.flow_steps):
            h = h + self.flow(torch.cat([self.flow_norm(h), feats], dim=1))
        return h

    def forward(self, x):
        x = x[0]
        raw = self.unscaler(x)
        t, d = raw[..., 0], raw[..., 1]
        mask = (d > 1e-3).float()
        batch, seq = mask.shape
        h = self.h0.expand(batch, -1)
        prev_t = torch.zeros(batch, device=x.device)
        dose_in = self.dose_embed(
            torch.stack([d / DOSE_SCALE, (d / DOSE_SCALE) ** 2, t / self.horizon], -1)
        )
        states = []
        for i in range(seq):
            m = mask[:, i : i + 1]
            dt = (t[:, i] - prev_t) * mask[:, i]
            h_new = self.jump(dose_in[:, i], self.evolve(h, dt))
            h = m * h_new + (1 - m) * h
            prev_t = mask[:, i] * t[:, i] + (1 - mask[:, i]) * prev_t
            states.append(h)
        h = self.evolve(h, (self.horizon - prev_t).clamp(min=0.0))
        return self.readout(h, torch.stack(states, dim=1), mask)


class CTEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with an additive per-head attention bias."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_bias, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.ln2(x)))


class ContinuousTimeTransformer(nn.Module):
    """Transformer over dose events with a learned relative-time attention bias
    (mode ``ct_transformer``). Padded rows are masked out of attention and pooling.
    """

    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
        readout: str = "linear",
        horizon: float = HORIZON,
        n_features: int = 3,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.horizon = horizon
        self.nhead = nhead
        self.unscaler = FeatureUnscaler(n_features)
        self.token_embed = nn.Sequential(
            nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.rel_bias = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, nhead))
        self.layers = nn.ModuleList(
            [CTEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.readout = Readout(d_model, readout)

    def set_feature_scaling(self, scalers):
        self.unscaler.set_from_scalers(scalers)

    def relative_time_bias(self, t, mask):
        delta = t.unsqueeze(2) - t.unsqueeze(1)
        ad = delta.abs()
        feats = torch.stack(
            [
                delta / self.horizon,
                torch.sign(delta) * torch.log1p(ad / 100.0) / math.log1p(self.horizon / 100.0),
                torch.exp(-ad / 1000.0),
                torch.exp(-ad / 10000.0),
            ],
            dim=-1,
        )
        bias = self.rel_bias(feats).permute(0, 3, 1, 2)
        bias = bias + (1.0 - mask).unsqueeze(1).unsqueeze(2) * -1e4
        batch, seq = mask.shape
        return bias.reshape(batch * self.nhead, seq, seq)

    def forward(self, x):
        x = x[0]
        raw = self.unscaler(x)
        t, d, gap = raw[..., 0], raw[..., 1], raw[..., 2]
        mask = (d > 1e-3).float()
        last_t = (t * mask).amax(dim=1, keepdim=True)
        tail = (self.horizon - last_t).clamp(min=0.0).expand_as(t)
        tok = torch.cat(
            [
                torch.stack([d / DOSE_SCALE, (d / DOSE_SCALE) ** 2, t / self.horizon], -1),
                gap_features(gap, self.horizon),
                (tail / self.horizon).unsqueeze(-1),
            ],
            dim=-1,
        )
        h = self.token_embed(tok)
        bias = self.relative_time_bias(t, mask)
        for layer in self.layers:
            h = layer(h, bias)
        h = self.norm(h)
        m = mask.unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.readout(pooled, h, mask)
