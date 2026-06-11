import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def remove_weight_norm(module: nn.Module) -> None:
    """Backward-compatible remover for parametrizations.weight_norm."""
    try:
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    except (ValueError, AttributeError):
        # Module may not have a weight-norm parametrization.
        pass


def _pick_nhead(d_model: int, preferred: int = 8) -> int:
    """Pick a valid nhead that divides d_model, preferring larger values."""
    max_head = min(preferred, d_model)
    for head in range(max_head, 0, -1):
        if d_model % head == 0:
            return head
    return 1


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for attention blocks."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Input sequence length ({x.size(1)}) exceeds max_len ({self.pe.size(1)})."
            )
        return x + self.pe[:, : x.size(1)]


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                )
            ),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                )
            ),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                )
            ),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
            ),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
            ),
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )
            ),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        stem_ch1 = max(16, h.ch_init_upsample // 16)
        stem_ch2 = max(32, h.ch_init_upsample // 8)

        # Spatial-temporal stem over per-window EEG maps (27 x 4).
        self.stem2d = nn.Sequential(
            nn.LazyConv2d(stem_ch1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv2d(stem_ch1, stem_ch2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv2d(stem_ch2, stem_ch2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((6, 2))

        # Tokenization: (B, W, D)
        g_d_model = h.ch_init_upsample // 2
        self.token_proj = nn.LazyLinear(g_d_model)

        g_nhead = _pick_nhead(g_d_model, preferred=8)
        self.pos_enc = PositionalEncoding(g_d_model)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=g_d_model,
                nhead=g_nhead,
                dim_feedforward=g_d_model * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )

        # Decoder/generator head: token -> mel channels per time-step.
        self.decoder = nn.Sequential(
            nn.Linear(g_d_model, g_d_model * 2),
            nn.GELU(),
            nn.Linear(g_d_model * 2, h.out_ch),
        )

        self.stem2d.apply(init_weights)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f"Generator expects EEG tensor with shape (B, channels, csp_components, windows), got {tuple(x.shape)}"
            )

        bsz, eeg_channels, csp_components, time_steps = x.shape

        # Build per-window 2D maps: (B, C, F, T) -> (B*T, 1, C, F)
        x = x.permute(0, 3, 1, 2).contiguous().reshape(bsz * time_steps, 1, eeg_channels, csp_components)

        # Spatial stem
        x = self.stem2d(x)
        x = self.spatial_pool(x)
        x = x.flatten(1)

        # Tokenization and attention
        x = self.token_proj(x).reshape(bsz, time_steps, -1)
        x = self.pos_enc(x)
        x = self.attn(x)

        # Decoder head
        x = torch.tanh(self.decoder(x))
        x = x.transpose(1, 2)

        # Temporal upsampling: EEG windows (e.g., 75) -> target mel windows (e.g., 85).
        target_time_steps = int(getattr(self.h, "out_time_steps", time_steps))
        if target_time_steps > 0 and x.size(-1) != target_time_steps:
            x = F.interpolate(x, size=target_time_steps, mode="linear", align_corners=False)

        out_freq_bins = int(getattr(self.h, "out_freq_bins", 80))
        out_spec_channels = int(getattr(self.h, "out_spec_channels", 1))
        if x.size(1) != out_spec_channels * out_freq_bins:
            if x.size(1) % out_freq_bins != 0:
                raise ValueError(
                    f"Generator output channels ({x.size(1)}) cannot be reshaped to spectrogram "
                    f"with out_freq_bins={out_freq_bins}. Set h.out_ch/out_freq_bins consistently."
                )
            out_spec_channels = x.size(1) // out_freq_bins

        return x.reshape(bsz, out_spec_channels, out_freq_bins, x.size(-1))


class Discriminator(torch.nn.Module):
    def __init__(self, h):
        super(Discriminator, self).__init__()
        self.h = h

        stem_ch1 = max(16, h.ch_init_downsample)
        stem_ch2 = max(32, h.ch_init_downsample * 2)

        # Spatial-temporal stem over per-window mel maps (spec_channels x mel_bins).
        self.stem2d = nn.Sequential(
            nn.LazyConv2d(stem_ch1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv2d(stem_ch1, stem_ch2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv2d(stem_ch2, stem_ch2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 8))

        # Tokenization: (B, T, D)
        d_d_model = h.ch_init_downsample * 4
        self.token_proj = nn.LazyLinear(d_d_model)

        d_nhead = _pick_nhead(d_d_model, preferred=8)
        self.pos_enc = PositionalEncoding(d_d_model)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_d_model,
                nhead=d_nhead,
                dim_feedforward=d_d_model * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )

        # Adversarial head only: real/fake judgement.
        self.adv_classifier = nn.Sequential(
            nn.Linear(d_d_model, 1),
            nn.Sigmoid(),
        )

        self.stem2d.apply(init_weights)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                f"Discriminator expects spectrogram tensor with shape (B, channels, mel_bins, windows), got {tuple(x.shape)}"
            )

        bsz, spec_channels, mel_bins, time_steps = x.shape

        # Match expected temporal length (e.g., 85) for both real and fake inputs.
        target_time_steps = int(getattr(self.h, "input_size", time_steps))
        if target_time_steps > 0 and time_steps != target_time_steps:
            x = F.interpolate(
                x,
                size=(mel_bins, target_time_steps),
                mode="bilinear",
                align_corners=False,
            )
            time_steps = target_time_steps

        # Build per-window 2D maps: (B, C, F, T) -> (B*T, 1, C, F)
        x = x.permute(0, 3, 1, 2).contiguous().reshape(bsz * time_steps, 1, spec_channels, mel_bins)

        # Spatial stem
        x = self.stem2d(x)
        x = self.spatial_pool(x)
        x = x.flatten(1)

        # Tokenization and attention
        x = self.token_proj(x).reshape(bsz, time_steps, -1)
        x = self.pos_enc(x)
        x = self.attn(x)

        # Pool over tokens and classify
        x = x.mean(dim=1)
        validity = self.adv_classifier(x)
        return validity

