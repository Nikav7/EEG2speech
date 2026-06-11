import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class PairedSpectrogramDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        eeg_key: str = "X",
        audio_key: str = "Y",
        reduce_eeg_channels: bool = True,
        eps: float = 1e-8,
        eeg_sample_rate: float = None,
        audio_sample_rate: float = None,
        sample_rate_ratio: float = 8.0,
    ):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset not found: {npz_path}")

        with np.load(npz_path) as payload:
            if eeg_key not in payload or audio_key not in payload:
                raise KeyError(
                    f"Missing keys in npz. Required keys: '{eeg_key}' and '{audio_key}'. "
                    f"Available keys: {list(payload.keys())}"
                )
            eeg = payload[eeg_key].astype(np.float32)
            audio = payload[audio_key].astype(np.float32)

            # Allow dataset-level ratio checks via metadata if available
            if eeg_sample_rate is None and "eeg_sample_rate" in payload:
                eeg_sample_rate = float(payload["eeg_sample_rate"])
            if audio_sample_rate is None and "audio_sample_rate" in payload:
                audio_sample_rate = float(payload["audio_sample_rate"])
            if "sample_rate_ratio" in payload:
                expected_ratio = float(payload["sample_rate_ratio"])
                if sample_rate_ratio is None:
                    sample_rate_ratio = expected_ratio

        self.eeg = self._prepare_eeg(eeg, reduce_eeg_channels)
        self.audio = self._prepare_audio(audio)

        if self.eeg.shape[0] != self.audio.shape[0]:
            raise ValueError(
                f"Mismatched sample counts: eeg={self.eeg.shape[0]}, audio={self.audio.shape[0]}"
            )

        if eeg_sample_rate is not None and audio_sample_rate is not None:
            if eeg_sample_rate <= 0 or audio_sample_rate <= 0:
                raise ValueError("Sample rates must be positive.")
            ratio = audio_sample_rate / eeg_sample_rate
            if abs(ratio - sample_rate_ratio) > 1e-6:
                raise ValueError(
                    f"Sample rate ratio mismatch: audio/eeg={ratio:.6f}, expected {sample_rate_ratio:.6f}."
                )

        self.eeg = self._zscore_per_sample(self.eeg, eps)
        self.audio = self._zscore_per_sample(self.audio, eps)

        if self.eeg.shape[-2:] != self.audio.shape[-2:]:
            raise ValueError(
                "EEG mel and audio spectrogram must have same (freq, time). "
                f"Got eeg={self.eeg.shape[-2:]}, audio={self.audio.shape[-2:]}"
            )

    @staticmethod
    def _prepare_eeg(eeg: np.ndarray, reduce_eeg_channels: bool) -> np.ndarray:
        if eeg.ndim == 4:
            if reduce_eeg_channels:
                eeg = eeg.mean(axis=1, keepdims=True)
            else:
                eeg = eeg
        elif eeg.ndim == 3:
            eeg = eeg[:, None, :, :]
        else:
            raise ValueError(f"Unsupported EEG shape: {eeg.shape}")
        return eeg

    @staticmethod
    def _prepare_audio(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 4:
            audio = audio[:, :1, :, :]
        elif audio.ndim == 3:
            audio = audio[:, None, :, :]
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")
        return audio

    @staticmethod
    def _zscore_per_sample(x: np.ndarray, eps: float) -> np.ndarray:
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        std = x.std(axis=(1, 2, 3), keepdims=True)
        return (x - mean) / np.maximum(std, eps)

    def __len__(self) -> int:
        return self.eeg.shape[0]

    def __getitem__(self, index: int):
        eeg = torch.from_numpy(self.eeg[index])
        audio = torch.from_numpy(self.audio[index])
        return eeg, audio


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.d1 = DownBlock(in_channels, base_channels, normalize=False)
        self.d2 = DownBlock(base_channels, base_channels * 2)
        self.d3 = DownBlock(base_channels * 2, base_channels * 4)
        self.d4 = DownBlock(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.u1 = UpBlock(base_channels * 8, base_channels * 8, dropout=True)
        self.u2 = UpBlock(base_channels * 16, base_channels * 4, dropout=True)
        self.u3 = UpBlock(base_channels * 8, base_channels * 2)
        self.u4 = UpBlock(base_channels * 4, base_channels)

        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        b = self.bottleneck(d4)

        u1 = self.u1(b)
        u1 = torch.cat([u1, d4], dim=1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d3], dim=1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d2], dim=1)

        u4 = self.u4(u3)
        u4 = torch.cat([u4, d1], dim=1)

        out = self.out_layer(u4)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        n_res_blocks: int = 6,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.down = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(*[ResBlock(base_channels * 4) for _ in range(n_res_blocks)])

        self.up = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        y = self.head(x)
        y = self.down(y)
        y = self.res_blocks(y)
        y = self.up(y)
        if y.shape[-2:] != input_hw:
            y = F.interpolate(y, size=input_hw, mode="bilinear", align_corners=False)
        y = self.tail(y)
        return y


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, eeg: torch.Tensor, target_or_fake: torch.Tensor) -> torch.Tensor:
        x = torch.cat([eeg, target_or_fake], dim=1)
        return self.model(x)


@dataclass
class GanLossWeights:
    adv: float = 1.0
    l1: float = 100.0


def gan_losses(
    discriminator: nn.Module,
    generator: nn.Module,
    eeg: torch.Tensor,
    audio_real: torch.Tensor,
    criterion_adv: nn.Module,
    criterion_l1: nn.Module,
    weights: GanLossWeights,
) -> Dict[str, torch.Tensor]:
    audio_fake = generator(eeg)

    pred_real = discriminator(eeg, audio_real)
    pred_fake = discriminator(eeg, audio_fake.detach())

    real_targets = torch.ones_like(pred_real)
    fake_targets = torch.zeros_like(pred_fake)

    loss_d_real = criterion_adv(pred_real, real_targets)
    loss_d_fake = criterion_adv(pred_fake, fake_targets)
    loss_d = 0.5 * (loss_d_real + loss_d_fake)

    pred_fake_for_g = discriminator(eeg, audio_fake)
    loss_g_adv = criterion_adv(pred_fake_for_g, real_targets)
    loss_g_l1 = criterion_l1(audio_fake, audio_real)
    loss_g = (weights.adv * loss_g_adv) + (weights.l1 * loss_g_l1)

    return {
        "loss_d": loss_d,
        "loss_g": loss_g,
        "loss_g_adv": loss_g_adv,
        "loss_g_l1": loss_g_l1,
    }


def make_sample_grid(tensor_batch: torch.Tensor, max_items: int = 4) -> torch.Tensor:
    batch = tensor_batch[:max_items].detach().cpu()
    if batch.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N,C,H,W], got {batch.shape}")
    return batch


def resize_to(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == target_hw:
        return x
    return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)


def build_generator(
    kind: str,
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 64,
    n_res_blocks: int = 6,
) -> nn.Module:
    kind_normalized = kind.lower()
    if kind_normalized == "unet":
        return UNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
    if kind_normalized == "resnet":
        return ResNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            n_res_blocks=n_res_blocks,
        )
    raise ValueError(f"Unsupported generator kind: {kind}. Use 'unet' or 'resnet'.")
