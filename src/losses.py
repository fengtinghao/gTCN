"""Multi-resolution STFT loss + time-domain L1, standard for neural audio
effect / amp modeling (captures both transient and spectral-tonal error).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stft_mag(x, fft_size, hop_size, win_length):
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    mag = torch.clamp(spec.abs(), min=1e-7)
    return mag


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes, hop_sizes, win_lengths):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))

    def forward(self, pred, target):
        # pred, target: (B, 1, T) -> (B, T)
        pred = pred.squeeze(1)
        target = target.squeeze(1)

        sc_loss = 0.0
        mag_loss = 0.0

        for fft_size, hop_size, win_length in self.resolutions:
            pred_mag = _stft_mag(pred, fft_size, hop_size, win_length)
            target_mag = _stft_mag(target, fft_size, hop_size, win_length)

            sc_loss = sc_loss + torch.norm(target_mag - pred_mag, p="fro") / torch.norm(
                target_mag, p="fro"
            )
            mag_loss = mag_loss + F.l1_loss(torch.log(pred_mag), torch.log(target_mag))

        n = len(self.resolutions)
        return sc_loss / n, mag_loss / n


class RenderLoss(nn.Module):
    def __init__(self, stft_cfg, stft_weight=1.0, time_weight=1.0):
        super().__init__()
        self.mrstft = MultiResolutionSTFTLoss(
            stft_cfg["fft_sizes"], stft_cfg["hop_sizes"], stft_cfg["win_lengths"]
        )
        self.stft_weight = stft_weight
        self.time_weight = time_weight

    def forward(self, pred, target):
        sc_loss, mag_loss = self.mrstft(pred, target)
        time_loss = F.l1_loss(pred, target)

        total = (
            self.stft_weight * (sc_loss + mag_loss) + self.time_weight * time_loss
        )

        parts = {
            "loss": total.detach(),
            "sc_loss": sc_loss.detach(),
            "mag_loss": mag_loss.detach(),
            "time_loss": time_loss.detach(),
        }
        return total, parts
