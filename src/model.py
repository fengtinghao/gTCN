"""Causal dilated TCN (WaveNet-style) renderer with FiLM effect conditioning.

new dry guitar --> [PerformanceEncoder] --> [renderer, causal dilated stack] --> wet guitar
                                                       ^ FiLM(z) on every block
                        (dry_ref, wet_ref) --> [EffectEncoder] --> z

Module split is deliberate for the bigger project plan: `performance_encoder`
+ `renderer` form the "backbone" that will eventually be frozen and shared
across effects, while `effect_encoder` only produces the conditioning vector
`z` (the exact interface a future hypernetwork will learn to predict from a
(dry, wet) reference pair, without touching the frozen backbone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class PerformanceEncoder(nn.Module):
    """Encodes the dry input into performance features, independent of effect.

    Part of the frozen backbone in the bigger hypernetwork project: it only
    ever sees the dry waveform, never the effect reference pair.
    """

    def __init__(self, residual_channels=64, kernel_size=15):
        super().__init__()
        pad = kernel_size - 1
        self.conv1 = nn.Conv1d(1, residual_channels, kernel_size)
        self.conv2 = nn.Conv1d(residual_channels, residual_channels, kernel_size)
        self.pad = pad

    def forward(self, dry):
        h = F.pad(dry, (self.pad, 0))
        h = F.relu(self.conv1(h))
        h = F.pad(h, (self.pad, 0))
        h = F.relu(self.conv2(h))
        return h


class EffectEncoder(nn.Module):
    """Encodes a (dry, wet) reference pair into a global effect embedding z.

    This is the module a future hypernetwork replaces: instead of a forward
    pass over a reference pair, the hypernetwork will directly predict a
    parameter set (or this same z) from a (dry, wet) pair.
    """

    def __init__(self, base_channels=32, cond_dim=128):
        super().__init__()

        c = base_channels
        self.net = nn.Sequential(
            nn.Conv1d(2, c, kernel_size=15, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c * 2, kernel_size=15, stride=4, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(c * 2, c * 4, kernel_size=15, stride=4, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(c * 4, c * 4, kernel_size=15, stride=4, padding=7),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(c * 4, cond_dim)

    def forward(self, dry_ref, wet_ref):
        x = torch.cat([dry_ref, wet_ref], dim=1)  # (B, 2, T)
        h = self.net(x)
        h = self.pool(h).squeeze(-1)  # (B, C)
        z = self.proj(h)  # (B, cond_dim)
        return z


class FiLMResidualBlock(nn.Module):
    """Causal dilated conv block with gated activation and FiLM conditioning."""

    def __init__(self, residual_channels, skip_channels, cond_dim, kernel_size, dilation):
        super().__init__()

        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size,
            dilation=dilation,
        )
        self.film = nn.Linear(cond_dim, 4 * residual_channels)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x, z):
        x_pad = F.pad(x, (self.pad, 0))
        h = self.conv(x_pad)  # (B, 2C, T)

        gamma, beta = self.film(z).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + gamma) + beta

        filt, gate = h.chunk(2, dim=1)
        activated = torch.tanh(filt) * torch.sigmoid(gate)

        residual = self.res_conv(activated)
        skip = self.skip_conv(activated)

        return (x + residual) * 0.7071067811865476, skip


class TCNRenderer(nn.Module):
    """Feed-forward causal dilated TCN that renders performance features -> wet given z."""

    def __init__(
        self,
        residual_channels=64,
        skip_channels=64,
        cond_dim=128,
        kernel_size=3,
        num_blocks=4,
        layers_per_block=10,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            for layer in range(layers_per_block):
                dilation = 2 ** layer
                self.blocks.append(
                    FiLMResidualBlock(
                        residual_channels, skip_channels, cond_dim, kernel_size, dilation
                    )
                )

        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, 1, kernel_size=1)

        self.receptive_field = 1 + num_blocks * (kernel_size - 1) * (2 ** layers_per_block - 1)

    def forward(self, perf_features, z):
        h = perf_features

        skip_sum = 0.0
        for block in self.blocks:
            if self.training and h.requires_grad:
                h, skip = torch.utils.checkpoint.checkpoint(block, h, z, use_reentrant=False)
            else:
                h, skip = block(h, z)
            skip_sum = skip_sum + skip

        out = F.relu(skip_sum)
        out = F.relu(self.output_conv1(out))
        out = torch.tanh(self.output_conv2(out))
        return out


class GTCNModel(nn.Module):
    """Full render pipeline: performance encoder + effect encoder + causal TCN renderer.

    `performance_encoder` + `renderer` are the "backbone": in the bigger
    hypernetwork project these get frozen and shared across all effects.
    `effect_encoder` is the only piece a hypernetwork needs to replace,
    predicting `z` directly from a (dry, wet) reference pair instead of
    running this module.
    """

    def __init__(
        self,
        residual_channels=64,
        skip_channels=64,
        kernel_size=3,
        num_blocks=4,
        layers_per_block=10,
        cond_dim=128,
        encoder_base_channels=32,
    ):
        super().__init__()

        self.performance_encoder = PerformanceEncoder(residual_channels=residual_channels)
        self.effect_encoder = EffectEncoder(base_channels=encoder_base_channels, cond_dim=cond_dim)
        self.renderer = TCNRenderer(
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
        )
        self.receptive_field = self.renderer.receptive_field + 2 * (self.performance_encoder.conv1.kernel_size[0] - 1)

    def forward(self, dry, dry_ref, wet_ref):
        perf_features = self.performance_encoder(dry)
        z = self.effect_encoder(dry_ref, wet_ref)
        wet_pred = self.renderer(perf_features, z)
        return wet_pred, z

    def encode_effect(self, dry_ref, wet_ref):
        """Encode a reference pair without rendering a target waveform."""
        return self.effect_encoder(dry_ref, wet_ref)

    def render_with_z(self, dry, z):
        """Render using a precomputed effect embedding."""
        perf_features = self.performance_encoder(dry)
        return self.renderer(perf_features, z)

    def backbone_parameters(self):
        """performance_encoder + renderer: frozen/shared in the hypernetwork stage."""
        yield from self.performance_encoder.parameters()
        yield from self.renderer.parameters()

    def effect_encoder_parameters(self):
        """The only parameters a future hypernetwork needs to replace/predict."""
        yield from self.effect_encoder.parameters()

    def freeze_backbone(self):
        for p in self.backbone_parameters():
            p.requires_grad = False

    @classmethod
    def from_config(cls, model_cfg):
        return cls(
            residual_channels=model_cfg["residual_channels"],
            skip_channels=model_cfg["skip_channels"],
            kernel_size=model_cfg["kernel_size"],
            num_blocks=model_cfg["num_blocks"],
            layers_per_block=model_cfg["layers_per_block"],
            cond_dim=model_cfg["cond_dim"],
            encoder_base_channels=model_cfg["encoder_base_channels"],
        )
