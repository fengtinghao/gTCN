"""Render a new dry guitar recording in the style of a chosen effect.

The effect is specified either by name (a reference dry/wet pair is sampled
from the training data for that timbre) or by explicitly passing a
--ref_dry/--ref_wet pair (e.g. a short recording of the target pedal).

Usage:
    python inference.py --checkpoint ../checkpoints/best.pt \
        --input path/to/dry.wav --output path/to/wet.wav --timbre BritPlexi
"""

import argparse
import random
from pathlib import Path

import soundfile as sf
import torch

from dataset import GuitarPairIndex, _read_crop
from model import GTCNModel
from utils import resolve_path


class GuitarRenderer:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.cfg = checkpoint["config"]
        self.sample_rate = self.cfg["data"]["sample_rate"]

        self.model = GTCNModel.from_config(self.cfg["model"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print("Loaded model from:", checkpoint_path)
        print("Available timbres:", checkpoint.get("timbres"))

    def _load_full(self, path):
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        assert sr == self.sample_rate, (
            f"Input sample rate {sr} != model sample rate {self.sample_rate}. Resample first."
        )
        return audio

    def sample_reference(self, timbre, x_root=None, y_root=None):
        data_cfg = self.cfg["data"]
        x_root = x_root or resolve_path(data_cfg["x_root"])
        y_root = y_root or resolve_path(data_cfg["y_root"])

        index = GuitarPairIndex(x_root, y_root, timbre)
        rng = random.Random()
        dry_path, wet_path, frames = rng.choice(index.pairs)

        ref_len = int(data_cfg["reference_seconds"] * self.sample_rate)
        dry_ref = _read_crop(dry_path, frames, ref_len, rng)
        wet_ref = _read_crop(wet_path, frames, ref_len, rng)
        return dry_ref, wet_ref

    @torch.no_grad()
    def render(self, input_wav, output_wav, timbre=None, ref_dry_wav=None, ref_wet_wav=None):
        dry = self._load_full(input_wav)

        if ref_dry_wav and ref_wet_wav:
            dry_ref = self._load_full(ref_dry_wav)
            wet_ref = self._load_full(ref_wet_wav)
        elif timbre:
            dry_ref, wet_ref = self.sample_reference(timbre)
        else:
            raise ValueError("Provide either --timbre or both --ref_dry and --ref_wet")

        dry_t = torch.from_numpy(dry).float().to(self.device).view(1, 1, -1)
        dry_ref_t = torch.from_numpy(dry_ref).float().to(self.device).view(1, 1, -1)
        wet_ref_t = torch.from_numpy(wet_ref).float().to(self.device).view(1, 1, -1)

        pred, _ = self.model(dry_t, dry_ref_t, wet_ref_t)
        pred = pred.squeeze().detach().cpu().numpy()

        sf.write(str(output_wav), pred, self.sample_rate)
        print("Saved:", output_wav)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--timbre", type=str, default=None)
    parser.add_argument("--ref_dry", type=str, default=None)
    parser.add_argument("--ref_wet", type=str, default=None)
    args = parser.parse_args()

    renderer = GuitarRenderer(args.checkpoint)
    renderer.render(
        args.input,
        args.output,
        timbre=args.timbre,
        ref_dry_wav=args.ref_dry,
        ref_wet_wav=args.ref_wet,
    )


if __name__ == "__main__":
    main()
