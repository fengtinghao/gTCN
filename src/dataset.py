"""Paired dry/wet guitar dataset for the TCN renderer.

Layout on disk (sibling repo GTtimbre):
    X/<source>/(...)/<stem>.wav
    y/<timbre>/<source>/(...)/<stem>-<timbre>.wav

Each __getitem__ returns two independent random crops from the *same*
timbre:
  - an (dry, wet) training pair used as model input/target
  - an (dry_ref, wet_ref) reference pair fed to the effect encoder

The reference pair is drawn independently (usually a different file) so
the encoder is forced to learn a general effect embedding instead of just
copying the target crop.
"""

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def _to_mono(audio):
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio.astype(np.float32)


def _read_crop(path, frames_total, crop_len, rng, start=None):
    if frames_total >= crop_len:
        if start is None:
            start = rng.randint(0, frames_total - crop_len)
        audio, _ = sf.read(path, start=start, frames=crop_len, dtype="float32", always_2d=False)
        audio = _to_mono(audio)
    else:
        audio, _ = sf.read(path, dtype="float32", always_2d=False)
        audio = _to_mono(audio)
        padded = np.zeros(crop_len, dtype=np.float32)
        padded[: len(audio)] = audio
        audio = padded
    return audio


class GuitarPairIndex:
    """Indexes (dry, wet, frame_count) triples for a single timbre."""

    def __init__(self, x_root, y_root, timbre, min_frames=1):
        self.timbre = timbre
        y_timbre_root = Path(y_root) / timbre
        pairs = []

        for dry_path in Path(x_root).rglob("*.wav"):
            rel = dry_path.relative_to(x_root)
            wet_name = f"{dry_path.stem}-{timbre}.wav"
            wet_path = y_timbre_root / rel.parent / wet_name

            if not wet_path.exists():
                candidates = sorted((y_timbre_root / rel.parent).glob(f"{dry_path.stem}-*.wav"))
                if len(candidates) != 1:
                    continue
                wet_path = candidates[0]

            frames = sf.info(str(dry_path)).frames
            if frames < min_frames:
                continue

            pairs.append((str(dry_path), str(wet_path), frames))

        if not pairs:
            raise RuntimeError(f"No dry/wet pairs found for timbre '{timbre}'")

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)


class MultiTimbreGuitarDataset(Dataset):
    """Combines all timbres into one dataset with per-sample timbre labels."""

    def __init__(
        self,
        x_root,
        y_root,
        timbres,
        sample_rate=44100,
        segment_seconds=1.0,
        reference_seconds=2.0,
        split="train",
        val_ratio=0.1,
        seed=0,
        epoch_length=2000,
    ):
        self.sample_rate = sample_rate
        self.segment_length = int(sample_rate * segment_seconds)
        self.reference_length = int(sample_rate * reference_seconds)
        self.split = split
        self.epoch_length = epoch_length

        self.timbres = list(timbres)
        self.timbre_to_idx = {t: i for i, t in enumerate(self.timbres)}

        self.indices = {}
        for timbre in self.timbres:
            full_index = GuitarPairIndex(x_root, y_root, timbre)
            self.indices[timbre] = self._split_pairs(full_index.pairs, val_ratio, seed)

        self.timbre_sizes = {t: len(pairs) for t, pairs in self.indices.items()}
        print(f"[{split}] timbre pair counts: {self.timbre_sizes}")

    def _split_pairs(self, pairs, val_ratio, seed):
        pairs = sorted(pairs)
        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 0
        if self.split == "val":
            return pairs[:n_val] if n_val else pairs
        return pairs[n_val:] if n_val else pairs

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        rng = random.Random()
        if self.split == "train":
            rng.seed(idx * 2_147_483_647 + torch.initial_seed())
        else:
            rng.seed(idx)

        timbre = rng.choice(self.timbres)
        pairs = self.indices[timbre]

        dry_path, wet_path, frames = rng.choice(pairs)
        segment_start = rng.randint(0, max(0, frames - self.segment_length))
        dry = _read_crop(dry_path, frames, self.segment_length, rng, start=segment_start)
        wet = _read_crop(wet_path, frames, self.segment_length, rng, start=segment_start)

        ref_dry_path, ref_wet_path, ref_frames = rng.choice(pairs)
        reference_start = rng.randint(0, max(0, ref_frames - self.reference_length))
        dry_ref = _read_crop(
            ref_dry_path, ref_frames, self.reference_length, rng, start=reference_start
        )
        wet_ref = _read_crop(
            ref_wet_path, ref_frames, self.reference_length, rng, start=reference_start
        )

        return {
            "dry": torch.from_numpy(dry).unsqueeze(0),
            "wet": torch.from_numpy(wet).unsqueeze(0),
            "dry_ref": torch.from_numpy(dry_ref).unsqueeze(0),
            "wet_ref": torch.from_numpy(wet_ref).unsqueeze(0),
            "timbre_idx": self.timbre_to_idx[timbre],
            "timbre": timbre,
        }
