"""Compute robust fixed effect embeddings for deterministic inference.

Saves one file per effect (e.g. checkpoints/BritPlexi_embeddings.pt).

Example:
    python src/compute_effect_embeddings.py \
        --checkpoint checkpoints/best.pt \
        --output-dir checkpoints

The embeddings must be used with the same checkpoint that produced them.
"""

import argparse
import random
from pathlib import Path

import torch

from dataset import GuitarPairIndex, _read_crop
from model import GTCNModel
from utils import resolve_path


def robust_center(embeddings, trim_fraction):
    """Return a trimmed mean after removing points far from a median center."""
    values = torch.stack(embeddings)
    median_center = values.median(dim=0).values
    distances = torch.linalg.vector_norm(values - median_center, dim=1)
    keep_count = max(1, int(len(values) * (1.0 - trim_fraction)))
    keep_indices = torch.argsort(distances)[:keep_count]
    return values[keep_indices].mean(dim=0), keep_indices


def compute_for_timbre(
    model,
    x_root,
    y_root,
    timbre,
    sample_rate,
    reference_seconds,
    num_references,
    trim_fraction,
    seed,
):
    index = GuitarPairIndex(x_root, y_root, timbre)
    rng = random.Random(seed)
    reference_length = int(sample_rate * reference_seconds)
    embeddings = []

    model.eval()
    with torch.inference_mode():
        for _ in range(num_references):
            dry_path, wet_path, frames = rng.choice(index.pairs)
            start = rng.randint(0, max(0, frames - reference_length))
            dry_ref = _read_crop(dry_path, frames, reference_length, rng, start=start)
            wet_ref = _read_crop(wet_path, frames, reference_length, rng, start=start)
            dry_ref = torch.from_numpy(dry_ref).float().to(next(model.parameters()).device).view(1, 1, -1)
            wet_ref = torch.from_numpy(wet_ref).float().to(next(model.parameters()).device).view(1, 1, -1)
            z = model.encode_effect(dry_ref, wet_ref).squeeze(0).cpu()
            embeddings.append(z)

    robust_z, keep_indices = robust_center(embeddings, trim_fraction)
    return robust_z, {
        "num_references": num_references,
        "num_inliers": int(len(keep_indices)),
        "trim_fraction": trim_fraction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timbres", nargs="+", default=None)
    parser.add_argument("--num-references", type=int, default=128)
    parser.add_argument("--trim-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = GTCNModel.from_config(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    x_root = resolve_path(cfg["data"]["x_root"])
    y_root = resolve_path(cfg["data"]["y_root"])
    timbres = args.timbres or checkpoint.get("timbres")
    if not timbres:
        raise ValueError("No timbres provided and checkpoint has no timbre list.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for offset, timbre in enumerate(timbres):
        z, stats = compute_for_timbre(
            model,
            x_root,
            y_root,
            timbre,
            cfg["data"]["sample_rate"],
            cfg["data"]["reference_seconds"],
            args.num_references,
            args.trim_fraction,
            args.seed + offset,
        )
        print(f"{timbre}: {stats}, z shape={tuple(z.shape)}")

        safe_name = timbre.replace(" ", "_")
        output_path = output_dir / f"{safe_name}_embeddings.pt"
        torch.save(
            {
                "checkpoint": str(checkpoint_path),
                "timbre": timbre,
                "z": z,
                "metadata": stats,
            },
            output_path,
        )
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
