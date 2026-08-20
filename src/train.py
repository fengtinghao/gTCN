"""Train the causal TCN renderer jointly across all timbres.

Usage:
    python train.py --config ../configs/default.yaml
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MultiTimbreGuitarDataset
from losses import RenderLoss
from model import GTCNModel
from utils import REPO_ROOT, count_parameters, load_config, resolve_path, set_seed


def discover_timbres(y_root):
    return sorted(p.name for p in Path(y_root).iterdir() if p.is_dir())


def make_loaders(cfg):
    data_cfg = cfg["data"]
    x_root = resolve_path(data_cfg["x_root"])
    y_root = resolve_path(data_cfg["y_root"])
    timbres = discover_timbres(y_root)
    print("Found timbres:", timbres)

    train_cfg = cfg["train"]

    common_kwargs = dict(
        x_root=x_root,
        y_root=y_root,
        timbres=timbres,
        sample_rate=data_cfg["sample_rate"],
        segment_seconds=data_cfg["segment_seconds"],
        reference_seconds=data_cfg["reference_seconds"],
        val_ratio=data_cfg["val_ratio"],
        seed=data_cfg["seed"],
    )

    train_set = MultiTimbreGuitarDataset(
        split="train",
        epoch_length=train_cfg["steps_per_epoch"] * train_cfg["batch_size"],
        **common_kwargs,
    )
    val_set = MultiTimbreGuitarDataset(
        split="val",
        epoch_length=train_cfg["val_steps"] * train_cfg["batch_size"],
        **common_kwargs,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # one single-timbre validation loader per effect, so we can report and
    # checkpoint model quality separately for each specific effect
    per_timbre_val_loaders = {}
    for timbre in timbres:
        timbre_kwargs = dict(common_kwargs, timbres=[timbre])
        timbre_val_set = MultiTimbreGuitarDataset(
            split="val",
            epoch_length=train_cfg["val_steps"] * train_cfg["batch_size"],
            **timbre_kwargs,
        )
        per_timbre_val_loaders[timbre] = DataLoader(
            timbre_val_set,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    return train_loader, val_loader, per_timbre_val_loaders, timbres


def run_epoch(model, loader, criterion, optimizer, device, train_cfg, writer, epoch, is_train):
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        dry = batch["dry"].to(device)
        wet = batch["wet"].to(device)
        dry_ref = batch["dry_ref"].to(device)
        wet_ref = batch["wet_ref"].to(device)

        with torch.set_grad_enabled(is_train):
            pred, _ = model(dry, dry_ref, wet_ref)
            loss, parts = criterion(pred, wet)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if train_cfg["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if is_train and step % train_cfg["log_every"] == 0:
            print(
                f"epoch {epoch} step {step}/{len(loader)} "
                f"loss={parts['loss']:.4f} sc={parts['sc_loss']:.4f} "
                f"mag={parts['mag_loss']:.4f} time={parts['time_loss']:.4f}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    tag = "train" if is_train else "val"
    if writer is not None:
        writer.add_scalar(f"{tag}/loss", avg_loss, epoch)

    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=str(REPO_ROOT / "configs" / "default.yaml")
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, per_timbre_val_loaders, timbres = make_loaders(cfg)

    model = GTCNModel.from_config(cfg["model"]).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Receptive field: {model.receptive_field} samples "
          f"({model.receptive_field / cfg['data']['sample_rate'] * 1000:.1f} ms)")

    criterion = RenderLoss(
        cfg["stft_loss"],
        stft_weight=cfg["train"]["stft_loss_weight"],
        time_weight=cfg["train"]["time_loss_weight"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    checkpoint_dir = resolve_path(cfg["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = resolve_path(cfg["train"]["tensorboard_dir"])
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tensorboard_dir))

    best_val_loss = float("inf")
    best_val_loss_per_timbre = {timbre: float("inf") for timbre in timbres}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, cfg["train"], writer, epoch, True
        )
        val_loss = run_epoch(
            model, val_loader, criterion, optimizer, device, cfg["train"], writer, epoch, False
        )

        val_loss_per_timbre = {}
        for timbre, timbre_loader in per_timbre_val_loaders.items():
            timbre_val_loss = run_epoch(
                model, timbre_loader, criterion, optimizer, device, cfg["train"], None, epoch, False
            )
            val_loss_per_timbre[timbre] = timbre_val_loss
            writer.add_scalar(f"val_per_timbre/{timbre}", timbre_val_loss, epoch)

        print(f"== epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} ==")
        print(f"   per-timbre val loss: {val_loss_per_timbre}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "timbres": timbres,
            "val_loss": val_loss,
            "val_loss_per_timbre": val_loss_per_timbre,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            print(f"Saved new best checkpoint (val_loss={val_loss:.4f})")

        for timbre, timbre_val_loss in val_loss_per_timbre.items():
            if timbre_val_loss < best_val_loss_per_timbre[timbre]:
                best_val_loss_per_timbre[timbre] = timbre_val_loss
                torch.save(checkpoint, checkpoint_dir / f"best_{timbre}.pt")
                print(f"Saved new best checkpoint for {timbre} (val_loss={timbre_val_loss:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
