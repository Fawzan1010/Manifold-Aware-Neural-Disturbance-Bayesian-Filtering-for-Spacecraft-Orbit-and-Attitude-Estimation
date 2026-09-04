from __future__ import annotations

"""Transformer baseline training.

Records epochs, wall time, device, batch size, dataset sizes and loss curves
to transformer_training_log.json.
"""

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.transformer import ResidualTransformer
from src.utils.progress import progress, progress_bar


class SequenceDataset(Dataset):
    def __init__(self, trajectories, window: int) -> None:
        self.samples = []
        for traj in progress(trajectories, desc="  Building sequence windows",
                             leave=False, unit="traj"):
            meas = np.nan_to_num(traj.measurements, nan=0.0)
            for k in range(window, len(traj.time) - 1):
                hist = meas[k - window:k]
                self.samples.append(
                    (
                        hist.astype(np.float32),
                        traj.states[k + 1].astype(np.float32),
                        np.hstack([traj.states[k][19:22], traj.states[k][22:25]]).astype(np.float32),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a, b, c = self.samples[idx]
        return torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c)


@dataclass
class TrainedTransformer:
    model: ResidualTransformer
    window: int
    feature_dim: int
    path: Path


def _device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"CPU ({platform.processor() or platform.machine()})"


def train_transformer(trajectories_train, trajectories_val, cfg: dict, output_dir: Path,
                      device: str = "cpu", progress_desc: str = "Transformer Training",
                      show_progress: bool = True) -> TrainedTransformer:
    tr = cfg["training"]
    window = int(tr["window"])
    max_epochs = int(tr["transformer_epochs"])
    train_ds = SequenceDataset(trajectories_train, window)
    val_ds = SequenceDataset(trajectories_val, window)
    feature_dim = train_ds[0][0].shape[-1]
    model = ResidualTransformer(
        feature_dim,
        model_dim=int(tr["model_dim"]),
        num_heads=int(tr["num_heads"]),
        num_layers=int(tr["num_layers"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]))
    batch_size = min(len(train_ds), max(int(tr["batch_size"]), 128))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best = float("inf")
    best_epoch = -1
    train_curve: list[float] = []
    val_curve: list[float] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    epoch_bar = progress_bar(progress_desc, total=max_epochs, unit="epoch",
                             disable=not show_progress)
    for epoch in range(max_epochs):
        epoch_bar.set_description_str(f"{progress_desc} | Epoch {epoch + 1}/{max_epochs}")
        model.train()
        ep_losses = []
        for hist, nxt, target in progress(train_loader, desc="    train batches",
                                          leave=False, unit="batch",
                                          disable=not show_progress):
            hist, nxt, target = hist.to(device), nxt.to(device), target.to(device)
            pred_res, pred_state = model(hist)
            loss = torch.mean((pred_res - target) ** 2) + 0.5 * torch.mean((pred_state - nxt) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())

        model.eval()
        vals = []
        with torch.no_grad():
            for hist, nxt, target in progress(val_loader, desc="    val batches",
                                              leave=False, unit="batch",
                                              disable=not show_progress):
                hist, nxt, target = hist.to(device), nxt.to(device), target.to(device)
                pred_res, pred_state = model(hist)
                vals.append(torch.mean((pred_state - nxt) ** 2).item())
        score = float(np.mean(vals))
        train_curve.append(float(np.mean(ep_losses)))
        val_curve.append(score)
        epoch_bar.set_postfix_str(
            f"train={train_curve[-1]:.4e} val={score:.4e} best={min(best, score):.4e}")
        epoch_bar.update(1)

        if score < best:
            best = score
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "window": window, "feature_dim": feature_dim, "cfg": tr},
                output_dir / "transformer.pt",
            )
    epoch_bar.close()

    train_time_s = time.perf_counter() - t_start

    log = {
        "model": "ResidualTransformer",
        "device": device,
        "device_name": _device_name(device),
        "n_parameters": int(n_params),
        "epochs_run": max_epochs,
        "best_epoch": best_epoch + 1,
        "training_wall_time_s": round(train_time_s, 2),
        "batch_size": batch_size,
        "window_size": window,
        "model_dim": int(tr["model_dim"]),
        "num_heads": int(tr["num_heads"]),
        "num_layers": int(tr["num_layers"]),
        "learning_rate": float(tr["lr"]),
        "optimizer": "AdamW",
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "best_val_loss": best,
        "train_loss_curve": train_curve,
        "val_loss_curve": val_curve,
    }
    (output_dir / "transformer_training_log.json").write_text(json.dumps(log, indent=2))

    return TrainedTransformer(model, window, feature_dim, output_dir / "transformer.pt")


def load_transformer(path: Path, device: str = "cpu") -> TrainedTransformer:
    ckpt = torch.load(path, map_location=device)
    model = ResidualTransformer(
        ckpt["feature_dim"],
        model_dim=int(ckpt["cfg"]["model_dim"]),
        num_heads=int(ckpt["cfg"]["num_heads"]),
        num_layers=int(ckpt["cfg"]["num_layers"]),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return TrainedTransformer(model, ckpt["window"], ckpt["feature_dim"], path)
