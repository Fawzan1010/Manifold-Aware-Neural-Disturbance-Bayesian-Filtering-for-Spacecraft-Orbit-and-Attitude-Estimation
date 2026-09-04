from __future__ import annotations

"""PINN training.

Records the number of epochs run, wall-clock training time, device, batch
size, window size, learning rate, loss weights, dataset split sizes,
parameter count and per-epoch train/validation loss curves to
outputs/models/pinn_training_log.json.

Depth, activation, the quaternion-norm loss weight (lambda_norm),
early-stopping patience and cosine LR annealing are read from the config.
"""

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.pinn.dataset import ResidualDataset
from src.pinn.model import PINNResidualNet
from src.utils.progress import progress, progress_bar


@dataclass
class TrainedPINN:
    model: PINNResidualNet
    window: int
    feature_dim: int
    path: Path


def _device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"CPU ({platform.processor() or platform.machine()})"


def train_pinn(trajectories_train, trajectories_val, cfg: dict, output_dir: Path,
               device: str = "cpu", progress_desc: str = "PINN Training",
               show_progress: bool = True) -> TrainedPINN:
    tr = cfg["training"]
    window = int(tr["window"])
    depth = int(tr.get("depth", 4))
    activation = str(tr.get("activation", "tanh"))
    lambda_norm = float(tr.get("lambda_norm", 0.1))
    patience = int(tr.get("patience", 10))
    max_epochs = int(tr["pinn_epochs"])
    lr = float(tr["lr"])

    ds_train = ResidualDataset(trajectories_train, window)
    ds_val = ResidualDataset(trajectories_val, window)
    feature_dim = ds_train[0][0].numel()
    model = PINNResidualNet(feature_dim, hidden=int(tr["hidden"]), depth=depth, activation=activation).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    batch_size = min(len(ds_train), max(int(tr["batch_size"]), 128))
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size)

    best = float("inf")
    best_epoch = -1
    epochs_run = 0
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
        for x, hist, target, nxt in progress(train_loader, desc="    train batches",
                                             leave=False, unit="batch",
                                             disable=not show_progress):
            x = x.to(device)
            target = target.to(device)
            pred, logvar = model(x)
            loss_res = torch.mean((pred - target) ** 2 * torch.exp(-logvar) + logvar)
            quat = x[:, :4]
            loss_norm = torch.mean((torch.linalg.norm(quat, dim=1) - 1.0) ** 2)
            loss = loss_res + lambda_norm * loss_norm
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())
        scheduler.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for x, hist, target, nxt in progress(val_loader, desc="    val batches",
                                                 leave=False, unit="batch",
                                                 disable=not show_progress):
                x = x.to(device)
                target = target.to(device)
                pred, logvar = model(x)
                vals.append(torch.mean((pred - target) ** 2).item())
        score = float(np.mean(vals))
        train_curve.append(float(np.mean(ep_losses)))
        val_curve.append(score)
        epochs_run = epoch + 1
        epoch_bar.set_postfix_str(
            f"train={train_curve[-1]:.4e} val={score:.4e} best={min(best, score):.4e}")
        epoch_bar.update(1)

        if score < best:
            best = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "window": window,
                    "feature_dim": feature_dim,
                    "hidden": int(tr["hidden"]),
                    "depth": depth,
                    "activation": activation,
                },
                output_dir / "pinn.pt",
            )
        elif epoch - best_epoch >= patience:
            break  # early stopping
    epoch_bar.close()

    train_time_s = time.perf_counter() - t_start

    # ---- training disclosure ----
    log = {
        "model": "PINNResidualNet",
        "device": device,
        "device_name": _device_name(device),
        "n_parameters": int(n_params),
        "epochs_max": max_epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch + 1,
        "early_stopping_patience": patience,
        "training_wall_time_s": round(train_time_s, 2),
        "batch_size": batch_size,
        "window_size": window,
        "hidden_width": int(tr["hidden"]),
        "depth": depth,
        "activation": activation,
        "learning_rate": lr,
        "lr_schedule": "cosine_annealing",
        "lambda_norm": lambda_norm,
        "optimizer": "Adam",
        "grad_clip_norm": 1.0,
        "n_train_samples": len(ds_train),
        "n_val_samples": len(ds_val),
        "n_train_trajectories": len(trajectories_train),
        "n_val_trajectories": len(trajectories_val),
        "best_val_loss": best,
        "train_loss_curve": train_curve,
        "val_loss_curve": val_curve,
    }
    (output_dir / "pinn_training_log.json").write_text(json.dumps(log, indent=2))

    return TrainedPINN(model, window, feature_dim, output_dir / "pinn.pt")


def load_pinn(path: Path, device: str = "cpu") -> TrainedPINN:
    ckpt = torch.load(path, map_location=device)
    model = PINNResidualNet(
        ckpt["feature_dim"],
        hidden=int(ckpt.get("hidden", 128)),
        depth=int(ckpt.get("depth", 3)),
        activation=str(ckpt.get("activation", "silu")),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return TrainedPINN(model, ckpt["window"], ckpt["feature_dim"], path)
