"""Training and evaluation loops for Seq-MPS."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dual_state: Dict[str, float],
) -> Dict[str, float]:
    model.train()
    stats = {
        "loss": 0.0,
        "pred": 0.0,
        "rate": 0.0,
        "recon": 0.0,
        "mse": 0.0,
    }
    num_batches = 0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        losses = model.compute_loss(outputs, targets, lambda_val=dual_state["lambda"])

        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            avg_rate = losses["avg_rate"].item()
            dual_state["lambda"] = max(
                0.0,
                dual_state["lambda"] + dual_state["dual_step"] * (avg_rate - dual_state["rate_budget"]),
            )

        stats["loss"] += losses["total_loss"].item()
        stats["pred"] += losses["pred_loss"].item()
        stats["rate"] += losses["rate_loss"].item()
        stats["recon"] += losses["recon_loss"].item()
        stats["mse"] += losses["mse"].item()
        num_batches += 1

        progress.set_postfix(
            loss=f"{stats['loss']/num_batches:.4f}",
            rate=f"{stats['rate']/num_batches:.4f}",
            lam=f"{dual_state['lambda']:.3f}",
        )

    for key in stats:
        stats[key] /= max(1, num_batches)
    stats["lambda"] = dual_state["lambda"]
    return stats


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    stats = {
        "pred_loss": 0.0,
        "rate": 0.0,
        "recon": 0.0,
        "mse": 0.0,
        "mae": 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Eval", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            losses = model.compute_loss(outputs, targets, lambda_val=0.0)

            stats["pred_loss"] += losses["pred_loss"].item()
            stats["rate"] += losses["rate_loss"].item()
            stats["recon"] += losses["recon_loss"].item()
            stats["mse"] += losses["mse"].item()
            stats["mae"] += losses["mae"].item()
            num_batches += 1

    for key in stats:
        stats[key] /= max(1, num_batches)
    return stats
