"""Training and evaluation loops for Seq-MPS."""
from __future__ import annotations

import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.utils import (
    cuda_max_memory_tracker,
    get_max_memory_allocated,
)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor for key, tensor in batch.items()}


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dual_state: Dict[str, float],
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    stats = {
        "loss": 0.0,
        "pred": 0.0,
        "rate": 0.0,
        "recon": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "nll": 0.0,
        "crps": 0.0,
        "inst": 0.0,
        "prefix": 0.0,
        "innovation": 0.0,
    }
    num_batches = 0
    num_samples = 0

    start_time = time.perf_counter()
    progress = tqdm(dataloader, desc="Train", leave=False)
    with cuda_max_memory_tracker(device):
        for batch in progress:
            batch = _to_device(batch, device)
            outputs = model(batch)

            constraint_cfg = {
                "lambda_inst": dual_state["lambda_inst"],
                "lambda_prefix": dual_state["lambda_prefix"],
                "beta_inst": dual_state["beta_inst"],
                "beta_prefix": dual_state["beta_prefix"],
            }
            losses = model.compute_loss(
                outputs,
                batch["targets"],
                constraint_cfg=constraint_cfg,
                target_mask=batch.get("target_mask"),
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            with torch.no_grad():
                inst_violation = losses["inst_violation"].item()
                prefix_violation = losses["prefix_violation"].item()
                dual_state["lambda_inst"] = max(
                    0.0,
                    dual_state["lambda_inst"]
                    + dual_state["dual_step_inst"] * inst_violation,
                )
                dual_state["lambda_prefix"] = max(
                    0.0,
                    dual_state["lambda_prefix"]
                    + dual_state["dual_step_prefix"] * prefix_violation,
                )

            batch_size = batch["inputs"].shape[0]
            num_samples += batch_size

            stats["loss"] += losses["total_loss"].item()
            stats["pred"] += losses["pred_loss"].item()
            stats["rate"] += losses["rate_loss"].item()
            stats["recon"] += losses["recon_loss"].item()
            stats["mse"] += losses["mse"].item()
            stats["mae"] += losses["mae"].item()
            stats["nll"] += losses["nll"].item()
            stats["crps"] += losses["crps"].item()
            stats["inst"] += losses["inst_violation"].item()
            stats["prefix"] += losses["prefix_violation"].item()
            stats["innovation"] += losses["innovation_loss"].item()
            num_batches += 1

            progress.set_postfix(
                loss=f"{stats['loss']/num_batches:.4f}",
                rate=f"{stats['rate']/num_batches:.4f}",
                lam_i=f"{dual_state['lambda_inst']:.3f}",
                lam_p=f"{dual_state['lambda_prefix']:.3f}",
            )

    elapsed = time.perf_counter() - start_time
    throughput = num_samples / max(elapsed, 1e-6)
    peak_mem = get_max_memory_allocated(device)

    for key in stats:
        stats[key] /= max(1, num_batches)
    stats["lambda_inst"] = dual_state["lambda_inst"]
    stats["lambda_prefix"] = dual_state["lambda_prefix"]
    stats["throughput"] = throughput
    stats["peak_mem"] = peak_mem
    stats["elapsed"] = elapsed
    return stats


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    constraint_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    model.eval()
    stats = {
        "pred_loss": 0.0,
        "rate": 0.0,
        "recon": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "nll": 0.0,
        "crps": 0.0,
        "inst": 0.0,
        "prefix": 0.0,
        "innovation": 0.0,
    }
    num_batches = 0

    constraint_cfg = constraint_cfg or {
        "lambda_inst": 0.0,
        "lambda_prefix": 0.0,
        "beta_inst": 0.0,
        "beta_prefix": 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            batch = _to_device(batch, device)
            outputs = model(batch)
            losses = model.compute_loss(
                outputs,
                batch["targets"],
                constraint_cfg=constraint_cfg,
                target_mask=batch.get("target_mask"),
            )

            stats["pred_loss"] += losses["pred_loss"].item()
            stats["rate"] += losses["rate_loss"].item()
            stats["recon"] += losses["recon_loss"].item()
            stats["mse"] += losses["mse"].item()
            stats["mae"] += losses["mae"].item()
            stats["nll"] += losses["nll"].item()
            stats["crps"] += losses["crps"].item()
            stats["inst"] += losses["inst_violation"].item()
            stats["prefix"] += losses["prefix_violation"].item()
            stats["innovation"] += losses["innovation_loss"].item()
            num_batches += 1

    for key in stats:
        stats[key] /= max(1, num_batches)
    return stats
