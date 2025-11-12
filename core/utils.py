"""Utility helpers for experiments and training."""
from __future__ import annotations

import math
import os
import random
from typing import Optional

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0, save_path: Optional[str] = None) -> None:
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_model_state: Optional[dict] = None
        if save_path is None:
            raise ValueError("EarlyStopping requires a 'save_path' to be provided.")

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: torch.nn.Module) -> None:
        save_model(model, self.save_path)
        self.best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_model(model: torch.nn.Module, path: str) -> None:
    """Persist model weights together with configuration metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model_config = {
        "enc_in": getattr(model, "enc_in", None),
        "pred_len": getattr(model, "pred_len", None),
        "seq_len": getattr(model, "seq_len", None),
        "d_model": getattr(model, "d_model", None),
        "d_state": getattr(model, "d_state", None),
        "gate_hidden": getattr(model, "gate_hidden", None),
        "scoring": getattr(model, "scoring", None),
        "recon_weight": getattr(model, "recon_weight", None),
        "min_dt": getattr(model, "min_dt", None),
        "max_dt": getattr(model, "max_dt", None),
        "feedback_delta": getattr(getattr(model, "feedback_cfg", None), "delta_strength", None),
        "feedback_matrix": getattr(getattr(model, "feedback_cfg", None), "matrix_strength", None),
        "dropout": getattr(model, "dropout", None),
    }

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
    }, path)


def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


# ---------------------------------------------------------------------------
# Gaussian utilities used across the Seq-MPS objectives
# ---------------------------------------------------------------------------

def gaussian_kl_divergence(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """KL divergence between two diagonal Gaussians."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    term1 = (var_q + (mu_q - mu_p) ** 2) / var_p
    term2 = logvar_p - logvar_q
    return 0.5 * (term1 + term2 - 1.0)


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood under a diagonal Gaussian."""
    var = torch.exp(logvar)
    return 0.5 * torch.mean((target - mean) ** 2 / var + logvar + math.log(2 * math.pi))


def gaussian_crps(target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form CRPS for a univariate Gaussian."""
    std = torch.exp(0.5 * logvar)
    std = torch.clamp(std, min=1e-6)
    diff = (target - mean) / std
    pdf = torch.exp(-0.5 * diff ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(diff / math.sqrt(2)))
    crps = std * (diff * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))
    return torch.mean(crps)
