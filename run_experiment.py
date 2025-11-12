#!/usr/bin/env python3
"""Experiment runner for Seq-MPS with budgeted selectivity."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from core.engine import evaluate, train_one_epoch
from core.utils import EarlyStopping, set_random_seed, save_model
from data_provider.data_loader import TimeSeriesDataset, get_dataloader
from data_provider.robustness import add_impulse_noise, add_spurious_correlation
from models.mps_ssm import MPSSSM


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        with open(args.config, "r") as f:
            self.config = yaml.safe_load(f)

        set_random_seed(self.config["training"]["seed"])
        self._setup_paths()

    # ------------------------------------------------------------------
    def _setup_paths(self) -> None:
        if self.args.mode == "lambda_search":
            self.result_dir = Path("results/lambda_search")
            self.result_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir = Path("results/lambda_search_models")
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = self.model_dir / f"{self.args.dataset}_{self.args.pred_len}_{self.args.lambda_val}.pth"
            self.result_file = self.result_dir / f"{self.args.dataset}_{self.args.pred_len}_{self.args.lambda_val}.json"
        else:
            self.result_dir = Path("results/final_runs")
            self.model_dir = Path("results/lambda_search_models")
            self.log_file = self.result_dir / "logs" / f"{self.args.dataset}_{self.args.pred_len}.json"
            (self.result_dir / "logs").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _data_config(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "dataset": self.args.dataset,
            "pred_len": self.args.pred_len,
            "batch_size": cfg["training"]["batch_size"],
            "seq_len": cfg["data"]["seq_len"],
            "data_path": cfg["data"]["data_path"],
        }

    def load_data(self, mode: str) -> Any:
        data_cfg = self._data_config()
        if mode == "train":
            train_loader = get_dataloader(data_cfg, mode="train")
            val_loader = get_dataloader(data_cfg, mode="val")
            return train_loader, val_loader
        if mode == "test":
            return get_dataloader(data_cfg, mode="test")
        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    def _enc_in(self) -> int:
        if self.args.dataset.startswith("ETT"):
            return self.config["data"].get("features", 7)
        if self.args.dataset == "weather":
            return 21
        if self.args.dataset == "traffic":
            return 862
        if self.args.dataset == "electricity":
            return 321
        raise ValueError(f"Unknown dataset: {self.args.dataset}")

    def _create_model(self) -> MPSSSM:
        model_cfg = self.config["model"].copy()
        return MPSSSM(
            enc_in=self._enc_in(),
            pred_len=self.args.pred_len,
            seq_len=self.config["data"]["seq_len"],
            d_model=model_cfg.get("d_model", 256),
            d_state=model_cfg.get("d_state", 64),
            gate_hidden=model_cfg.get("gate_hidden", 128),
            scoring=self.config["training"].get("scoring", "nll"),
            recon_weight=self.config["training"].get("recon_weight", 0.1),
            min_dt=model_cfg.get("min_dt", 0.01),
            max_dt=model_cfg.get("max_dt", 1.0),
            feedback_delta=model_cfg.get("feedback_delta", 0.2),
            feedback_matrix=model_cfg.get("feedback_matrix", 0.2),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(self.device)

    # ------------------------------------------------------------------
    def train_model(self, model: MPSSSM, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[float, float]:
        cfg = self.config["training"]
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

        dual_state = {
            "lambda": self.args.lambda_val,
            "rate_budget": cfg.get("rate_budget", 0.05),
            "dual_step": cfg.get("dual_step", 0.05),
        }

        early_stopping = EarlyStopping(patience=cfg["patience"], save_path=str(self.model_path))
        best_val = float("inf")

        for epoch in range(cfg["max_epochs"]):
            train_stats = train_one_epoch(model, train_loader, optimizer, self.device, dual_state)
            val_stats = evaluate(model, val_loader, self.device)

            val_loss = val_stats["pred_loss"]
            early_stopping(val_loss, model)
            best_val = min(best_val, val_loss)

            print(
                f"Epoch {epoch+1}/{cfg['max_epochs']} | "
                f"train_loss={train_stats['loss']:.4f} | train_rate={train_stats['rate']:.4f} | "
                f"val_pred={val_stats['pred_loss']:.4f} | lambda={dual_state['lambda']:.4f}"
            )

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        if early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)
        else:
            save_model(model, str(self.model_path))

        return best_val, dual_state["lambda"]

    # ------------------------------------------------------------------
    def _evaluate_robustness(self, model: MPSSSM, base_mse: float) -> Dict[str, float]:
        data_cfg = self._data_config()
        data_file = os.path.join(data_cfg["data_path"], f"{self.args.dataset}.csv")

        clean_dataset = TimeSeriesDataset(
            data_path=data_file,
            mode="test",
            seq_len=data_cfg["seq_len"],
            pred_len=data_cfg["pred_len"],
            dataset_type=self.args.dataset.split("_")[0],
        )

        noise_scenarios = {
            "impulse": lambda arr: add_impulse_noise(arr, scale=clean_dataset.scaler.scale_),
            "spurious": lambda arr: add_spurious_correlation(arr, scale=clean_dataset.scaler.scale_),
        }

        robustness = {}
        for name, noise_fn in noise_scenarios.items():
            noisy_dataset = TimeSeriesDataset(
                data_path=data_file,
                mode="test",
                seq_len=data_cfg["seq_len"],
                pred_len=data_cfg["pred_len"],
                dataset_type=self.args.dataset.split("_")[0],
                noise_fn=noise_fn,
            )
            loader = DataLoader(noisy_dataset, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=4)
            metrics = evaluate(model, loader, self.device)
            mse = metrics["mse"]
            degradation = (mse - base_mse) / base_mse if base_mse > 0 else 0.0
            robustness[f"{name}_mse"] = float(mse)
            robustness[f"{name}_degradation"] = float(degradation)
        return robustness

    # ------------------------------------------------------------------
    def run(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file_path = log_dir / f"gpu_{self.args.gpu_id}.log"

        original_stdout, original_stderr = sys.stdout, sys.stderr
        with open(log_file_path, "w") as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            try:
                print(f"--- Experiment start {datetime.now().isoformat()} ---")
                print(f"Dataset: {self.args.dataset} | Horizon: {self.args.pred_len} | Mode: {self.args.mode}")

                if self.args.mode == "lambda_search":
                    train_loader, val_loader = self.load_data("train")
                    model = self._create_model()
                    best_val, final_lambda = self.train_model(model, train_loader, val_loader)

                    test_loader = self.load_data("test")
                    test_metrics = evaluate(model, test_loader, self.device)

                    results = {
                        "dataset": self.args.dataset,
                        "pred_len": self.args.pred_len,
                        "initial_lambda": self.args.lambda_val,
                        "final_lambda": final_lambda,
                        "val_pred_loss": float(best_val),
                        "test_mse": float(test_metrics["mse"]),
                        "test_mae": float(test_metrics["mae"]),
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(self.result_file, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"Lambda search complete. Saved to {self.result_file}")
                else:
                    search_dir = Path("results/lambda_search")
                    best_file = None
                    best_val = float("inf")
                    for candidate in search_dir.glob(f"{self.args.dataset}_{self.args.pred_len}_*.json"):
                        with open(candidate, "r") as f:
                            data = json.load(f)
                        if data.get("val_pred_loss", float("inf")) < best_val:
                            best_val = data["val_pred_loss"]
                            best_file = data

                    if best_file is None:
                        raise FileNotFoundError("No lambda search results found. Run lambda_search first.")

                    final_lambda = best_file.get("final_lambda", best_file.get("initial_lambda", 0.1))
                    model_path = self.model_dir / f"{self.args.dataset}_{self.args.pred_len}_{best_file['initial_lambda']}.pth"
                    if not model_path.exists():
                        raise FileNotFoundError(f"Model checkpoint missing: {model_path}")

                    model = self._create_model()
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(checkpoint["model_state_dict"])

                    test_loader = self.load_data("test")
                    test_metrics = evaluate(model, test_loader, self.device)
                    robustness = self._evaluate_robustness(model, base_mse=test_metrics["mse"])

                    final_results = {
                        "dataset": self.args.dataset,
                        "pred_len": self.args.pred_len,
                        "best_lambda": final_lambda,
                        "test_mse": float(test_metrics["mse"]),
                        "test_mae": float(test_metrics["mae"]),
                        "test_pred_loss": float(test_metrics["pred_loss"]),
                        "robustness": robustness,
                        "timestamp": datetime.now().isoformat(),
                        "checkpoint": str(model_path),
                    }
                    with open(self.log_file, "w") as f:
                        json.dump(final_results, f, indent=2)
                    print(f"Final evaluation logged to {self.log_file}")

                print(f"--- Experiment completed {datetime.now().isoformat()} ---")
            except Exception:
                import traceback

                print(traceback.format_exc())
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Seq-MPS experiment")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--lambda_val", type=float, required=True)
    parser.add_argument("--mode", type=str, choices=["lambda_search", "test_only"], required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    runner = ExperimentRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
