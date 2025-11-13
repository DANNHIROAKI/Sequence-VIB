# Sequence-VIB: Seq-MPS with Budgeted Selectivity

Sequence-VIB implements the **Seq-MPS** architecture introduced in the refreshed theory (Sections 1–5) and the experimental protocol described in Sections 6–11. The codebase provides a reproducible pipeline that covers multi-seed training, robustness diagnostics, resource accounting, and result summarisation.

- [1. Prerequisites](#1-prerequisites)
- [2. Project Layout](#2-project-layout)
- [3. Dataset Preparation](#3-dataset-preparation)
- [4. Configuration Schema](#4-configuration-schema)
- [5. Running Experiments](#5-running-experiments)
  - [5.1 Single dataset / quick sanity check](#51-single-dataset--quick-sanity-check)
  - [5.2 Multi-dataset benchmark sweep](#52-multi-dataset-benchmark-sweep)
  - [5.3 Full two-stage reproduction script](#53-full-two-stage-reproduction-script)
- [6. Monitoring Outputs & Analysing Results](#6-monitoring-outputs--analysing-results)
- [7. Customisation Tips & Troubleshooting](#7-customisation-tips--troubleshooting)
- [8. License](#8-license)

## 1. Prerequisites

1. **Python** ≥ 3.9 (3.10 tested). Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support that matches your drivers; CPU-only mode also works for smoke tests.
2. **Create an isolated environment** and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The requirements file captures NumPy/Pandas/PyTorch-lightweight utilities used by the training/evaluation stack. If you maintain a custom CUDA build of PyTorch, install it first and then `pip install -r requirements.txt --no-deps`.

3. **GPU allocation (optional)**. Export `CUDA_VISIBLE_DEVICES` or pass `--num_gpus`/`--gpu_id` CLI flags. Mixed precision is supported automatically via PyTorch AMP when CUDA is available.

## 2. Project Layout

```
Sequence-VIB/
├── configs/seq_mps_suite.yaml   # Canonical benchmark definition (datasets × horizons × seeds)
├── core/                        # Training loop, evaluation helpers, utils
├── data_provider/               # Dataset loaders, preprocessing, robustness perturbations
├── models/mps_ssm.py            # Seq-MPS selective state space model
├── run_experiment.py            # Runs a single dataset/horizon across multiple seeds
├── main.py                      # Multi-GPU dispatcher that sweeps all dataset/horizon pairs
├── scripts/summarize_results.py # Aggregates per-run JSON into CSV/Markdown tables
└── results/                     # Created automatically; stores checkpoints, logs, summaries
```

Keep raw datasets under `data/`. Logs emitted by orchestration helpers appear in `results/dispatch_*.log` and per-seed artefacts land under `results/benchmarks/<dataset>/H<pred_len>/`.

## 3. Dataset Preparation

Seq-MPS targets the standard Long-Term Series Forecasting (LTSF) suite. Place CSV files under `data/<dataset>/`. The following snippet downloads the ETT family directly from the original repository:

```bash
mkdir -p data/ETT-small
wget -P data/ETT-small https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv
wget -P data/ETT-small https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh2.csv
wget -P data/ETT-small https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm1.csv
wget -P data/ETT-small https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm2.csv
```

Repeat the process for the remaining datasets:

| Dataset Directory | Required Files | Notes |
|-------------------|----------------|-------|
| `data/weather`    | `weather.csv`  | Original Weather benchmark from Autoformer/LTSF suites. |
| `data/electricity`| `electricity.csv` | Hourly demand; loaders perform z-score normalisation using the training split only. |
| `data/traffic`    | `traffic.csv`  | Set `enc_in=862` in the config (already provided). |
| `data/exchange`   | `exchange_rate.csv` | Convert Excel source to CSV if needed; keep header row intact. |

The loaders in `data_provider/data_loader.py` automatically:

- split data according to the ratios in the config (train/val/test),
- fill missing values (forward/backward),
- optionally detrend/standardise features,
- emit masks for structured-missing robustness sweeps.

## 4. Configuration Schema

All reproducibility settings live in [`configs/seq_mps_suite.yaml`](configs/seq_mps_suite.yaml). Key sections:

- `experiment`: lists datasets, their root paths, encoder dimensions, history lengths (`seq_len`), prediction horizons, and random seeds. You can override or drop datasets to reduce runtime.
- `training`: optimiser (Adam/AdamW), learning rate, gradient clipping, patience, batch size, KL-rate budgets (`rate_budget_*`) and dual-ascent hyper-parameters (`dual_step_*`).
- `model`: SSM widths (`d_model`, `d_state`), gating hidden sizes, discretisation limits, dropout, and spectral radius clamps.
- `evaluation`: latency probe warm-up iterations and measurement repetitions.
- `robustness`: toggles perturbations implemented in `data_provider/robustness.py` (Gaussian, impulse, spurious-correlation, structured missingness) plus their magnitudes.

To customise experiments:

1. Duplicate the YAML file, e.g., `configs/my_run.yaml`.
2. Adjust dataset blocks or horizons of interest.
3. Reference the new file when invoking `run_experiment.py` or `main.py`.

## 5. Running Experiments

### 5.1 Single dataset / quick sanity check

Use `run_experiment.py` when you want to iterate on one dataset-horizon combination. The runner automatically loops through all seeds defined in the config, performs early stopping, evaluates robustness, measures latency, and writes a `summary.json` aggregating mean/std statistics.

```bash
python run_experiment.py \
  --dataset ETTh1 \
  --pred_len 336 \
  --config configs/seq_mps_suite.yaml \
  --gpu_id 0
```

Arguments:

- `--dataset`: key under `experiment.datasets`.
- `--pred_len`: one of the horizons specified in `experiment.horizons`.
- `--config`: path to the YAML file (required).
- `--gpu_id`: local CUDA index (default `0`). Set to `-1` for CPU tests.

Outputs are stored in `results/benchmarks/ETTh1/H336/`, including checkpoints (`checkpoints/seed*.pth`) and per-seed JSON logs.

### 5.2 Multi-dataset benchmark sweep

`main.py` dispatches all dataset × horizon jobs concurrently across available GPUs. It spawns independent `run_experiment.py` processes and streams their logs to `results/dispatch_*.log`.

```bash
python main.py --config configs/seq_mps_suite.yaml --num_gpus 2
```

- `--num_gpus` controls how many concurrent workers to launch. The orchestrator cycles through the GPU IDs `[0, num_gpus-1]`.
- Ensure the `data/` tree already contains every dataset referenced in the config; the orchestrator does not download files automatically.

### 5.3 Full two-stage reproduction script

For a turnkey reproduction of the paper results (λ-search followed by frozen-model evaluation and summarisation), run the sequential helper script. It executes the orchestrator in *lambda-search* mode first and then in *test-only* mode, finally calling the summariser.

```bash
bash run_experiments-Copy1.sh
```

Before launching, update the file if you wish to:

1. Point to custom configs (replace `configs/ett.yaml`, etc., with your variants).
2. Reduce `NUM_GPUS` to match your hardware.
3. Comment out dataset groups you do not need.

The script performs sanity checks (presence of config files, dataset directory, and GPU visibility) and prints elapsed times for each stage. Intermediate artefacts are saved under `results/lambda_search*/` while final evaluation logs appear in `results/final_runs/logs/`.

> **Note:** `main.py` currently exposes only `train_eval` mode. The staged script assumes future extensions (`lambda_search` / `test_only`). If you only rely on the public CLI, keep using `main.py` with the default mode.

## 6. Monitoring Outputs & Analysing Results

- **Per-epoch traces:** Each `seed_*.json` stores training/validation losses, KL rates, λ values, throughput, and memory statistics for reproducibility.
- **Aggregated summary:** `summary.json` combines metrics across seeds, including MSE/MAE/NLL/CRPS, throughput, latency, and robustness degradations.
- **Checkpoints:** Located under `results/benchmarks/<dataset>/H<pred_len>/checkpoints/`. Each file contains the PyTorch `state_dict` and configuration snapshot, making it straightforward to resume or fine-tune.

To generate publication-ready tables (Sections 8–11), run:

```bash
python scripts/summarize_results.py \
  --root results/benchmarks \
  --output results/summary_tables
```

The command writes CSV and Markdown tables (`accuracy.*`, `resources.*`, `robustness.*`). These files feed directly into papers or dashboards.

## 7. Customisation Tips & Troubleshooting

- **Overriding horizons or seeds:** Edit `experiment.horizons` and `experiment.seeds` in the YAML file. The orchestrator automatically iterates over the Cartesian product.
- **Changing optimiser or learning rate:** Modify `training.optimizer`. Unsupported optimiser names raise explicit errors inside `run_experiment.py`.
- **Robustness sweeps:** Toggle individual perturbations (Gaussian/impulse/spurious/structured missingness) under `robustness`. Setting a magnitude to `0.0` effectively disables that test.
- **Latency probes:** Increase `evaluation.latency.reps` for more stable measurements. Latency is computed with inference-only forward passes using a single batch.
- **GPU memory errors:** Lower `training.batch_size` or `model.d_model` / `model.d_state` and rerun. Check `seed_*.json` for `peak_mem_gb` to gauge usage.
- **Resuming / inspecting failures:** Dispatcher logs live in `results/dispatch_*`. Each process prints configuration info, training progress, and traceback if something fails.

## 8. License

This project is released under the MIT License (see `LICENSE`). Open an issue if you encounter problems reproducing the reported results or have questions about extending Seq-MPS.
