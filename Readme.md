# MPS-SSM: Minimal Predictive Sufficiency State Space Model

This repository contains the implementation of MPS-SSM, a theoretically-principled state space model that combines selective mechanisms with information-theoretic regularization based on the principle of minimal predictive sufficiency.

## Overview
 
MPS-SSM addresses the theoretical gap in modern selective state space models (like Mamba) by providing a first-principles approach to designing selective mechanisms. Instead of relying on heuristic designs, MPS-SSM derives its architecture from the fundamental principle that hidden states should be minimal sufficient statistics for predicting the future.

### Key Features

- **Theoretically Grounded**: Based on the principle of minimal predictive sufficiency
- **Robustness Guarantees**: Provable invariance to non-causal disturbances
- **Efficient Implementation**: Leverages modern SSM architectures for linear complexity
- **Comprehensive Evaluation**: Includes robustness testing against multiple noise types

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mps-ssm.git
cd mps-ssm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Download the ETT (Electricity Transformer Temperature) datasets:
```bash
mkdir -p data
cd data
# Download ETTh1.csv, ETTm1.csv, and other datasets
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTh1.csv
wget https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/ETTm1.csv
# Add Weather dataset as needed
cd ..
```

## Running Experiments

The experimental pipeline consists of two stages:

### Stage 1: Lambda Hyperparameter Search
```bash
python main.py --mode lambda_search --config configs/etth_config.yaml
```

This will run experiments across all combinations of datasets, prediction lengths, and lambda values to find the optimal regularization strength.

### Stage 2: Final Training and Evaluation
```bash
python main.py --mode final_run --config configs/etth_config.yaml
```

This stage uses the best lambda values from Stage 1 to train final models and evaluate their performance, including robustness tests.

### Stage 3: Summarize Results
```bash
python scripts/summarize_results.py
```

This generates comprehensive reports including:
- Detailed results tables
- Robustness analysis
- Hyperparameter insights
- Key findings summary

## Project Structure

```
mps-ssm-project/
├── main.py                 # Main orchestrator script
├── run_experiment.py       # Single experiment runner
├── configs/               
│   └── etth_config.yaml   # Experiment configuration
├── data_provider/         
│   ├── data_loader.py     # Dataset and DataLoader implementations
│   └── robustness.py      # Noise injection functions
├── models/                
│   └── mps_ssm.py         # MPS-SSM model implementation
├── core/                  
│   ├── engine.py          # Training and evaluation loops
│   ├── metrics.py         # Evaluation metrics
│   └── utils.py           # Utility functions
├── scripts/               
│   └── summarize_results.py  # Results aggregation
└── results/               # Output directory (auto-created)
```

## Model Architecture

MPS-SSM consists of:
1. **Selective SSM Backbone**: Based on Mamba architecture with input-dependent state transitions
2. **Minimality Regularizer**: Auxiliary decoder implementing information bottleneck principle
3. **Prediction Head**: Maps hidden states to future predictions

The key innovation is the joint optimization of prediction accuracy and state minimality, guided by theoretical guarantees.

## Theoretical Contributions

1. **Predictive Sufficiency Principle**: Hidden states should capture all and only the information necessary for prediction
2. **Provable Robustness**: Theoretical guarantee of invariance to non-causal disturbances
3. **Information-Theoretic Objective**: Principled regularization based on mutual information minimization

## Citation

If you use this code in your research, please cite:
```bibtex
@article{mps-ssm2024,
  title={MPS-SSM: Minimal Predictive Sufficiency State Space Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

# requirements.txt

# Core dependencies
torch>=2.0.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
pyyaml>=6.0
tqdm>=4.65.0

# For results visualization
matplotlib>=3.6.0
seaborn>=0.12.0
tabulate>=0.9.0

# Development tools (optional)
pytest>=7.2.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0

# Hardware optimization (optional)
# For CUDA 11.8
# torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
