#!/usr/bin/env python3
"""
MPS-SSM Project: Main Control Script
Modified with a dynamic scheduler for maximum GPU utilization.
"""

import os
import sys
import time
import yaml
import json
import argparse
import itertools
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import torch.multiprocessing as mp

class ExperimentOrchestrator:
    """Manages experiment execution and coordination with a dynamic scheduler."""
    
    def __init__(self, config_path: str, mode: str, num_gpus: int = 8):
        self.config = self._load_config(config_path)
        self.mode = mode
        self.num_gpus = num_gpus
        self.results_dir = Path("results")
        self._setup_directories()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f: return yaml.safe_load(f)
            
    def _setup_directories(self):
        dirs = [
            self.results_dir / "lambda_search", self.results_dir / "lambda_search_models",
            self.results_dir / "final_runs" / "logs", Path("logs")
        ]
        for d in dirs: d.mkdir(parents=True, exist_ok=True)
    
    # 在 ExperimentOrchestrator 类里添加这个新函数
    def _find_best_lambda_for_test(self, dataset: str, pred_len: int) -> float:
        """
        为给定的数据集和预测长度查找最佳lambda值。
        """
        search_dir = self.results_dir / "lambda_search"
        best_lambda = 0.0
        best_val_loss = float('inf')

        # 构造文件匹配模式
        pattern = f"{dataset}_{pred_len}_*.json"
        
        # 遍历所有匹配的lambda搜索结果文件
        for f in search_dir.glob(pattern):
            try:
                with open(f, 'r') as res_file:
                    data = json.load(res_file)
                    candidate_val = data.get('val_pred_loss', data.get('val_loss', float('inf')))
                    if candidate_val < best_val_loss:
                        best_val_loss = candidate_val
                        best_lambda = data.get('final_lambda', data.get('initial_lambda', 0.0))
            except (json.JSONDecodeError, KeyError):
                # 如果文件损坏或格式不正确，则跳过
                continue
                
        return best_lambda
    
    def _generate_experiment_grid(self) -> List[Dict[str, Any]]:
        datasets = self.config['datasets']
        pred_lens = self.config['prediction_lengths']
        
        if self.mode == 'lambda_search':
            if 'lambda_search' not in self.config or 'values' not in self.config['lambda_search']:
                raise ValueError("Lambda values not found in config for lambda_search mode.")
            lambda_vals = self.config['lambda_search']['values']
            grid = itertools.product(datasets, pred_lens, lambda_vals)
            return [{'dataset': d, 'pred_len': p, 'lambda_val': l, 'mode': 'lambda_search'} for d, p, l in grid]
        else: # test_only
            experiments = []
            print("Finding best lambda values for test mode...")
            grid = itertools.product(datasets, pred_lens)
            for d, p in grid:
                # 为每个测试任务查找最佳lambda
                best_lambda = self._find_best_lambda_for_test(d, p)
                
                # 将带有真实lambda值的任务添加到列表中
                experiments.append({
                    'dataset': d, 
                    'pred_len': p, 
                    'lambda_val': best_lambda,  # 使用找到的最佳lambda
                    'mode': 'test_only'
                })
            print("Finished finding lambda values.")
            return experiments
            
    def _launch_experiment(self, exp_config: Dict[str, Any], gpu_id: int):
        cmd = [
            sys.executable, "run_experiment.py",
            "--dataset", exp_config['dataset'], "--pred_len", str(exp_config['pred_len']),
            "--lambda_val", str(exp_config['lambda_val']), "--mode", exp_config['mode'],
            "--gpu_id", str(gpu_id), "--config", self.config['config_file']
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"--- ERROR on GPU {gpu_id} for experiment: {exp_config} ---")
            print(f"--- STDERR ---\n{e.stderr}\n--- END OF ERROR ---")
        
    def run(self):
        """Executes all experiments using a dynamic scheduling loop."""
        experiments = self._generate_experiment_grid()
        print(f"Total experiments to run: {len(experiments)}")
        if not experiments: return

        task_queue = list(experiments)
        # Dictionary to track the process running on each GPU: {gpu_id: Process}
        active_processes = {}

        print("Starting dynamic scheduler...")
        while task_queue or active_processes:
            # Check for finished processes and free up their GPUs
            finished_gpus = []
            for gpu_id, process in active_processes.items():
                if not process.is_alive():
                    process.join()  # Clean up the completed process
                    finished_gpus.append(gpu_id)
            
            for gpu_id in finished_gpus:
                del active_processes[gpu_id]

            # Launch new tasks on any available GPU
            while len(active_processes) < self.num_gpus and task_queue:
                # Find the next available GPU id
                available_gpu_id = -1
                for i in range(self.num_gpus):
                    if i not in active_processes:
                        available_gpu_id = i
                        break
                
                if available_gpu_id != -1:
                    exp_config = task_queue.pop(0)
                    print(f"Dispatching to free GPU {available_gpu_id}: {exp_config['dataset']}, len={exp_config['pred_len']}, lambda={exp_config['lambda_val']}")
                    p = mp.Process(target=self._launch_experiment, args=(exp_config, available_gpu_id))
                    p.start()
                    active_processes[available_gpu_id] = p
                else:
                    # This case should not be reached if len(active_processes) < self.num_gpus
                    break 

            # If all GPUs are busy and there are still tasks, wait a bit before checking again.
            if len(active_processes) == self.num_gpus and task_queue:
                time.sleep(5) # Polling interval to check for finished jobs

            # If no tasks are left in the queue, just wait for the last running jobs to finish.
            if not task_queue:
                time.sleep(1)

        print(f"\nAll experiments completed for mode: {self.mode}")

def main():
    parser = argparse.ArgumentParser(description="MPS-SSM Experiment Orchestrator")
    parser.add_argument('--mode', type=str, required=True, choices=['lambda_search', 'test_only'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=8)
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    config['config_file'] = args.config
    with open(args.config, 'w') as f: yaml.dump(config, f)
    
    orchestrator = ExperimentOrchestrator(args.config, args.mode, args.num_gpus)
    orchestrator.run()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
