#!/usr/bin/env python
"""
Run full experiment pipeline for MS-HGNN
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_experiment(config_path: str, data_dir: str, output_dir: str, 
                   num_epochs: int = 100, batch_size: int = 16,
                   device: str = 'cuda'):
    """
    Run a complete experiment
    
    Args:
        config_path: Path to configuration file
        data_dir: Data directory
        output_dir: Output directory
        num_epochs: Number of epochs
        batch_size: Batch size
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"MS-HGNN Experiment")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Config: {config_path}")
    print(f"Data: {data_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Step 1: Train the model
    print("\n[Step 1] Training MS-HGNN...")
    train_cmd = [
        "python", "main.py", "train",
        "--config", config_path,
        "--data_dir", data_dir,
        "--checkpoint_dir", str(run_dir / "checkpoints"),
        "--epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--device", device
    ]
    
    subprocess.run(train_cmd, check=True)
    
    # Find the best checkpoint
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("best_model_epoch_*.pt"))
    if not checkpoints:
        print("No checkpoint found!")
        return
    
    best_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\nBest checkpoint: {best_checkpoint}")
    
    # Step 2: Evaluate on test set
    print("\n[Step 2] Evaluating on test set...")
    eval_cmd = [
        "python", "main.py", "eval",
        "--model_path", str(best_checkpoint),
        "--data_dir", data_dir,
        "--output_dir", str(run_dir / "results")
    ]
    
    subprocess.run(eval_cmd, check=True)
    
    # Step 3: External validation if TCGA data is available
    tcga_dir = Path(data_dir) / "tcga_luad"
    if tcga_dir.exists():
        print("\n[Step 3] Running external validation on TCGA-LUAD...")
        eval_ext_cmd = [
            "python", "main.py", "eval",
            "--model_path", str(best_checkpoint),
            "--data_dir", data_dir,
            "--output_dir", str(run_dir / "external_results"),
            "--external_dir", str(tcga_dir)
        ]
        subprocess.run(eval_ext_cmd, check=True)
    
    # Step 4: Generate summary report
    print("\n[Step 4] Generating summary report...")
    generate_report(run_dir)
    
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)


def generate_report(run_dir: Path):
    """
    Generate a summary report
    
    Args:
        run_dir: Run directory
    """
    results_dir = run_dir / "results"
    
    report_path = run_dir / "experiment_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MS-HGNN Experiment Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Check for metrics file
        metrics_path = results_dir / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as mf:
                metrics = json.load(mf)
            
            f.write("Test Set Metrics:\n")
            f.write("-" * 30 + "\n")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.4f}\n")
        
        # Check for external metrics
        ext_results = run_dir / "external_results"
        ext_metrics = ext_results / "external_metrics.json"
        if ext_metrics.exists():
            with open(ext_metrics, 'r') as mf:
                ext_metrics_data = json.load(mf)
            
            f.write("\nTCGA-LUAD External Validation:\n")
            f.write("-" * 30 + "\n")
            for key, value in ext_metrics_data.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run MS-HGNN experiment')
    parser.add_argument('--config', type=str, default='configs/paper_config.json',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    run_experiment(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
