#!/usr/bin/env python3
"""
Complete SVM Training Pipeline

This script runs the SVM training pipeline with optional optimization:
1. Feature selection optimization (Find_best_k.py) - optional
2. Hyperparameter optimization (Optimize_C_and_gamma.py) - optional
3. Final model training (generate_and_save_models.py)
4. Cross-validation (10-fold, LOCO)

Usage:
  # Full optimization pipeline (default)
  python train_svm_pipeline.py

  # Skip optimization, use provided parameters
  python train_svm_pipeline.py --k 529 --c 1.0 --gamma 0.001 --skip-optimization
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from tqdm import tqdm

def run_script(script_path, args=None, description=""):
    """Run a Python script with optional arguments"""
    if description:
        print(f"\n{'='*60}")
        print(f"🔧 {description}")
        print('='*60)

    # Ensure results and models directories exist
    results_dir = Path("svm_training/results")
    results_dir.mkdir(exist_ok=True, parents=True)
    models_dir = Path("svm_training/models")
    models_dir.mkdir(exist_ok=True, parents=True)

    cmd = [sys.executable, str(script_path)]  # Convert Path to string
    if args:
        cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")

    try:
        # Run from the script's directory so relative paths work correctly
        script_dir = script_path.parent
        script_name = script_path.name
        cmd = [sys.executable, script_name]  # Use just the script name
        if args:
            cmd.extend(args)
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print("✅ Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete SVM training pipeline')
    parser.add_argument('--k', type=int, default=529,
                       help='Number of features to select (default: 529)')
    parser.add_argument('--c', type=float, default=1.0,
                       help='SVM C parameter (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.001,
                       help='SVM gamma parameter (default: 0.001)')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='Skip feature selection and hyperparameter optimization, use provided parameters directly')
    args = parser.parse_args()

    print("🚀 IApred SVM Training Pipeline")
    print("=" * 50)
    print(f"Parameters: k={args.k}, C={args.c}, gamma={args.gamma}")
    print()

    # Ensure directories exist
    svm_scripts_dir = Path("svm_training/scripts")
    if not svm_scripts_dir.exists():
        print("❌ SVM scripts directory not found!")
        return False

    # Check if we should skip optimization
    if args.skip_optimization:
        print(f"🔧 Using provided parameters: k={args.k}, C={args.c}, gamma={args.gamma}")
        print("⏭️  Skipping feature selection and hyperparameter optimization")
        steps = [
            ("Model Training", svm_scripts_dir / "generate_and_save_models.py",
             ["--k", str(args.k), "--c", str(args.c), "--gamma", str(args.gamma)]),
            ("10-fold CV", svm_scripts_dir / "10fold_CV.py",
             ["--k", str(args.k), "--c", str(args.c), "--gamma", str(args.gamma)]),
            ("LOCO CV", svm_scripts_dir / "LOCO-CV.py",
             ["--k", str(args.k), "--c", str(args.c), "--gamma", str(args.gamma)])
        ]
    else:
        print("🔍 Running full optimization pipeline (feature selection + hyperparameter tuning)")
        steps = [
            ("Feature Selection", svm_scripts_dir / "Find_best_k.py", []),
            ("Hyperparameter Optimization", svm_scripts_dir / "Optimize_C_and_gamma.py", ["--k", str(args.k)]),
            ("Model Training", svm_scripts_dir / "generate_and_save_models.py",
             ["--k", str(args.k), "--c", str(args.c), "--gamma", str(args.gamma)]),
            ("10-fold CV", svm_scripts_dir / "10fold_CV.py",
             ["--k", str(args.k), "--c", str(args.c), "--gamma", str(args.gamma)]),
            ("LOCO CV", svm_scripts_dir / "LOCO-CV.py",
             ["--k", str(args.k), "--c", str(args.c), "--gamma", str(args.gamma)])
        ]

    # Run pipeline with progress bar
    with tqdm(total=len(steps), desc="SVM Pipeline Progress", unit="step") as pbar:
        for step_name, script_path, script_args in steps:
            pbar.set_description(f"Running {step_name}")
            start_time = time.time()

            success = run_script(script_path, script_args, description=f"Running {step_name}")

            elapsed = time.time() - start_time
            if success:
                pbar.set_postfix({"status": "✅", "time": f"{elapsed:.1f}s"})
            else:
                pbar.set_postfix({"status": "❌", "time": f"{elapsed:.1f}s"})
                if step_name in ["10-fold CV", "LOCO CV"]:
                    print(f"⚠️  {step_name} failed, but continuing...")

            pbar.update(1)

    print("\n" + "="*60)
    print("🎉 SVM Training Pipeline Completed!")
    print("=" * 60)
    print("📁 Results saved in: svm_training/results/")
    print("🤖 Model saved in: svm_training/models/")
    print("\nTo run external evaluation:")
    print("  cd svm_training/scripts && python External_Evaluation.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
