#!/usr/bin/env python3
"""
TabPFN Training Pipeline with Cross-Validation

This script trains a TabPFN model with specified feature count and runs 10-fold and LOCO CVs.

Usage: python train_tabpfn_pipeline.py --k [all|529|100]
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
    results_dir = Path("tabpfn_training/results")
    results_dir.mkdir(exist_ok=True, parents=True)
    models_dir = Path("tabpfn_training/models")
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
    parser = argparse.ArgumentParser(description='TabPFN training pipeline with CV')
    parser.add_argument('--k', type=str, required=True, choices=['all', '529', '100'],
                       help='Number of features to use (required: all, 529, or 100)')
    args = parser.parse_args()

    print("🚀 IApred TabPFN Training Pipeline")
    print("=" * 50)
    print(f"Feature count: {args.k}")
    print()

    # Ensure directories exist
    tabpfn_scripts_dir = Path("tabpfn_training/scripts")
    if not tabpfn_scripts_dir.exists():
        print("❌ TabPFN scripts directory not found!")
        return False

    # Define pipeline steps
    steps = [
        (f"Training TabPFN ({args.k} features)", tabpfn_scripts_dir / "train_model.py", ["--k", args.k]),
        (f"10-fold CV ({args.k} features)", tabpfn_scripts_dir / "10fold_cv.py", ["--k", args.k]),
       # (f"LOCO CV ({args.k} features)", tabpfn_scripts_dir / "loco_cv.py", ["--k", args.k])
    ]

    # Run pipeline with progress bar
    with tqdm(total=len(steps), desc="TabPFN Pipeline Progress", unit="step") as pbar:
        for step_name, script_path, script_args in steps:
            pbar.set_description(f"Running {step_name}")
            start_time = time.time()

            success = run_script(script_path, script_args, description=f"Running {step_name}")

            elapsed = time.time() - start_time
            if success:
                pbar.set_postfix({"status": "✅", "time": f"{elapsed:.1f}s"})
            else:
                pbar.set_postfix({"status": "❌", "time": f"{elapsed:.1f}s"})
                if "CV" in step_name:
                    print(f"⚠️  {step_name} failed, but continuing...")

            pbar.update(1)

    print("\n" + "="*60)
    print("🎉 TabPFN Training Pipeline Completed!")
    print("=" * 60)
    print(f"📁 Results saved in: tabpfn_training/results/{args.k}_features/")
    print(f"🤖 Model saved in: tabpfn_training/models/{args.k}_features/")
    print(f"\nTo run external evaluation:")
    print(f"  cd tabpfn_training/scripts && python external_evaluation.py --k {args.k}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
