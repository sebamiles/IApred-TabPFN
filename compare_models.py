#!/usr/bin/env python3
"""
Compare SVM with Multiple TabPFN Models

This script compares the SVM model with multiple TabPFN models and generates:
1. Comprehensive metrics CSV file
2. Comparison plots (heatmap, ROC curves, calibration plots)

Usage: python compare_models.py --tabpfn all 529 100
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_script(script_path, args=None, description=""):
    """Run a Python script with optional arguments"""
    if description:
        print(f"\n{'='*60}")
        print(f"🔧 {description}")
        print('='*60)

    # Get script directory and name
    script_dir = os.path.dirname(str(script_path))
    script_name = os.path.basename(str(script_path))
    
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend([str(arg) for arg in args])

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        print("✅ Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def validate_models(tabpfn_models):
    """Check if all required models exist"""
    missing_models = []

    # Check SVM model
    svm_model = Path("svm_training/models/IApred_SVM.joblib")
    if not svm_model.exists():
        missing_models.append("SVM model (svm_training/models/IApred_SVM.joblib)")

    # Check TabPFN models
    for k in tabpfn_models:
        model_suffix = 'all_features' if k == 'all' else f'{k}_features'
        tabpfn_model = Path(f"tabpfn_training/models/{model_suffix}/IApred_TabPFN.joblib")
        if not tabpfn_model.exists():
            missing_models.append(f"TabPFN-{k} model (tabpfn_training/models/{model_suffix}/IApred_TabPFN.joblib)")

    if missing_models:
        print("❌ Missing required models:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nPlease train the missing models first using:")
        print("  - SVM: python train_svm_pipeline.py")
        print("  - TabPFN: python train_tabpfn_pipeline.py --k [all|529|100]")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description='Compare SVM with TabPFN models')
    parser.add_argument('--tabpfn', nargs='+', required=True,
                       choices=['all', '529', '100'],
                       help='TabPFN models to compare (e.g., --tabpfn all 529 100)')
    args = parser.parse_args()

    print("🔍 IApred Model Comparison")
    print("=" * 40)
    print(f"Comparing SVM with TabPFN models: {', '.join(args.tabpfn)}")
    print()

    # Validate that all models exist
    if not validate_models(args.tabpfn):
        return False

    # Ensure comparison directory exists
    comparison_scripts_dir = Path("comparison/scripts")
    if not comparison_scripts_dir.exists():
        print("❌ Comparison scripts directory not found!")
        return False

    # Step 1: Generate evaluations for all models
    if not run_script(
        comparison_scripts_dir / "generate_all_evaluations.py",
        description="Step 1/2: Generating External Evaluations"
    ):
        return False

    # Step 2: Generate comparison plots
    if not run_script(
        comparison_scripts_dir / "generate_comparison_plots.py",
        description="Step 2/2: Generating Comparison Plots"
    ):
        return False

    print("\n" + "="*60)
    print("🎉 Model Comparison Completed!")
    print("=" * 60)
    print("📊 Results saved in: comparison/results/")
    print("   ├── all_models_comparison.csv    # Comprehensive metrics")
    print("   ├── performance_heatmap.png      # Metrics heatmap")
    print("   ├── roc_comparison.png           # ROC curves")
    print("   └── calibration_comparison.png   # Calibration plots")
    print("\n📈 Models compared:")
    print("   ├── SVM")
    for k in args.tabpfn:
        print(f"   └── TabPFN-{k}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
