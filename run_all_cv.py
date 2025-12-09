#!/usr/bin/env python3
"""
Master script to run all cross-validations for TabPFN models with different feature sets.
This script runs 10-fold CV and LOCO CV for all three TabPFN variants.
"""

import os
import sys
import subprocess
import argparse

def run_cv_script(script_name, k_value):
    """Run a cross-validation script with the specified k value"""
    cmd = [sys.executable, f"tabpfn_training/scripts/{script_name}.py", "--k", str(k_value)]
    print(f"\n{'='*60}")
    print(f"Running {script_name} with k={k_value}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__), capture_output=False)
        if result.returncode != 0:
            print(f"ERROR: {script_name} with k={k_value} failed with return code {result.returncode}")
        else:
            print(f"SUCCESS: {script_name} with k={k_value} completed")
    except Exception as e:
        print(f"ERROR: Failed to run {script_name} with k={k_value}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Run all cross-validations for TabPFN models')
    parser.add_argument('--k', nargs='+', default=['all', '529', '100'],
                       help='Feature counts to test (default: all 529 100)')
    args = parser.parse_args()

    print("IApred-PFN Cross-Validation Runner")
    print("==================================")
    print(f"Will run CV for k values: {args.k}")
    print()

    # Scripts to run
    cv_scripts = ['10fold_cv', 'loco_cv']

    # Run all combinations
    for k in args.k:
        for script in cv_scripts:
            run_cv_script(script, k)

    print("\n" + "="*60)
    print("All cross-validation runs completed!")
    print("Check the results in tabpfn_training/results/*/ for output files and plots.")
    print("="*60)

if __name__ == "__main__":
    main()
