#!/usr/bin/env python3
"""
Script to retrain all 4 models:
1. SVM
2. TabPFN (all features)
3. TabPFN (529 features)
4. TabPFN (100 features)
"""

import sys
import os
import subprocess

def run_script(script_path, description, k_arg=None):
    """Run a Python script"""
    print(f"\n{'='*80}")
    print(f"  {description}")
    print(f"{'='*80}")
    print(f"Script: {script_path}")
    if k_arg:
        print(f"Arguments: --k {k_arg}")
    print()

    if not os.path.exists(script_path):
        print(f"  ⚠️  WARNING: Script not found: {script_path}")
        return False

    cmd = [sys.executable, script_path]
    if k_arg:
        cmd.extend(['--k', k_arg])

    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(script_path),
            check=True
        )
        print(f"  ✅ Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ ERROR: Failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        return False

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(project_root)
    
    print("="*80)
    print("RETRAINING ALL MODELS")
    print("="*80)
    print("\nThis script will retrain all 4 models:")
    print("  1. SVM")
    print("  2. TabPFN (all features)")
    print("  3. TabPFN (529 features)")
    print("  4. TabPFN (100 features)")
    
    results = {}
    
    # 1. SVM
    results['SVM'] = run_script(
        os.path.join(project_root, 'svm_training/scripts/generate_and_save_models.py'),
        "Training SVM Model"
    )

    # 2. TabPFN (all features)
    results['TabPFN-all'] = run_script(
        os.path.join(project_root, 'tabpfn_training/scripts/train_model.py'),
        "Training TabPFN (all features)",
        k_arg='all'
    )

    # 3. TabPFN (529 features)
    results['TabPFN-529'] = run_script(
        os.path.join(project_root, 'tabpfn_training/scripts/train_model.py'),
        "Training TabPFN (529 features)",
        k_arg='529'
    )

    # 4. TabPFN (100 features)
    results['TabPFN-100'] = run_script(
        os.path.join(project_root, 'tabpfn_training/scripts/train_model.py'),
        "Training TabPFN (100 features)",
        k_arg='100'
    )
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    successful = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for model, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {model}: {status}")
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    print("="*80)
    
    if failed > 0:
        print("\n⚠️  Some models failed to train. Check the logs above.")
        return 1
    else:
        print("\n✅ All models trained successfully!")
        return 0

if __name__ == "__main__":
    exit(main())

