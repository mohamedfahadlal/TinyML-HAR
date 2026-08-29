# =================================================================
# HAR DATASETS LEAKAGE AUDIT UTILITY
# FILE: d:\Github\TinyML-HAR\notebooks\WISDM Dataset Models\audit_leakage.py
# =================================================================

import numpy as np
import pandas as pd

def perform_leakage_audit():
    print("="*65)
    print("        RUNNING STRUCTURAL HAR DATA LEAKAGE AUDIT")
    print("="*65)
    
    # 1. Load the original preprocessed base structures if tracking arrays are intact
    # Note: If you only saved X_train/X_test without tracking columns, we can infer 
    # overlap by evaluating exact statistical matches in signal characteristics.
    try:
        X_train = np.load("X_train_processed.npy")
        X_test = np.load("X_test_processed.npy")
    except FileNotFoundError:
        print("Error: Processed array files (.npy) missing from current workspace.")
        return

    print(f"Total Training Windows Loaded: {X_train.shape[0]}")
    print(f"Total Testing Windows Loaded : {X_test.shape[0]}")
    
    # 2. Check for Exact Spatial Leakage (Duplicate Windows across splits)
    print("\n[Audit 1/2] Checking for exact window row duplication...")
    
    # Flatten windows to 2D for fast hash matching
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Convert to pandas DataFrames to utilize internal row matching
    df_train = pd.DataFrame(X_train_flat)
    df_test = pd.DataFrame(X_test_flat)
    
    # Identify intersecting rows
    merged = pd.merge(df_train, df_test, how='inner')
    duplicate_count = len(merged)
    
    if duplicate_count > 0:
        print(f"❌ LEAKAGE DETECTED: {duplicate_count} exact time-series snapshots exist in BOTH splits!")
        print("   Reason: Shuffling was likely applied directly to raw rows prior to continuous window segmentation.")
    else:
        print("   Clean: Zero duplicate window snapshots found between train and test boundaries.")

    # 3. Structural Evaluation Guidance
    print("\n[Audit 2/2] Subject Separation Review...")
    print("-> Recommendation: Check your raw database generation script.")
    print("   If your training code used 'train_test_split(X, y)' without keeping track")
    print("   of the unique Subject IDs, your model is benefiting from subject personalization.")
    print("\n" + "="*25 + " END OF AUDIT " + "="*25)

if __name__ == "__main__":
    perform_leakage_audit()