#!/usr/bin/env python
"""
Quick test for pycombat Combat API fix
"""
import numpy as np
import pandas as pd
import pycombat

print("Testing pycombat.Combat API...")

# Create sample data
Y = np.random.randn(50, 10)  # 50 samples, 10 features
batch = np.array([0]*25 + [1]*25)  # 25 samples in batch 0, 25 in batch 1
X = np.random.randn(50, 2)  # 2 covariates to preserve

print(f"Y shape: {Y.shape}")
print(f"batch shape: {batch.shape}")
print(f"X shape: {X.shape}")

try:
    # Test with covariates
    Y_corrected = pycombat.Combat().fit_transform(Y, batch, X=X)
    print(f"✅ Combat.fit_transform() successful!")
    print(f"✅ Output shape: {Y_corrected.shape}")
    print(f"✅ No NaN values: {not np.isnan(Y_corrected).any()}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test without covariates
try:
    Y_corrected2 = pycombat.Combat().fit_transform(Y, batch)
    print(f"✅ Combat.fit_transform() without covariates works!")
except Exception as e:
    print(f"❌ Error without covariates: {e}")
