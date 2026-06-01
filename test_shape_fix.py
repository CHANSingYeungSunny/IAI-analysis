#!/usr/bin/env python
"""
Test the corrected Combat API with proper shapes
"""
import numpy as np
import pandas as pd
import pycombat

print("Testing corrected Combat API with proper data shapes...")

# Simulate the actual data structure
n_donors = 10
n_covariates = 3

# Data: n_donors samples, 1 feature
Y = np.random.randn(n_donors, 1)
print(f"Y shape: {Y.shape}")

# Batch ids: n_donors samples
batch = np.array([0]*5 + [1]*5)
print(f"batch shape: {batch.shape}")

# Covariates: n_donors samples, 3 features
X = np.random.randn(n_donors, n_covariates)
print(f"X (covariates) shape: {X.shape}")

try:
    # Test with correct shapes (no transpose!)
    Y_corrected = pycombat.Combat().fit_transform(Y, batch, X=X)
    print(f"\n✅ Combat.fit_transform() successful!")
    print(f"✅ Output shape: {Y_corrected.shape}")
    print(f"✅ All samples matched: {Y.shape[0] == batch.shape[0] == X.shape[0]}")
except ValueError as e:
    print(f"\n❌ ValueError: {e}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
