#!/usr/bin/env python
"""
Test compute_Zcells with the actual data and corrected Combat API
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pycombat

print("Testing compute_Zcells with actual data...")

try:
    # Load the actual data
    obs = pd.read_csv("input_data/pbmc_obs.csv", index_col=0, low_memory=False)
    print(f"✅ Loaded pbmc_obs.csv with shape {obs.shape}")
    print(f"   Columns: {list(obs.columns[:5])}...")
    
    # Calculate B/T ratio with division by zero protection
    donor_groups = obs.groupby("donor_id")["cell_type"].apply(
        lambda x: (x=="B cell").sum() / max((x=="T cell").sum(), 1)
    )
    print(f"✅ Calculated B/T ratio for {len(donor_groups)} donors")
    
    # Standardize
    zcells_raw = StandardScaler().fit_transform(donor_groups.values.reshape(-1,1))
    print(f"✅ Standardized data shape: {zcells_raw.shape}")
    
    # Prepare Combat inputs
    X = pd.DataFrame(zcells_raw, index=donor_groups.index, columns=["B_T_ratio"])
    batch = obs.groupby("donor_id")["batch"].first().values
    covariates = obs.groupby("donor_id")[["age","sex","disease"]].first().values
    
    print(f"\n   X shape: {X.shape}")
    print(f"   batch shape: {batch.shape}")
    print(f"   covariates shape: {covariates.shape}")
    
    # Test Combat with corrected API
    print(f"\n⏳ Running Combat.fit_transform()...")
    zcells_corrected = pycombat.Combat().fit_transform(X.values, batch, X=covariates)
    
    print(f"✅ Combat.fit_transform() successful!")
    print(f"   Output shape: {zcells_corrected.shape}")
    print(f"   No NaN values: {not np.isnan(zcells_corrected).any()}")
    print(f"   Sample means: min={np.nanmin(zcells_corrected):.4f}, max={np.nanmax(zcells_corrected):.4f}")
    
except FileNotFoundError as e:
    print(f"❌ Data file not found: {e}")
except ValueError as e:
    print(f"❌ ValueError (shape mismatch): {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
