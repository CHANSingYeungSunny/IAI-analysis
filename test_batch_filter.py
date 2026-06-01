#!/usr/bin/env python
"""
Test the batch filtering logic
"""
import pandas as pd
import numpy as np

print("Testing batch filtering logic...")

# Simulate batch data with some batches having only 1 sample
batch_info = np.array(['batch_A', 'batch_A', 'batch_B', 'batch_C', 'batch_C', 'batch_C', 'batch_D'])
print(f"Original batch distribution: {pd.Series(batch_info).value_counts().to_dict()}")

# Filter to keep only batches with at least 2 observations
batch_counts = pd.Series(batch_info).value_counts()
print(f"Batch counts:\n{batch_counts}")

valid_batches = batch_counts[batch_counts >= 2].index
print(f"Valid batches (count >= 2): {list(valid_batches)}")

valid_mask = pd.Series(batch_info).isin(valid_batches).values
print(f"Valid mask: {valid_mask}")

filtered_batch = batch_info[valid_mask]
print(f"\nFiltered batches: {filtered_batch}")
print(f"Filtered batch distribution: {pd.Series(filtered_batch).value_counts().to_dict()}")

# Verify all batches have at least 2 observations
filtered_counts = pd.Series(filtered_batch).value_counts()
all_valid = (filtered_counts >= 2).all()
print(f"✅ All filtered batches have >= 2 observations: {all_valid}")
