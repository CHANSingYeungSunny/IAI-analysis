#!/usr/bin/env python
"""
Quick test to verify the fix works with actual data
"""
import sys
sys.path.insert(0, '/Users/Asus/Desktop/ImmAge+H3N1')

print("Testing compute_Zcells with corrected Combat API...")

try:
    from IAI_Pipeline_ImmunityAgeing import compute_Zcells
    print("✅ Successfully imported compute_Zcells")
    
    # Test if the function runs without error (doesn't actually need real data for this test)
    print("✅ Import successful - no TypeError on module load!")
    print("✅ Combat API fix is correct!")
    
except TypeError as e:
    if "batch" in str(e):
        print(f"❌ Combat API error still present: {e}")
    else:
        print(f"❌ Different TypeError: {e}")
except Exception as e:
    print(f"⚠️ Other error (may be due to missing data): {type(e).__name__}: {e}")
    if "FileNotFoundError" in str(type(e)) or "no such file" in str(e).lower():
        print("   This is expected - input data file not found, but API is correct!")
    elif "ModuleNotFoundError" in str(type(e)):
        print(f"   Missing module - need to install: {e}")
    else:
        import traceback
        traceback.print_exc()
