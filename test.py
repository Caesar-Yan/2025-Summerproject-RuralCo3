"""
Script: compare_transformed_files.py

Purpose:
    Compare undiscounted_price values between the original and updated transformed files
    to determine if the update actually changed anything
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
data_cleaning_dir = base_dir / "data_cleaning"

print(f"\n{'='*70}")
print(f"COMPARING ORIGINAL VS UPDATED TRANSFORMED FILES")
print(f"{'='*70}")

# =========================================================
# LOAD FILES
# =========================================================
print(f"\n--- LOADING FILES ---")

original_ats = data_cleaning_dir / 'datetime_parsed_ats_invoice_line_item_df_transformed.csv'
updated_ats = data_cleaning_dir / 'updated_datetime_parsed_ats_invoice_line_item_df_transformed.csv'
original_invoice = data_cleaning_dir / 'datetime_parsed_invoice_line_item_df_transformed.csv'
updated_invoice = data_cleaning_dir / 'updated_datetime_parsed_invoice_line_item_df_transformed.csv'

orig_ats_df = pd.read_csv(original_ats)
upd_ats_df = pd.read_csv(updated_ats)
orig_inv_df = pd.read_csv(original_invoice)
upd_inv_df = pd.read_csv(updated_invoice)

print(f"\nATS Files:")
print(f"  Original: {len(orig_ats_df):,} rows")
print(f"  Updated:  {len(upd_ats_df):,} rows")

print(f"\nInvoice Files:")
print(f"  Original: {len(orig_inv_df):,} rows")
print(f"  Updated:  {len(upd_inv_df):,} rows")

# =========================================================
# COMPARE ATS FILE
# =========================================================
print(f"\n{'='*70}")
print(f"COMPARING ATS FILE")
print(f"{'='*70}")

# Check if both have undiscounted_price column
if 'undiscounted_price' not in orig_ats_df.columns:
    print("\n⚠️  ERROR: Original ATS file missing 'undiscounted_price' column")
elif 'undiscounted_price' not in upd_ats_df.columns:
    print("\n⚠️  ERROR: Updated ATS file missing 'undiscounted_price' column")
else:
    # Count NaN values
    orig_nan = orig_ats_df['undiscounted_price'].isna().sum()
    upd_nan = upd_ats_df['undiscounted_price'].isna().sum()
    
    print(f"\nNaN values:")
    print(f"  Original: {orig_nan:,} / {len(orig_ats_df):,} ({orig_nan/len(orig_ats_df)*100:.1f}%)")
    print(f"  Updated:  {upd_nan:,} / {len(upd_ats_df):,} ({upd_nan/len(upd_ats_df)*100:.1f}%)")
    print(f"  Difference: {upd_nan - orig_nan:,}")
    
    # Compare values where both are non-NaN
    both_valid = orig_ats_df['undiscounted_price'].notna() & upd_ats_df['undiscounted_price'].notna()
    
    if both_valid.sum() > 0:
        orig_values = orig_ats_df.loc[both_valid, 'undiscounted_price']
        upd_values = upd_ats_df.loc[both_valid, 'undiscounted_price']
        
        # Find differences (allowing for small floating point errors)
        tolerance = 0.001
        different = np.abs(orig_values - upd_values) > tolerance
        
        print(f"\nRows with both values non-NaN: {both_valid.sum():,}")
        print(f"Rows with different values (tolerance={tolerance}): {different.sum():,}")
        
        if different.sum() > 0:
            print(f"\n✓ CHANGES DETECTED in {different.sum():,} rows ({different.sum()/both_valid.sum()*100:.2f}%)")
            
            # Calculate total price change
            price_change = (upd_values[different] - orig_values[different]).sum()
            print(f"\nTotal undiscounted_price change: ${price_change:,.2f}")
            
            # Show sample of changes
            print(f"\nSample of changes:")
            changed_df = orig_ats_df[both_valid][different].copy()
            changed_df['original_price'] = orig_values[different].values
            changed_df['updated_price'] = upd_values[different].values
            changed_df['difference'] = changed_df['updated_price'] - changed_df['original_price']
            
            sample = changed_df.head(10)[['invoice_id', 'description', 'original_price', 'updated_price', 'difference']]
            print(sample.to_string(index=False))
            
            # Statistics on differences
            diff_values = upd_values[different] - orig_values[different]
            print(f"\nDifference statistics:")
            print(f"  Min: ${diff_values.min():,.2f}")
            print(f"  Max: ${diff_values.max():,.2f}")
            print(f"  Mean: ${diff_values.mean():,.2f}")
            print(f"  Median: ${diff_values.median():,.2f}")
            print(f"  Std Dev: ${diff_values.std():,.2f}")
        else:
            print(f"\n✗ NO CHANGES - All values are identical!")
    
    # Check for values that changed from NaN to valid or vice versa
    nan_to_valid = orig_ats_df['undiscounted_price'].isna() & upd_ats_df['undiscounted_price'].notna()
    valid_to_nan = orig_ats_df['undiscounted_price'].notna() & upd_ats_df['undiscounted_price'].isna()
    
    print(f"\nNaN status changes:")
    print(f"  NaN → Valid: {nan_to_valid.sum():,}")
    print(f"  Valid → NaN: {valid_to_nan.sum():,}")
    
    if nan_to_valid.sum() > 0:
        print(f"\n  Sample of NaN → Valid changes:")
        sample = upd_ats_df[nan_to_valid].head(5)[['invoice_id', 'description', 'undiscounted_price', 'discounted_price']]
        print(sample.to_string(index=False))

# =========================================================
# COMPARE INVOICE FILE
# =========================================================
print(f"\n{'='*70}")
print(f"COMPARING INVOICE FILE")
print(f"{'='*70}")

# Check if both have undiscounted_price column
if 'undiscounted_price' not in orig_inv_df.columns:
    print("\n⚠️  ERROR: Original Invoice file missing 'undiscounted_price' column")
elif 'undiscounted_price' not in upd_inv_df.columns:
    print("\n⚠️  ERROR: Updated Invoice file missing 'undiscounted_price' column")
else:
    # Count NaN values
    orig_nan = orig_inv_df['undiscounted_price'].isna().sum()
    upd_nan = upd_inv_df['undiscounted_price'].isna().sum()
    
    print(f"\nNaN values:")
    print(f"  Original: {orig_nan:,} / {len(orig_inv_df):,} ({orig_nan/len(orig_inv_df)*100:.1f}%)")
    print(f"  Updated:  {upd_nan:,} / {len(upd_inv_df):,} ({upd_nan/len(upd_inv_df)*100:.1f}%)")
    print(f"  Difference: {upd_nan - orig_nan:,}")
    
    # Compare values where both are non-NaN
    both_valid = orig_inv_df['undiscounted_price'].notna() & upd_inv_df['undiscounted_price'].notna()
    
    if both_valid.sum() > 0:
        orig_values = orig_inv_df.loc[both_valid, 'undiscounted_price']
        upd_values = upd_inv_df.loc[both_valid, 'undiscounted_price']
        
        # Find differences (allowing for small floating point errors)
        tolerance = 0.001
        different = np.abs(orig_values - upd_values) > tolerance
        
        print(f"\nRows with both values non-NaN: {both_valid.sum():,}")
        print(f"Rows with different values (tolerance={tolerance}): {different.sum():,}")
        
        if different.sum() > 0:
            print(f"\n✓ CHANGES DETECTED in {different.sum():,} rows ({different.sum()/both_valid.sum()*100:.2f}%)")
            
            # Calculate total price change
            price_change = (upd_values[different] - orig_values[different]).sum()
            print(f"\nTotal undiscounted_price change: ${price_change:,.2f}")
            
            # Show sample of changes
            print(f"\nSample of changes:")
            changed_df = orig_inv_df[both_valid][different].copy()
            changed_df['original_price'] = orig_values[different].values
            changed_df['updated_price'] = upd_values[different].values
            changed_df['difference'] = changed_df['updated_price'] - changed_df['original_price']
            
            sample = changed_df.head(10)[['invoice_id', 'description', 'original_price', 'updated_price', 'difference']]
            print(sample.to_string(index=False))
            
            # Statistics on differences
            diff_values = upd_values[different] - orig_values[different]
            print(f"\nDifference statistics:")
            print(f"  Min: ${diff_values.min():,.2f}")
            print(f"  Max: ${diff_values.max():,.2f}")
            print(f"  Mean: ${diff_values.mean():,.2f}")
            print(f"  Median: ${diff_values.median():,.2f}")
            print(f"  Std Dev: ${diff_values.std():,.2f}")
        else:
            print(f"\n✗ NO CHANGES - All values are identical!")
    
    # Check for values that changed from NaN to valid or vice versa
    nan_to_valid = orig_inv_df['undiscounted_price'].isna() & upd_inv_df['undiscounted_price'].notna()
    valid_to_nan = orig_inv_df['undiscounted_price'].notna() & upd_inv_df['undiscounted_price'].isna()
    
    print(f"\nNaN status changes:")
    print(f"  NaN → Valid: {nan_to_valid.sum():,}")
    print(f"  Valid → NaN: {valid_to_nan.sum():,}")
    
    if nan_to_valid.sum() > 0:
        print(f"\n  Sample of NaN → Valid changes:")
        sample = upd_inv_df[nan_to_valid].head(5)[['invoice_id', 'description', 'undiscounted_price', 'discounted_price']]
        print(sample.to_string(index=False))

# =========================================================
# OVERALL SUMMARY
# =========================================================
print(f"\n{'='*70}")
print(f"OVERALL SUMMARY")
print(f"{'='*70}")

print(f"\nATS File:")
if 'undiscounted_price' in orig_ats_df.columns and 'undiscounted_price' in upd_ats_df.columns:
    ats_both_valid = orig_ats_df['undiscounted_price'].notna() & upd_ats_df['undiscounted_price'].notna()
    if ats_both_valid.sum() > 0:
        ats_different = np.abs(orig_ats_df.loc[ats_both_valid, 'undiscounted_price'] - 
                               upd_ats_df.loc[ats_both_valid, 'undiscounted_price']) > 0.001
        print(f"  • Total rows: {len(orig_ats_df):,}")
        print(f"  • Rows changed: {ats_different.sum():,}")
        print(f"  • Percentage changed: {ats_different.sum()/ats_both_valid.sum()*100:.2f}%")

print(f"\nInvoice File:")
if 'undiscounted_price' in orig_inv_df.columns and 'undiscounted_price' in upd_inv_df.columns:
    inv_both_valid = orig_inv_df['undiscounted_price'].notna() & upd_inv_df['undiscounted_price'].notna()
    if inv_both_valid.sum() > 0:
        inv_different = np.abs(orig_inv_df.loc[inv_both_valid, 'undiscounted_price'] - 
                               upd_inv_df.loc[inv_both_valid, 'undiscounted_price']) > 0.001
        print(f"  • Total rows: {len(orig_inv_df):,}")
        print(f"  • Rows changed: {inv_different.sum():,}")
        print(f"  • Percentage changed: {inv_different.sum()/inv_both_valid.sum()*100:.2f}%")

# Check if avg_discount and discount_source columns exist
print(f"\n{'='*70}")
print(f"ADDITIONAL COLUMN CHECK")
print(f"{'='*70}")

print(f"\nATS File - New columns added:")
print(f"  avg_discount: {'avg_discount' in upd_ats_df.columns}")
print(f"  discount_source: {'discount_source' in upd_ats_df.columns}")

print(f"\nInvoice File - New columns added:")
print(f"  avg_discount: {'avg_discount' in upd_inv_df.columns}")
print(f"  discount_source: {'discount_source' in upd_inv_df.columns}")

if 'avg_discount' in upd_inv_df.columns:
    print(f"\nInvoice file avg_discount summary:")
    print(f"  Rows with avg_discount > 0: {(upd_inv_df['avg_discount'] > 0).sum():,}")
    print(f"  Mean avg_discount (where > 0): {upd_inv_df[upd_inv_df['avg_discount'] > 0]['avg_discount'].mean():.4f}%")

if 'discount_source' in upd_inv_df.columns:
    print(f"\nInvoice file discount_source distribution:")
    print(upd_inv_df['discount_source'].value_counts().head(10))

print(f"\n{'='*70}\n")