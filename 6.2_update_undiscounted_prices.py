"""
Script: update_undiscounted_prices_in_transformed_files.py

Purpose:
    Update the undiscounted_price column in the transformed line item files
    using the corrected values from 14.2_updated_invoice_line_items_with_discounts.csv
    Matching is done using the unique combination of invoice_id and description

Inputs:
    - 14.2_updated_invoice_line_items_with_discounts.csv (source of correct prices)
    - datetime_parsed_ats_invoice_line_item_df_transformed.csv
    - datetime_parsed_invoice_line_item_df_transformed.csv

Outputs:
    - datetime_parsed_ats_invoice_line_item_df_transformed.csv (updated)
    - datetime_parsed_invoice_line_item_df_transformed.csv (updated)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
merchant_folder_dir = base_dir / "merchant"
data_cleaning_dir = base_dir / "data_cleaning"

print(f"\n{'='*70}")
print(f"UPDATING UNDISCOUNTED PRICES IN TRANSFORMED FILES")
print(f"{'='*70}")

# =========================================================
# LOAD SOURCE FILE WITH CORRECT UNDISCOUNTED PRICES
# =========================================================
print(f"\n--- LOADING SOURCE FILE ---")

source_file = merchant_folder_dir / '14.2_updated_invoice_line_items_with_discounts.csv'
source_df = pd.read_csv(source_file)

print(f"Loaded source file: {len(source_df):,} rows")

# Check if join keys exist
if 'invoice_id' not in source_df.columns:
    print("\n⚠️  ERROR: 'invoice_id' not found in source file")
    print(f"Available columns: {list(source_df.columns)[:15]}...")
    exit()
    
if 'description' not in source_df.columns:
    print("\n⚠️  ERROR: 'description' not found in source file")
    print(f"Available columns: {list(source_df.columns)[:15]}...")
    exit()

# Select only the columns we need from source for the merge
merge_cols = ['invoice_id', 'description', 'undiscounted_price', 'avg_discount', 'discount_source']
source_merge = source_df[merge_cols].copy()

print(f"\nSource data prepared with columns: {merge_cols}")
print(f"Unique (invoice_id, description) combinations: {source_merge[['invoice_id', 'description']].drop_duplicates().shape[0]:,}")

# Check for duplicates in the join key
duplicate_keys = source_merge.groupby(['invoice_id', 'description']).size()
if (duplicate_keys > 1).any():
    num_duplicates = (duplicate_keys > 1).sum()
    print(f"\n⚠️  Warning: {num_duplicates:,} duplicate (invoice_id, description) combinations found in source")
    print(f"   Max duplicates for one key: {duplicate_keys.max()}")
    print("\n   Using first occurrence for each duplicate key")
    # Keep only first occurrence of each key
    source_merge = source_merge.drop_duplicates(subset=['invoice_id', 'description'], keep='first')
    print(f"   After deduplication: {len(source_merge):,} unique keys")

# =========================================================
# LOAD TARGET FILES TO UPDATE
# =========================================================
print(f"\n--- LOADING TARGET FILES ---")

ats_file = data_cleaning_dir / 'datetime_parsed_ats_invoice_line_item_df_transformed.csv'
invoice_file = data_cleaning_dir / 'datetime_parsed_invoice_line_item_df_transformed.csv'

ats_df = pd.read_csv(ats_file)
invoice_df = pd.read_csv(invoice_file)

print(f"\nATS file: {len(ats_df):,} rows")
if 'invoice_id' in ats_df.columns and 'description' in ats_df.columns:
    unique_keys = ats_df[['invoice_id', 'description']].drop_duplicates().shape[0]
    print(f"  Unique (invoice_id, description) combinations: {unique_keys:,}")
    # Check for duplicates
    dup_check = ats_df.groupby(['invoice_id', 'description']).size()
    if (dup_check > 1).any():
        print(f"  ⚠️  Warning: {(dup_check > 1).sum():,} duplicate keys in ATS file")
else:
    print(f"  ⚠️  ERROR: Missing join keys!")
    print(f"  Available columns: {list(ats_df.columns)[:15]}...")
    
print(f"\nInvoice file: {len(invoice_df):,} rows")
if 'invoice_id' in invoice_df.columns and 'description' in invoice_df.columns:
    unique_keys = invoice_df[['invoice_id', 'description']].drop_duplicates().shape[0]
    print(f"  Unique (invoice_id, description) combinations: {unique_keys:,}")
    # Check for duplicates
    dup_check = invoice_df.groupby(['invoice_id', 'description']).size()
    if (dup_check > 1).any():
        print(f"  ⚠️  Warning: {(dup_check > 1).sum():,} duplicate keys in Invoice file")
else:
    print(f"  ⚠️  ERROR: Missing join keys!")
    print(f"  Available columns: {list(invoice_df.columns)[:15]}...")

# =========================================================
# UPDATE ATS FILE
# =========================================================
print(f"\n{'='*70}")
print(f"UPDATING ATS FILE")
print(f"{'='*70}")

# Store original undiscounted_price for comparison
ats_df['undiscounted_price_original'] = ats_df['undiscounted_price'].copy()

# Merge with source data
ats_updated = ats_df.merge(
    source_merge,
    on=['invoice_id', 'description'],
    how='left',
    suffixes=('_old', '_new')
)

print(f"\nMerge result: {len(ats_updated):,} rows (original: {len(ats_df):,})")

# Check if merge created new columns
if 'undiscounted_price_new' in ats_updated.columns:
    # Update undiscounted_price with new values, keep old if no match
    matched_mask = ats_updated['undiscounted_price_new'].notna()
    print(f"  Matched rows: {matched_mask.sum():,} / {len(ats_updated):,} ({matched_mask.sum()/len(ats_updated)*100:.1f}%)")
    
    ats_updated['undiscounted_price'] = ats_updated['undiscounted_price_new'].fillna(ats_updated['undiscounted_price_old'])
    
    # Update avg_discount and discount_source if they exist
    if 'avg_discount_new' in ats_updated.columns:
        if 'avg_discount' in ats_df.columns:
            ats_updated['avg_discount'] = ats_updated['avg_discount_new'].fillna(ats_updated.get('avg_discount_old', 0))
        else:
            ats_updated['avg_discount'] = ats_updated['avg_discount_new']
    
    if 'discount_source_new' in ats_updated.columns:
        if 'discount_source' in ats_df.columns:
            ats_updated['discount_source'] = ats_updated['discount_source_new'].fillna(ats_updated.get('discount_source_old', 'none'))
        else:
            ats_updated['discount_source'] = ats_updated['discount_source_new']
    
    # Drop temporary columns
    cols_to_drop = [col for col in ats_updated.columns if col.endswith('_old') or col.endswith('_new')]
    ats_updated = ats_updated.drop(columns=cols_to_drop)
    
    # Calculate change
    price_change = (ats_updated['undiscounted_price'] - ats_updated['undiscounted_price_original']).sum()
    print(f"  Total undiscounted_price change: ${price_change:,.2f}")
    
    # Show some examples of changes
    changed_rows = ats_updated[ats_updated['undiscounted_price'] != ats_updated['undiscounted_price_original']]
    if len(changed_rows) > 0:
        print(f"  Number of rows with price changes: {len(changed_rows):,}")
        print(f"\n  Sample of changes:")
        sample = changed_rows.head(3)[['invoice_id', 'description', 'undiscounted_price_original', 'undiscounted_price', 'discounted_price']]
        sample.columns = ['invoice_id', 'description', 'old_undiscounted', 'new_undiscounted', 'discounted']
        print(sample.to_string(index=False))
    
else:
    print("\n  ⚠️  No new undiscounted_price values found in merge")
    ats_updated = ats_df.copy()

# Drop the original comparison column
if 'undiscounted_price_original' in ats_updated.columns:
    ats_updated = ats_updated.drop(columns=['undiscounted_price_original'])

# =========================================================
# UPDATE INVOICE FILE
# =========================================================
print(f"\n{'='*70}")
print(f"UPDATING INVOICE FILE")
print(f"{'='*70}")

# Store original undiscounted_price for comparison
invoice_df['undiscounted_price_original'] = invoice_df['undiscounted_price'].copy()

# Merge with source data
invoice_updated = invoice_df.merge(
    source_merge,
    on=['invoice_id', 'description'],
    how='left',
    suffixes=('_old', '_new')
)

print(f"\nMerge result: {len(invoice_updated):,} rows (original: {len(invoice_df):,})")

# Check if merge created new columns
if 'undiscounted_price_new' in invoice_updated.columns:
    # Update undiscounted_price with new values, keep old if no match
    matched_mask = invoice_updated['undiscounted_price_new'].notna()
    print(f"  Matched rows: {matched_mask.sum():,} / {len(invoice_updated):,} ({matched_mask.sum()/len(invoice_updated)*100:.1f}%)")
    
    invoice_updated['undiscounted_price'] = invoice_updated['undiscounted_price_new'].fillna(invoice_updated['undiscounted_price_old'])
    
    # Update avg_discount and discount_source if they exist
    if 'avg_discount_new' in invoice_updated.columns:
        if 'avg_discount' in invoice_df.columns:
            invoice_updated['avg_discount'] = invoice_updated['avg_discount_new'].fillna(invoice_updated.get('avg_discount_old', 0))
        else:
            invoice_updated['avg_discount'] = invoice_updated['avg_discount_new']
    
    if 'discount_source_new' in invoice_updated.columns:
        if 'discount_source' in invoice_df.columns:
            invoice_updated['discount_source'] = invoice_updated['discount_source_new'].fillna(invoice_updated.get('discount_source_old', 'none'))
        else:
            invoice_updated['discount_source'] = invoice_updated['discount_source_new']
    
    # Drop temporary columns
    cols_to_drop = [col for col in invoice_updated.columns if col.endswith('_old') or col.endswith('_new')]
    invoice_updated = invoice_updated.drop(columns=cols_to_drop)
    
    # Calculate change
    price_change = (invoice_updated['undiscounted_price'] - invoice_updated['undiscounted_price_original']).sum()
    print(f"  Total undiscounted_price change: ${price_change:,.2f}")
    
    # Show some examples of changes
    changed_rows = invoice_updated[invoice_updated['undiscounted_price'] != invoice_updated['undiscounted_price_original']]
    if len(changed_rows) > 0:
        print(f"  Number of rows with price changes: {len(changed_rows):,}")
        print(f"\n  Sample of changes:")
        sample = changed_rows.head(3)[['invoice_id', 'description', 'undiscounted_price_original', 'undiscounted_price', 'discounted_price']]
        sample.columns = ['invoice_id', 'description', 'old_undiscounted', 'new_undiscounted', 'discounted']
        print(sample.to_string(index=False))
    
else:
    print("\n  ⚠️  No new undiscounted_price values found in merge")
    invoice_updated = invoice_df.copy()

# Drop the original comparison column
if 'undiscounted_price_original' in invoice_updated.columns:
    invoice_updated = invoice_updated.drop(columns=['undiscounted_price_original'])

# =========================================================
# VALIDATION
# =========================================================
print(f"\n{'='*70}")
print(f"VALIDATION")
print(f"{'='*70}")

print(f"\nATS file validation:")
print(f"  Rows: {len(ats_updated):,}")
print(f"  Total discounted price: ${ats_updated['discounted_price'].sum():,.2f}")
print(f"  Total undiscounted price: ${ats_updated['undiscounted_price'].sum():,.2f}")
print(f"  Implied total discount: ${(ats_updated['undiscounted_price'] - ats_updated['discounted_price']).sum():,.2f}")
if 'avg_discount' in ats_updated.columns:
    print(f"  Rows with avg_discount > 0: {(ats_updated['avg_discount'] > 0).sum():,}")
    print(f"  Mean avg_discount: {ats_updated[ats_updated['avg_discount'] > 0]['avg_discount'].mean():.4f}%")

print(f"\nInvoice file validation:")
print(f"  Rows: {len(invoice_updated):,}")
print(f"  Total discounted price: ${invoice_updated['discounted_price'].sum():,.2f}")
print(f"  Total undiscounted price: ${invoice_updated['undiscounted_price'].sum():,.2f}")
print(f"  Implied total discount: ${(invoice_updated['undiscounted_price'] - invoice_updated['discounted_price']).sum():,.2f}")
if 'avg_discount' in invoice_updated.columns:
    print(f"  Rows with avg_discount > 0: {(invoice_updated['avg_discount'] > 0).sum():,}")
    print(f"  Mean avg_discount: {invoice_updated[invoice_updated['avg_discount'] > 0]['avg_discount'].mean():.4f}%")

# =========================================================
# SAVE UPDATED FILES
# =========================================================
print(f"\n{'='*70}")
print(f"SAVING UPDATED FILES")
print(f"{'='*70}")

# Create backup of original files
backup_suffix = '_backup_before_discount_update'
ats_backup = ats_file.parent / (ats_file.stem + backup_suffix + ats_file.suffix)
invoice_backup = invoice_file.parent / (invoice_file.stem + backup_suffix + invoice_file.suffix)

print(f"\nCreating backups:")
ats_df.to_csv(ats_backup, index=False)
print(f"  ✓ {ats_backup.name}")
invoice_df.to_csv(invoice_backup, index=False)
print(f"  ✓ {invoice_backup.name}")

# Save updated files
print(f"\nSaving updated files:")
ats_updated.to_csv(ats_file, index=False)
print(f"  ✓ {ats_file.name} ({len(ats_updated):,} rows)")
invoice_updated.to_csv(invoice_file, index=False)
print(f"  ✓ {invoice_file.name} ({len(invoice_updated):,} rows)")

# =========================================================
# SUMMARY
# =========================================================
print(f"\n{'='*70}")
print(f"UPDATE COMPLETE!")
print(f"{'='*70}")

print(f"\n📊 SUMMARY:")
print(f"  • ATS file: {len(ats_updated):,} rows updated")
print(f"  • Invoice file: {len(invoice_updated):,} rows updated")
print(f"  • Join key: invoice_id + description")
print(f"  • Backups created with suffix: '{backup_suffix}'")

print(f"\n💰 TOTAL PRICE SUMMARY:")
total_discounted = ats_updated['discounted_price'].sum() + invoice_updated['discounted_price'].sum()
total_undiscounted = ats_updated['undiscounted_price'].sum() + invoice_updated['undiscounted_price'].sum()
total_discount_amount = total_undiscounted - total_discounted

print(f"  • Combined discounted price: ${total_discounted:,.2f}")
print(f"  • Combined undiscounted price: ${total_undiscounted:,.2f}")
print(f"  • Combined discount amount: ${total_discount_amount:,.2f}")
if total_undiscounted > 0:
    print(f"  • Effective discount rate: {(total_discount_amount / total_undiscounted * 100):.2f}%")

print(f"\n💡 NEXT STEPS:")
print(f"  1. Run script 08_group_by_invoice_transformed.py to regenerate grouped data")
print(f"  2. Verify the discount calculations are correct")
print(f"  3. If issues arise, restore from backup files:")
print(f"     - {ats_backup.name}")
print(f"     - {invoice_backup.name}")

print(f"\n{'='*70}\n")