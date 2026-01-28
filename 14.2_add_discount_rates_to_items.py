"""
Script: 14.2_add_discount_rates_to_items.py

Purpose:
    Map average discount rates from match_layer analysis to invoice line items
    and calculate undiscounted prices.
    For L1, L2, L3_blackwoods, and L3_methven_motors: match to actual merchant discounts
    For other layers: use average discount by match_layer

Inputs:
    - 13.7_matching_progress.csv (invoice line items with match_layers)
    - 14.1_average_discount_by_match_layer.csv (average discounts by layer)
    - 13.99.1_all_merchants_with_cleaned_discount.csv (merchant discount data)

Outputs:
    - 14.2_updated_invoice_line_items_with_discounts.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os

# Import your custom functions
from matching_functions import (
    calculate_percentage_true,
    calculate_percentage_not_null,
    add_category,
    save_and_summarize2,
    analyze_dataframes,
    analyze_dataframe,
    get_non_null_percentage,
    check_diff,
    merge_updates_to_main_df,
    filter_non_empty_column
)

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
merchant_folder_dir = base_dir / "merchant"
data_cleaning_dir = base_dir / "data_cleaning"
output_dir = merchant_folder_dir

# =========================================================
# LOAD AVERAGE DISCOUNT BY MATCH LAYER
# =========================================================
print(f"\n{'='*70}")
print(f"LOADING AVERAGE DISCOUNT RATES BY MATCH LAYER")
print(f"{'='*70}")

discount_rates_file = merchant_folder_dir / '14.1_average_discount_by_match_layer.csv'
discount_rates_df = pd.read_csv(discount_rates_file)

print(f"Loaded {len(discount_rates_df)} match_layer discount rates")
print("\nDiscount rates by match_layer:")
print(discount_rates_df[['match_layer', 'avg_discount', 'count']].to_string(index=False))
print(f"{'='*70}\n")

# =========================================================
# LOAD MERCHANT DISCOUNT DATA
# =========================================================
print(f"\n{'='*70}")
print(f"LOADING MERCHANT DISCOUNT DATA")
print(f"{'='*70}")

merchant_discount_file = merchant_folder_dir / '13.99.1_all_merchants_with_cleaned_discount.csv'
merchant_discount_df = pd.read_csv(merchant_discount_file)

print(f"Loaded {len(merchant_discount_df):,} merchant records")
print(f"Unique ATS Numbers: {merchant_discount_df['ATS Number'].nunique():,}")

# Create a lookup dictionary: ATS Number -> cleaned_discount
# Use the first cleaned_discount value for each ATS Number
merchant_discount_lookup = merchant_discount_df.groupby('ATS Number')['cleaned_discount'].first().to_dict()

print(f"Created discount lookup for {len(merchant_discount_lookup):,} unique merchants")
print(f"Merchants with non-null discounts: {sum(1 for v in merchant_discount_lookup.values() if pd.notna(v)):,}")
print(f"{'='*70}\n")

# =========================================================
# LOAD FULL PARENT FILE
# =========================================================
print(f"\n{'='*70}")
print(f"LOADING INVOICE LINE ITEMS")
print(f"{'='*70}")

full_df = pd.read_csv(merchant_folder_dir / '13.7_matching_progress.csv')

print(f"Loaded {len(full_df):,} invoice line items")
print(f"\nCurrent match_layer distribution:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}\n")

# =========================================================
# CREATE DISCOUNT MAPPING DICTIONARY
# =========================================================
print(f"\n{'='*70}")
print(f"CREATING DISCOUNT MAPPING")
print(f"{'='*70}")

# Create mapping dictionary from discount_rates_df
discount_mapping = discount_rates_df.set_index('match_layer')['avg_discount'].to_dict()

# Handle 'exclude' -> map to 'unmatched' discount
if 'exclude' in discount_mapping:
    exclude_discount = discount_mapping['exclude']
    print(f"\nMapping 'exclude' discount ({exclude_discount:.4f}) to 'unmatched' match_layer")
else:
    exclude_discount = 0
    print(f"\nNo 'exclude' layer found in discount rates, using 0 for unmatched")

print(f"\nDiscount mapping dictionary:")
for layer, discount in sorted(discount_mapping.items()):
    print(f"  {layer}: {discount:.4f}")

# =========================================================
# MAP DISCOUNTS TO INVOICE LINE ITEMS
# =========================================================
print(f"\n{'='*70}")
print(f"MAPPING DISCOUNT RATES TO INVOICE LINE ITEMS")
print(f"{'='*70}")

# Initialize avg_discount column with 0
full_df['avg_discount'] = 0.0
full_df['discount_source'] = 'none'

# Define layers that should use merchant-specific discounts
merchant_specific_layers = ['L1', 'L2', 'L3_blackwoods', 'L3_methven_motors']

# =========================================================
# STEP 1: Apply merchant-specific discounts for L1, L2, L3_blackwoods, L3_methven_motors
# =========================================================
print(f"\n--- APPLYING MERCHANT-SPECIFIC DISCOUNTS ---")

for layer in merchant_specific_layers:
    layer_mask = full_df['match_layer'] == layer
    
    if not layer_mask.any():
        print(f"  {layer}: No rows found")
        continue
    
    # Determine which ATS column to use
    if layer == 'L1':
        ats_column = 'matched_ats_number'
    elif layer == 'L2':
        ats_column = 'matched_ats_number_L2'
    elif layer == 'L3_blackwoods':
        ats_column = 'matched_ats_number'
    elif layer == 'L3_methven_motors':
        ats_column = 'matched_ats_number'
    
    # Map the discount from merchant lookup
    matched_count = 0
    unmatched_count = 0
    
    for idx in full_df[layer_mask].index:
        ats_number = full_df.loc[idx, ats_column]
        
        if pd.notna(ats_number) and ats_number in merchant_discount_lookup:
            merchant_discount = merchant_discount_lookup[ats_number]
            
            if pd.notna(merchant_discount):
                full_df.loc[idx, 'avg_discount'] = merchant_discount
                full_df.loc[idx, 'discount_source'] = f'{layer}_merchant_specific'
                matched_count += 1
            else:
                unmatched_count += 1
        else:
            unmatched_count += 1
    
    print(f"  {layer}: {matched_count:,} matched to merchant discounts, {unmatched_count:,} not matched")

# =========================================================
# STEP 2: Apply average discounts for all other layers
# =========================================================
print(f"\n--- APPLYING AVERAGE DISCOUNTS FOR OTHER LAYERS ---")

for match_layer, avg_discount in discount_mapping.items():
    if match_layer in merchant_specific_layers:
        # Skip layers we already handled
        continue
    
    if match_layer == 'exclude':
        # Map 'exclude' discount to 'unmatched' rows
        mask = (full_df['match_layer'] == 'unmatched') & (full_df['avg_discount'] == 0)
        full_df.loc[mask, 'avg_discount'] = avg_discount
        full_df.loc[mask, 'discount_source'] = 'exclude_average'
        print(f"  Mapped 'exclude' discount ({avg_discount:.4f}) to {mask.sum():,} 'unmatched' rows")
    else:
        # Map other match_layers normally
        mask = (full_df['match_layer'] == match_layer) & (full_df['avg_discount'] == 0)
        full_df.loc[mask, 'avg_discount'] = avg_discount
        full_df.loc[mask, 'discount_source'] = f'{match_layer}_average'
        print(f"  Mapped {match_layer}: {avg_discount:.4f} to {mask.sum():,} rows")

# Check for unmapped rows
unmapped_mask = (full_df['avg_discount'] == 0) & (full_df['match_layer'] != 'unmatched')
if unmapped_mask.sum() > 0:
    print(f"\n⚠️  Warning: {unmapped_mask.sum():,} rows have match_layer but no discount rate")
    print("Match layers without discount rates:")
    print(full_df[unmapped_mask]['match_layer'].value_counts())

# =========================================================
# DISCOUNT SOURCE SUMMARY
# =========================================================
print(f"\n{'='*70}")
print(f"DISCOUNT SOURCE SUMMARY")
print(f"{'='*70}")
print(full_df['discount_source'].value_counts().to_string())

# =========================================================
# CALCULATE UNDISCOUNTED PRICES
# =========================================================
print(f"\n{'='*70}")
print(f"CALCULATING UNDISCOUNTED PRICES")
print(f"{'='*70}")

# Initialize undiscounted_price column
full_df['undiscounted_price'] = 0.0

# Standard calculation: undiscounted_price = discounted_price / (1 - avg_discount/100)
# Avoid division by zero: where avg_discount is 0, undiscounted_price = discounted_price
standard_mask = ~full_df['match_layer'].isin(['L4_petrol_no_merchant', 'L4_diesel_no_merchant'])

full_df.loc[standard_mask & (full_df['avg_discount'] != 0), 'undiscounted_price'] = (
    full_df.loc[standard_mask & (full_df['avg_discount'] != 0), 'discounted_price'] / 
    (1 - full_df.loc[standard_mask & (full_df['avg_discount'] != 0), 'avg_discount'] / 100)
)

full_df.loc[standard_mask & (full_df['avg_discount'] == 0), 'undiscounted_price'] = (
    full_df.loc[standard_mask & (full_df['avg_discount'] == 0), 'discounted_price']
)

print(f"  Standard calculation applied to {standard_mask.sum():,} rows")

# Special calculation for petrol/diesel: undiscounted_price = quantity * 0.12 + discounted_price
petrol_diesel_mask = full_df['match_layer'].isin(['L4_petrol_no_merchant', 'L4_diesel_no_merchant'])

full_df.loc[petrol_diesel_mask, 'undiscounted_price'] = (
    full_df.loc[petrol_diesel_mask, 'quantity'] * 0.12 + 
    full_df.loc[petrol_diesel_mask, 'discounted_price']
)

print(f"  Petrol/Diesel calculation applied to {petrol_diesel_mask.sum():,} rows")

# =========================================================
# VALIDATION
# =========================================================
print(f"\n{'='*70}")
print(f"VALIDATION")
print(f"{'='*70}")

# Check for any issues
print(f"\nData validation:")
print(f"  Rows with avg_discount: {(full_df['avg_discount'] > 0).sum():,}")
print(f"  Rows with undiscounted_price: {(full_df['undiscounted_price'] > 0).sum():,}")
print(f"  Rows with zero undiscounted_price: {(full_df['undiscounted_price'] == 0).sum():,}")

print(f"\nSample calculations by match_layer:")
sample_summary = full_df.groupby('match_layer').agg({
    'avg_discount': 'first',
    'discounted_price': 'mean',
    'undiscounted_price': 'mean',
    'match_layer': 'count'
}).round(4)
sample_summary.columns = ['avg_discount', 'mean_discounted', 'mean_undiscounted', 'count']
print(sample_summary.to_string())

# Detailed breakdown by discount source
print(f"\nDetailed breakdown by discount_source:")
source_summary = full_df.groupby('discount_source').agg({
    'avg_discount': ['mean', 'min', 'max'],
    'discounted_price': 'sum',
    'undiscounted_price': 'sum',
    'discount_source': 'count'
}).round(4)
source_summary.columns = ['avg_discount_mean', 'avg_discount_min', 'avg_discount_max', 
                          'total_discounted', 'total_undiscounted', 'count']
print(source_summary.to_string())

# =========================================================
# SAVE OUTPUT
# =========================================================
print(f"\n{'='*70}")
print(f"SAVING OUTPUT")
print(f"{'='*70}")

output_file = merchant_folder_dir / '14.2_updated_invoice_line_items_with_discounts.csv'
full_df.to_csv(output_file, index=False)
print(f"Saved: {output_file.name} ({len(full_df):,} rows)")

# =========================================================
# SUMMARY
# =========================================================
print(f"\n{'='*70}")
print(f"PROCESS COMPLETE!")
print(f"{'='*70}")

print(f"\n📊 SUMMARY:")
print(f"  • Total invoice line items: {len(full_df):,}")
print(f"  • Items with discount rates: {(full_df['avg_discount'] > 0).sum():,}")
print(f"  • Items with zero discount: {(full_df['avg_discount'] == 0).sum():,}")
print(f"  • Petrol/Diesel items (special calc): {petrol_diesel_mask.sum():,}")

# Merchant-specific vs average breakdown
merchant_specific_count = full_df['discount_source'].str.contains('merchant_specific', na=False).sum()
average_count = full_df['discount_source'].str.contains('average', na=False).sum()
print(f"\n🎯 DISCOUNT APPLICATION METHOD:")
print(f"  • Merchant-specific discounts: {merchant_specific_count:,}")
print(f"  • Average layer discounts: {average_count:,}")
print(f"  • No discount applied: {(full_df['avg_discount'] == 0).sum():,}")

print(f"\n💰 DISCOUNT STATISTICS:")
discount_stats = full_df[full_df['avg_discount'] > 0]['avg_discount'].describe()
print(f"  • Mean discount: {discount_stats['mean']:.4f}%")
print(f"  • Median discount: {discount_stats['50%']:.4f}%")
print(f"  • Min discount: {discount_stats['min']:.4f}%")
print(f"  • Max discount: {discount_stats['max']:.4f}%")

print(f"\n💵 PRICE STATISTICS:")
print(f"  • Total discounted spend: ${full_df['discounted_price'].sum():,.2f}")
print(f"  • Total undiscounted spend: ${full_df['undiscounted_price'].sum():,.2f}")
print(f"  • Total discount value: ${(full_df['undiscounted_price'] - full_df['discounted_price']).sum():,.2f}")

print(f"\n📁 OUTPUT FILE:")
print(f"  • {output_file.name}")
print(f"    Contains invoice line items with avg_discount, discount_source, and undiscounted_price columns")

print(f"\n{'='*70}\n")