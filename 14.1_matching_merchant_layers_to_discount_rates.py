"""
Script: 14.1_matching_merchant_layers_to_discount_rates.py

Purpose:
    Calculate average discount rates for each match_layer from manually mapped merchants.

Inputs:
    - 14_filtered_merchants_with_manual_labels.csv

Outputs:
    - 14.1_average_discount_by_match_layer.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
data_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202')
output_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3/merchant')

# =========================================================
# LOAD MERCHANTS WITH MANUAL LABELS
# =========================================================
print("\n" + "="*70)
print("LOADING MERCHANTS WITH MANUAL LABELS")
print("="*70)

input_file = output_dir / '14_filtered_merchants_with_manual_labels.csv'
merchant_df = pd.read_csv(input_file)

print(f"Loaded {len(merchant_df):,} merchant records")
print(f"Columns: {merchant_df.columns.tolist()}")

# =========================================================
# FILTER TO MERCHANTS WITH MATCH LAYER
# =========================================================
print("\n" + "="*70)
print("FILTERING TO MERCHANTS WITH MATCH LAYER")
print("="*70)

# Keep only rows with a match_layer value
merchants_with_layer = merchant_df[merchant_df['match_layer'].notna()].copy()

print(f"Merchants with match_layer: {len(merchants_with_layer):,} ({len(merchants_with_layer)/len(merchant_df)*100:.1f}%)")
print(f"Unique match_layers: {merchants_with_layer['match_layer'].nunique()}")

# =========================================================
# CALCULATE AVERAGE DISCOUNT BY MATCH LAYER
# =========================================================
print("\n" + "="*70)
print("CALCULATING AVERAGE DISCOUNT BY MATCH LAYER")
print("="*70)

# Group by match_layer and calculate mean of cleaned_discount
discount_by_layer = merchants_with_layer.groupby('match_layer').agg({
    'cleaned_discount': ['mean', 'count', 'std', 'min', 'max']
}).reset_index()

# Flatten column names
discount_by_layer.columns = ['match_layer', 'avg_discount', 'count', 'std_discount', 'min_discount', 'max_discount']

# Sort by average discount descending
discount_by_layer = discount_by_layer.sort_values('avg_discount', ascending=False)

print("\nAverage Discount by Match Layer:")
print(discount_by_layer.to_string(index=False))

# =========================================================
# SAVE OUTPUT
# =========================================================
print("\n" + "="*70)
print("SAVING OUTPUT")
print("="*70)

output_file = output_dir / '14.1_average_discount_by_match_layer.csv'
discount_by_layer.to_csv(output_file, index=False)
print(f"Saved: {output_file.name} ({len(discount_by_layer)} rows)")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "="*70)
print("PROCESS COMPLETE!")
print("="*70)

print(f"\n📊 SUMMARY:")
print(f"  • Input merchants: {len(merchant_df):,}")
print(f"  • Merchants with match_layer: {len(merchants_with_layer):,}")
print(f"  • Unique match_layers: {len(discount_by_layer)}")
print(f"  • Average discount range: {discount_by_layer['avg_discount'].min():.4f} - {discount_by_layer['avg_discount'].max():.4f}")

print(f"\n📁 OUTPUT FILE:")
print(f"  • {output_file.name}")
print(f"    Contains average discount rates for each match_layer")