"""
Script: 13.99_retrieve_match_layer_labels.py

Purpose:
    Extract and save all unique match_layer labels and matched merchants from the 
    matching progress file.

Input:
    - 13.6_matching_progress.csv (or latest matching progress file)

Output:
    - 13.99_unique_match_layer_labels.csv
    - 13.99_matched_merchants.csv
"""

import pandas as pd
from pathlib import Path

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
merchant_folder_dir = base_dir / "merchant"
output_dir = merchant_folder_dir

# =========================================================
# LOAD MATCHING PROGRESS FILE
# =========================================================
progress_file = merchant_folder_dir / '13.6_matching_progress.csv'
df = pd.read_csv(progress_file)

print(f"\n{'='*70}")
print(f"EXTRACTING UNIQUE MATCH_LAYER LABELS AND MERCHANTS")
print(f"{'='*70}")
print(f"Total rows in progress file: {len(df):,}")

# =========================================================
# EXTRACT UNIQUE MATCH_LAYER VALUES
# =========================================================
print(f"\n{'='*70}")
print(f"EXTRACTING UNIQUE MATCH_LAYER VALUES")
print(f"{'='*70}")

# Get unique match_layer values
unique_match_layers = df['match_layer'].dropna().unique()

# Create DataFrame with match_layer values and their counts
match_layer_counts = df['match_layer'].value_counts().sort_index()
match_layers_df = pd.DataFrame({
    'match_layer': match_layer_counts.index,
    'count': match_layer_counts.values
})

print(f"\nFound {len(unique_match_layers)} unique match_layer values:")
print(match_layers_df.to_string(index=False))

# =========================================================
# SAVE UNIQUE MATCH_LAYER VALUES
# =========================================================
match_layer_output_file = output_dir / '13.99_unique_match_layer_labels.csv'
match_layers_df.to_csv(match_layer_output_file, index=False)

print(f"\n{'='*70}")
print(f"Saved unique match_layer labels to: {match_layer_output_file.name}")
print(f"{'='*70}\n")

# =========================================================
# EXTRACT UNIQUE MATCHED MERCHANTS (ATS NUMBERS)
# =========================================================
print(f"\n{'='*70}")
print(f"EXTRACTING UNIQUE MATCHED MERCHANTS")
print(f"{'='*70}")

# Get unique ATS numbers from both columns
ats_l1 = df['matched_ats_number'].dropna().unique()
ats_l2 = df['matched_ats_number_L2'].dropna().unique()

# Combine and get unique values
all_ats = pd.Series(list(ats_l1) + list(ats_l2)).unique()

print(f"\nUnique ATS numbers from matched_ats_number: {len(ats_l1):,}")
print(f"Unique ATS numbers from matched_ats_number_L2: {len(ats_l2):,}")
print(f"Total unique ATS numbers: {len(all_ats):,}")

# Create DataFrame with ATS numbers
merchants_df = pd.DataFrame({
    'matched_ats_number': sorted(all_ats)
})

print(f"\nFirst 20 matched ATS numbers:")
print(merchants_df.head(20).to_string(index=False))

# =========================================================
# SAVE MATCHED MERCHANTS
# =========================================================
merchants_output_file = output_dir / '13.99_matched_merchants.csv'
merchants_df.to_csv(merchants_output_file, index=False)

print(f"\n{'='*70}")
print(f"Saved matched merchants to: {merchants_output_file.name}")
print(f"{'='*70}\n")

# =========================================================
# CREATE MATCH_LAYER LABELS WITHOUT SPECIFIC MERCHANTS
# =========================================================
print(f"\n{'='*70}")
print(f"CREATING MATCH_LAYER LABELS (EXCLUDING SPECIFIC MERCHANT LAYERS)")
print(f"{'='*70}")

# Define layers to exclude
exclude_layers = ['L1', 'L2', 'L3_blackwoods', 'L3_methven_motors']

# Filter out excluded layers
filtered_match_layers_df = match_layers_df[~match_layers_df['match_layer'].isin(exclude_layers)].copy()

print(f"\nExcluded layers: {', '.join(exclude_layers)}")
print(f"Remaining match_layer values: {len(filtered_match_layers_df)}")
print(f"\nFiltered match_layer labels:")
print(filtered_match_layers_df.to_string(index=False))

# =========================================================
# SAVE FILTERED MATCH_LAYER LABELS
# =========================================================
filtered_output_file = output_dir / '13.99_match_layer_labels_without_merchants.csv'
filtered_match_layers_df.to_csv(filtered_output_file, index=False)

print(f"\n{'='*70}")
print(f"Saved filtered match_layer labels to: {filtered_output_file.name}")
print(f"{'='*70}\n")