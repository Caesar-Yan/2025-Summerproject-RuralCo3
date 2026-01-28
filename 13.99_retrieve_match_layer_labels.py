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

# Get ATS numbers with their match_layer labels from both columns
# For matched_ats_number (L1 matches)
ats_l1_df = df[df['matched_ats_number'].notna()][['matched_ats_number', 'match_layer']].copy()
ats_l1_df.columns = ['matched_ats_number', 'match_layer']

# For matched_ats_number_L2 (L2 matches)
ats_l2_df = df[df['matched_ats_number_L2'].notna()][['matched_ats_number_L2', 'match_layer']].copy()
ats_l2_df.columns = ['matched_ats_number', 'match_layer']

# Combine both DataFrames
all_ats_df = pd.concat([ats_l1_df, ats_l2_df], ignore_index=True)

# Get unique combinations of ATS number and match_layer
merchants_df = all_ats_df.drop_duplicates().sort_values('matched_ats_number').reset_index(drop=True)

print(f"\nUnique ATS numbers from matched_ats_number: {df['matched_ats_number'].dropna().nunique():,}")
print(f"Unique ATS numbers from matched_ats_number_L2: {df['matched_ats_number_L2'].dropna().nunique():,}")
print(f"Total unique ATS numbers: {merchants_df['matched_ats_number'].nunique():,}")
print(f"Total unique ATS-match_layer combinations: {len(merchants_df):,}")

print(f"\nFirst 20 matched ATS numbers with their match_layer:")
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