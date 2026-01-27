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
# LOAD FULL PARENT FILE
# =========================================================
full_df = pd.read_csv(merchant_folder_dir / '13.5_matching_progress.csv')

# Load merchant reference data
MERCHANT_PATH = base_dir.parent / "Merchant Discount Detail.xlsx"
merchant_df = pd.read_excel(MERCHANT_PATH)

# Load remaining unmatched items from previous script
unmatched_df = pd.read_csv(merchant_folder_dir / '13.5_invoice_line_items_still_unmatched.csv')

print(f"\n{'='*70}")
print(f"STARTING 13.6 L9 MATCHING")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Currently unmatched rows: {len(unmatched_df):,}")
print(f"\nCurrent match_layer distribution:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}\n")

# =========================================================
# CONTINUE L9 MATCHING
# =========================================================

# =========================================================================================================================
# LA - Vet Per Min Charge ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'la - vet per min charge'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.6_filtered_mask_WIP.csv',
    'la - vet per min charge',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - vet per min charge':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la - vet per min charge rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'la - vet per min charge') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} la - vet per min charge 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.6_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.6_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.6_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - vet per min charge update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - vet per min charge MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - vet per min charge rows flagged: {mask_Charge_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 20 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(20))
print(f"{'='*70}\n")
