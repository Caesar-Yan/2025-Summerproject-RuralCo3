'''
Docstring for 13.1_L4_matching
this script is continuing on from the work done in the 13_ script.
L1 and L2 matching was done based on merchant id numbers and parsing text fields in invoice data
L3 matching method was done on merchant_branch
L4 matching is being done on description

Inputs:
- 13_invoice_line_items_still_unmatched.csv
- Merchant Discount Detail.xlsx

Outputs:
- 
'''
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
full_df = pd.read_csv(merchant_folder_dir / '13_matching_progress.csv')

# Load merchant reference data
MERCHANT_PATH = base_dir.parent / "Merchant Discount Detail.xlsx"
merchant_df = pd.read_excel(MERCHANT_PATH)

unmatched_df = pd.read_csv(merchant_folder_dir / '13_invoice_line_items_still_unmatched.csv')

# =========================================================
# FREIGHT ANALYSIS
# =========================================================
# Filter for rows with "freight" in description column
mask_freight = unmatched_df['description'].str.contains('freight', case=False, na=False)

freight_df = save_and_summarize2(
    unmatched_df, 
    mask_freight, 
    '13.1_filtered_mask_WIP.csv',
    'Freight items',
    output_dir=output_dir  # Add this parameter
)

# Print unique values
print("\nUnique description values containing 'freight':")
print(freight_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH FREIGHT FLAGS
# =========================================================
# Update the freight rows in the FULL dataframe
mask_freight_full = (
    full_df['description'].str.contains('freight', case=False, na=False) & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_freight_full, 'match_layer'] = 'L4_no_discount_freight'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_freight_full.sum():,} freight rows to 'L4_no_discount_freight'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.1_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.1_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.1_invoice_line_items_still_unmatched.csv',
    'Still unmatched after freight update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 FREIGHT MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Freight rows flagged: {mask_freight_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# PETROL ANALYSIS
# =========================================================
# Filter for rows with "petrol" in description column
mask_petrol = remaining_unmatched_df['description'].str.contains('petrol', case=False, na=False)

petrol_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_petrol, 
    '13.1_filtered_mask_WIP.csv',
    'Petrol items',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values containing 'petrol':")
print(petrol_df['description'].unique())

