'''
Docstring for 13.3_L6_matching_continued
This script continues on from the work done in the 13.2_L4_matching script.
L1 and L2 matching was done based on merchant id numbers and parsing text fields in invoice data
L3 matching method was done on merchant_branch
L4 matching in 13.1 was done on description (freight, petrol, diesel, shop, bookkeeping artefacts, null descriptions)
L5 matching in 13.2 was done on description (gas products, vet services, car wash, rounding)
L6 matching in 13.3 continues with additional description-based matching

Inputs:
- 13.2_matching_progress.csv
- 13.2_invoice_line_items_still_unmatched.csv
- Merchant Discount Detail.xlsx

Outputs:
- 13.3_matching_progress.csv
- 13.3_invoice_line_items_still_unmatched.csv
- 13.3_filtered_mask_WIP.csv (temporary working file)
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
full_df = pd.read_csv(merchant_folder_dir / '13.2_matching_progress.csv')

# Load merchant reference data
MERCHANT_PATH = base_dir.parent / "Merchant Discount Detail.xlsx"
merchant_df = pd.read_excel(MERCHANT_PATH)

# Load remaining unmatched items from previous script
unmatched_df = pd.read_csv(merchant_folder_dir / '13.2_invoice_line_items_still_unmatched.csv')

print(f"\n{'='*70}")
print(f"STARTING 13.3 L6 MATCHING")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Currently unmatched rows: {len(unmatched_df):,}")
print(f"\nCurrent match_layer distribution:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}\n")

# =========================================================
# CONTINUE L6 MATCHING
# =========================================================

# =========================================================================================================================
# FUEL ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "fuel" in description column
mask_fuel = unmatched_df['description'].str.lower() == 'fuel'

fuel_df = save_and_summarize2(
    unmatched_df, 
    mask_fuel, 
    '13.3_filtered_mask_WIP.csv',
    'fuel',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'fuel':")
print(fuel_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH FUEL FLAGS
# =========================================================
# Update the fuel rows in the FULL dataframe
mask_fuel_full = (
    (full_df['description'].str.lower() == 'fuel') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_fuel_full, 'match_layer'] = 'L6_equipment_hire_fuel'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_fuel_full.sum():,} fuel rows to 'L6_equipment_hire_fuel'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.3_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.3_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.3_invoice_line_items_still_unmatched.csv',
    'Still unmatched after fuel update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L6 FUEL MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Fuel rows flagged: {mask_fuel_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 20 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(remaining_unmatched_df['description'].value_counts().head(20))
print(f"{'='*70}\n")

# =========================================================================================================================
# VETSERVE DRONCIT MAILOUT TABLET - 20KG D ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "VetServe Droncit Mailout Tablet - 20kg D" in description column
mask_droncit = remaining_unmatched_df['description'].str.lower() == 'vetserve droncit mailout tablet - 20kg d'

droncit_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_droncit, 
    '13.3_filtered_mask_WIP.csv',
    'vetserve droncit mailout tablet - 20kg d',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vetserve droncit mailout tablet - 20kg d':")
print(droncit_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH VETSERVE DRONCIT MAILOUT TABLET - 20KG D FLAGS
# =========================================================
# Update the vetserve droncit mailout tablet - 20kg d rows in the FULL dataframe
mask_droncit_full = (
    (full_df['description'].str.lower() == 'vetserve droncit mailout tablet - 20kg d') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_droncit_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_droncit_full.sum():,} vetserve droncit mailout tablet - 20kg d rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.3_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.3_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.3_invoice_line_items_still_unmatched.csv',
    'Still unmatched after vetserve droncit mailout tablet - 20kg d update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 VETSERVE DRONCIT MAILOUT TABLET - 20KG D MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"VetServe Droncit Mailout Tablet - 20kg D rows flagged: {mask_droncit_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 20 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(remaining_unmatched_df['description'].value_counts().head(20))
print(f"{'='*70}\n")