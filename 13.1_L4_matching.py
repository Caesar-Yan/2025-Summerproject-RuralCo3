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
# PETROL 91 ANALYSIS
# =========================================================
# Filter for rows with exact "petrol 91" in description column
mask_petrol = remaining_unmatched_df['description'].str.lower() == 'petrol 91'

petrol_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_petrol, 
    '13.1_filtered_mask_WIP.csv',
    'Petrol 91',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'petrol 91':")
print(petrol_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH PETROL FLAGS
# =========================================================
# Update the petrol rows in the FULL dataframe
mask_petrol_full = (
    (full_df['description'].str.lower() == 'petrol 91') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_petrol_full, 'match_layer'] = 'L4_petrol_no_merchant'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_petrol_full.sum():,} petrol rows to 'L4_petrol_no_merchant'")
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
    'Still unmatched after petrol update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 PETROL MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Petrol rows flagged: {mask_petrol_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# PETROL 95/96 ANALYSIS
# =========================================================
# Filter for rows with exact "petrol 95/96" in description column
mask_petrol = remaining_unmatched_df['description'].str.lower() == 'petrol 95/96'

petrol_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_petrol, 
    '13.1_filtered_mask_WIP.csv',
    'Petrol 95/96',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'petrol 95/96':")
print(petrol_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH PETROL FLAGS
# =========================================================
# Update the petrol rows in the FULL dataframe
mask_petrol_full = (
    (full_df['description'].str.lower() == 'petrol 95/96') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_petrol_full, 'match_layer'] = 'L4_petrol_no_merchant'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_petrol_full.sum():,} petrol rows to 'L4_petrol_no_merchant'")
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
    'Still unmatched after petrol update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 PETROL MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Petrol rows flagged: {mask_petrol_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================================================================================
# PETROL 98 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "petrol 98" in description column
mask_petrol = remaining_unmatched_df['description'].str.lower() == 'petrol 98'

petrol_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_petrol, 
    '13.1_filtered_mask_WIP.csv',
    'Petrol 98',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'petrol 98':")
print(petrol_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH PETROL FLAGS
# =========================================================
# Update the petrol rows in the FULL dataframe
mask_petrol_full = (
    (full_df['description'].str.lower() == 'petrol 98') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_petrol_full, 'match_layer'] = 'L4_petrol_no_merchant'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_petrol_full.sum():,} petrol rows to 'L4_petrol_no_merchant'")
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
    'Still unmatched after petrol update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 PETROL MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Petrol rows flagged: {mask_petrol_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================================================================================
# DIESEL ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "diesel" in description column
mask_diesel = remaining_unmatched_df['description'].str.lower() == 'diesel'

diesel_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_diesel, 
    '13.1_filtered_mask_WIP.csv',
    'diesel',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'diesel':")
print(diesel_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH DIESEL FLAGS
# =========================================================
# Update the diesel rows in the FULL dataframe
mask_diesel_full = (
    (full_df['description'].str.lower() == 'diesel') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_diesel_full, 'match_layer'] = 'L4_diesel_no_merchant'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_diesel_full.sum():,} diesel rows to 'L4_diesel_no_merchant'")
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
    'Still unmatched after diesel update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 DIESEL MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Diesel rows flagged: {mask_diesel_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================================================================================
# SHOP ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "shop" in description column
mask_shop = remaining_unmatched_df['description'].str.lower() == 'shop'

shop_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_shop, 
    '13.1_filtered_mask_WIP.csv',
    'shop',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'shop':")
print(shop_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH SHOP FLAGS
# =========================================================
# Update the shop rows in the FULL dataframe
mask_shop_full = (
    (full_df['description'].str.lower() == 'shop') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_shop_full, 'match_layer'] = 'L4_shop_no_merchant'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_shop_full.sum():,} shop rows to 'L4_shop_no_merchant'")
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
    'Still unmatched after shop update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 SHOP MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Shop rows flagged: {mask_shop_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================================================================================
# STATEMENT FEE ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "statement fee" in description column
mask_statement_fee = remaining_unmatched_df['description'].str.lower() == 'statement fee'

statement_fee_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_statement_fee, 
    '13.1_filtered_mask_WIP.csv',
    'statement fee',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'statement fee':")
print(statement_fee_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH STATEMENT FEE FLAGS
# =========================================================
# Update the statement fee rows in the FULL dataframe
mask_statement_fee_full = (
    (full_df['description'].str.lower() == 'statement fee') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_statement_fee_full, 'match_layer'] = 'L4_bookkeeping_artefacts'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_statement_fee_full.sum():,} statement fee rows to 'L4_bookkeeping_artefacts'")
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
    'Still unmatched after statement fee update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 STATEMENT FEE MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Statement fee rows flagged: {mask_statement_fee_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================================================================================
# PAYMENT MADE ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "payment made" in description column
mask_payment_made = remaining_unmatched_df['description'].str.lower() == 'paymentmade'

payment_made_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_payment_made, 
    '13.1_filtered_mask_WIP.csv',
    'paymentmade',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'payment made':")
print(payment_made_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH PAYMENT MADE FLAGS
# =========================================================
# Update the payment made rows in the FULL dataframe
mask_payment_made_full = (
    (full_df['description'].str.lower() == 'paymentmade') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_payment_made_full, 'match_layer'] = 'L4_bookkeeping_artefacts'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_payment_made_full.sum():,} payment made rows to 'L4_bookkeeping_artefacts'")
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
    'Still unmatched after payment made update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 PAYMENT MADE MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Payment made rows flagged: {mask_payment_made_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# =========================================================================================================================
# LESS DISCOUNTS ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "less discount" in description column
mask_less_discounts = remaining_unmatched_df['description'].str.lower() == 'less discount'

less_discounts_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_less_discounts, 
    '13.1_filtered_mask_WIP.csv',
    'less discount',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'less discount':")
print(less_discounts_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LESS DISCOUNTS FLAGS
# =========================================================
# Update the less discount rows in the FULL dataframe
mask_less_discount_full = (
    (full_df['description'].str.lower() == 'less discount') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_less_discount_full, 'match_layer'] = 'L4_bookkeeping_artefacts'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_less_discount_full.sum():,} less discount rows to 'L4_bookkeeping_artefacts'")
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
    'Still unmatched after less discounts update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 LESS DISCOUNT MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Less discounts rows flagged: {mask_less_discount_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")