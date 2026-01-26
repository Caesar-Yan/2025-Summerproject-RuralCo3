'''
Docstring for 13.2_L5_matching_continued
This script continues on from the work done in the 13.1_L4_matching script.
L1 and L2 matching was done based on merchant id numbers and parsing text fields in invoice data
L3 matching method was done on merchant_branch
L4 matching in 13.1 was done on description (freight, petrol, diesel, shop, bookkeeping artefacts, null descriptions)
L5 matching in 13.2 continues with additional description-based matching

Inputs:
- 13.1_matching_progress.csv
- 13.1_invoice_line_items_still_unmatched.csv
- Merchant Discount Detail.xlsx

Outputs:
- 13.2_matching_progress.csv
- 13.2_invoice_line_items_still_unmatched.csv
- 13.2_filtered_mask_WIP.csv (temporary working file)
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
full_df = pd.read_csv(merchant_folder_dir / '13.1_matching_progress.csv')

# Load merchant reference data
MERCHANT_PATH = base_dir.parent / "Merchant Discount Detail.xlsx"
merchant_df = pd.read_excel(MERCHANT_PATH)

# Load remaining unmatched items from previous script
unmatched_df = pd.read_csv(merchant_folder_dir / '13.1_invoice_line_items_still_unmatched.csv')

print(f"\n{'='*70}")
print(f"STARTING 13.2 L4 MATCHING")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Currently unmatched rows: {len(unmatched_df):,}")
print(f"\nCurrent match_layer distribution:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}\n")

# =========================================================================================================================
# INCLUDES DISCOUNT OF ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Includes discount of" in description column
mask_includes_discount = unmatched_df['description'].str.lower() == 'includes discount of'

includes_discount_df = save_and_summarize2(
    unmatched_df, 
    mask_includes_discount, 
    '13.2_filtered_mask_WIP.csv',
    'includes discount of',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'includes discount of':")
print(includes_discount_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH INCLUDES DISCOUNT OF FLAGS
# =========================================================
# Update the includes discount of rows in the FULL dataframe
mask_includes_discount_full = (
    (full_df['description'].str.lower() == 'includes discount of') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_includes_discount_full, 'match_layer'] = 'L4_bookkeeping_artefacts'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_includes_discount_full.sum():,} includes discount of rows to 'L4_bookkeeping_artefacts'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after includes discount of update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 INCLUDES DISCOUNT OF MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Includes discount of rows flagged: {mask_includes_discount_full.sum():,}")
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
# 10% DISCOUNT ON 45KG LPG ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "10% DISCOUNT ON 45KG LPG" in description column
mask_lpg_discount = remaining_unmatched_df['description'].str.lower() == '10% discount on 45kg lpg'

lpg_discount_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_lpg_discount, 
    '13.2_filtered_mask_WIP.csv',
    '10% discount on 45kg lpg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching '10% discount on 45kg lpg':")
print(lpg_discount_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH 10% DISCOUNT ON 45KG LPG FLAGS
# =========================================================
# Update the 10% discount on 45kg lpg rows in the FULL dataframe
mask_lpg_discount_full = (
    (full_df['description'].str.lower() == '10% discount on 45kg lpg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_lpg_discount_full, 'match_layer'] = 'L5_Gas'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_lpg_discount_full.sum():,} 10% discount on 45kg lpg rows to 'L5_Gas'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after 10% discount on 45kg lpg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 10% DISCOUNT ON 45KG LPG MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"10% discount on 45kg lpg rows flagged: {mask_lpg_discount_full.sum():,}")
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
# ONGAS CYLINDER RENTAL ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "ONGAS CYLINDER RENTAL" in description column
mask_ongas_rental = remaining_unmatched_df['description'].str.lower() == 'ongas cylinder rental'

ongas_rental_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_ongas_rental, 
    '13.2_filtered_mask_WIP.csv',
    'ongas cylinder rental',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ongas cylinder rental':")
print(ongas_rental_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH ONGAS CYLINDER RENTAL FLAGS
# =========================================================
# Update the ongas cylinder rental rows in the FULL dataframe
mask_ongas_rental_full = (
    (full_df['description'].str.lower() == 'ongas cylinder rental') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_ongas_rental_full, 'match_layer'] = 'L5_Gas'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_ongas_rental_full.sum():,} ongas cylinder rental rows to 'L5_Gas'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after ongas cylinder rental update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ONGAS CYLINDER RENTAL MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Ongas cylinder rental rows flagged: {mask_ongas_rental_full.sum():,}")
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
# LPG BOTTLES ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "LPG Bottles" in description column
mask_lpg_bottles = remaining_unmatched_df['description'].str.lower() == 'lpg bottles'

lpg_bottles_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_lpg_bottles, 
    '13.2_filtered_mask_WIP.csv',
    'lpg bottles',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'lpg bottles':")
print(lpg_bottles_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LPG BOTTLES FLAGS
# =========================================================
# Update the lpg bottles rows in the FULL dataframe
mask_lpg_bottles_full = (
    (full_df['description'].str.lower() == 'lpg bottles') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_lpg_bottles_full, 'match_layer'] = 'L5_Gas'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_lpg_bottles_full.sum():,} lpg bottles rows to 'L5_Gas'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after lpg bottles update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 LPG BOTTLES MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"LPG bottles rows flagged: {mask_lpg_bottles_full.sum():,}")
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
# MONTHLY CYLINDER RENTAL ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "MONTHLY CYLINDER RENTAL" in description column
mask_monthly_rental = remaining_unmatched_df['description'].str.lower() == 'monthly cylinder rental'

monthly_rental_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_monthly_rental, 
    '13.2_filtered_mask_WIP.csv',
    'monthly cylinder rental',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'monthly cylinder rental':")
print(monthly_rental_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH MONTHLY CYLINDER RENTAL FLAGS
# =========================================================
# Update the monthly cylinder rental rows in the FULL dataframe
mask_monthly_rental_full = (
    (full_df['description'].str.lower() == 'monthly cylinder rental') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_monthly_rental_full, 'match_layer'] = 'L5_Gas'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_monthly_rental_full.sum():,} monthly cylinder rental rows to 'L5_Gas'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after monthly cylinder rental update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 MONTHLY CYLINDER RENTAL MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Monthly cylinder rental rows flagged: {mask_monthly_rental_full.sum():,}")
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
# DISCOUNT: CARE PLAN ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Discount: Care Plan" in description column
mask_care_plan = remaining_unmatched_df['description'].str.lower() == 'discount: care plan'

care_plan_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_care_plan, 
    '13.2_filtered_mask_WIP.csv',
    'discount: care plan',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'discount: care plan':")
print(care_plan_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH DISCOUNT: CARE PLAN FLAGS
# =========================================================
# Update the discount: care plan rows in the FULL dataframe
mask_care_plan_full = (
    (full_df['description'].str.lower() == 'discount: care plan') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_care_plan_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_care_plan_full.sum():,} discount: care plan rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after discount: care plan update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 DISCOUNT: CARE PLAN MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Discount: care plan rows flagged: {mask_care_plan_full.sum():,}")
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
# LA - FARM VISIT FEE ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "LA - Farm Visit Fee" in description column
mask_farm_visit = remaining_unmatched_df['description'].str.lower() == 'la - farm visit fee'

farm_visit_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_farm_visit, 
    '13.2_filtered_mask_WIP.csv',
    'la - farm visit fee',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - farm visit fee':")
print(farm_visit_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - FARM VISIT FEE FLAGS
# =========================================================
# Update the la - farm visit fee rows in the FULL dataframe
mask_farm_visit_full = (
    (full_df['description'].str.lower() == 'la - farm visit fee') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_farm_visit_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_farm_visit_full.sum():,} la - farm visit fee rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - farm visit fee update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 LA - FARM VISIT FEE MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"LA - Farm visit fee rows flagged: {mask_farm_visit_full.sum():,}")
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
# CAREPLANDEBIT ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "CarePlanDebit" in description column
mask_careplandebit = remaining_unmatched_df['description'].str.lower() == 'careplandebit'

careplandebit_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_careplandebit, 
    '13.2_filtered_mask_WIP.csv',
    'careplandebit',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'careplandebit':")
print(careplandebit_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH CAREPLANDEBIT FLAGS
# =========================================================
# Update the careplandebit rows in the FULL dataframe
mask_careplandebit_full = (
    (full_df['description'].str.lower() == 'careplandebit') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_careplandebit_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_careplandebit_full.sum():,} careplandebit rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after careplandebit update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 CAREPLANDEBIT MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"CarePlanDebit rows flagged: {mask_careplandebit_full.sum():,}")
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
# CAR WASH ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "CAR WASH" in description column
mask_car_wash = remaining_unmatched_df['description'].str.lower() == 'car wash'

car_wash_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_car_wash, 
    '13.2_filtered_mask_WIP.csv',
    'car wash',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'car wash':")
print(car_wash_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH CAR WASH FLAGS
# =========================================================
# Update the car wash rows in the FULL dataframe
mask_car_wash_full = (
    (full_df['description'].str.lower() == 'car wash') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_car_wash_full, 'match_layer'] = 'L3_no_discount_offered'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_car_wash_full.sum():,} car wash rows to 'L3_no_discount_offered'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after car wash update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 CAR WASH MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Car wash rows flagged: {mask_car_wash_full.sum():,}")
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
# ROUNDING ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "rounding" in description column
mask_rounding = remaining_unmatched_df['description'].str.lower() == 'rounding'

rounding_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_rounding, 
    '13.2_filtered_mask_WIP.csv',
    'rounding',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'rounding':")
print(rounding_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH ROUNDING FLAGS
# =========================================================
# Update the rounding rows in the FULL dataframe
mask_rounding_full = (
    (full_df['description'].str.lower() == 'rounding') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_rounding_full, 'match_layer'] = 'L4_bookkeeping_artefacts'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_rounding_full.sum():,} rounding rows to 'L4_bookkeeping_artefacts'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.2_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.2_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.2_invoice_line_items_still_unmatched.csv',
    'Still unmatched after rounding update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 ROUNDING MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Rounding rows flagged: {mask_rounding_full.sum():,}")
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

