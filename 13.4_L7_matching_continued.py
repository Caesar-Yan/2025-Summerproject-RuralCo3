'''
Docstring for 13.4_L7_matching_continued
This script continues on from the work done in the 13.3_L6_matching_continued script.
L1 and L2 matching was done based on merchant id numbers and parsing text fields in invoice data
L3 matching method was done on merchant_branch
L4 matching in 13.1 was done on description (freight, petrol, diesel, shop, bookkeeping artefacts, null descriptions)
L5 matching in 13.2 was done on description (gas products, vet services, car wash, rounding)
L6 matching in 13.3 was done on description (fuel, vet products, shop items, freight)
L7 matching in 13.4 continues with additional description-based matching

Inputs:
- 13.3_matching_progress.csv
- 13.3_invoice_line_items_still_unmatched.csv
- Merchant Discount Detail.xlsx

Outputs:
- 13.4_matching_progress.csv
- 13.4_invoice_line_items_still_unmatched.csv
- 13.4_filtered_mask_WIP.csv (temporary working file)
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
full_df = pd.read_csv(merchant_folder_dir / '13.3_matching_progress.csv')

# Load merchant reference data
MERCHANT_PATH = base_dir.parent / "Merchant Discount Detail.xlsx"
merchant_df = pd.read_excel(MERCHANT_PATH)

# Load remaining unmatched items from previous script
unmatched_df = pd.read_csv(merchant_folder_dir / '13.3_invoice_line_items_still_unmatched.csv')

print(f"\n{'='*70}")
print(f"STARTING 13.4 L7 MATCHING")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Currently unmatched rows: {len(unmatched_df):,}")
print(f"\nCurrent match_layer distribution:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}\n")

# =========================================================
# CONTINUE L7 MATCHING
# =========================================================

# =========================================================================================================================
# CONSUMABLE ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "consumable" in description column
mask_consumable = unmatched_df['description'].str.lower() == 'consumable'

consumable_df = save_and_summarize2(
    unmatched_df, 
    mask_consumable, 
    '13.4_filtered_mask_WIP.csv',
    'consumable',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'consumable':")
print(consumable_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH CONSUMABLE FLAGS
# =========================================================
# Update the consumable rows in the FULL dataframe
mask_consumable_full = (
    (full_df['description'].str.lower() == 'consumable') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_consumable_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_consumable_full.sum():,} consumable rows to 'L7_mechanic'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after consumable update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 CONSUMABLE MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Consumable rows flagged: {mask_consumable_full.sum():,}")
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
# CA : CONSULTATION : ANNUAL/TRIENNIAL VAC ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "CA : Consultation : Annual/Triennial Vac" in description column
mask_annual_vac = remaining_unmatched_df['description'].str.lower() == 'ca : consultation : annual/triennial vac'

annual_vac_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_annual_vac, 
    '13.4_filtered_mask_WIP.csv',
    'ca : consultation : annual/triennial vac',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : consultation : annual/triennial vac':")
print(annual_vac_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH CA : CONSULTATION : ANNUAL/TRIENNIAL VAC FLAGS
# =========================================================
# Update the ca : consultation : annual/triennial vac rows in the FULL dataframe
mask_annual_vac_full = (
    (full_df['description'].str.lower() == 'ca : consultation : annual/triennial vac') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_annual_vac_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_annual_vac_full.sum():,} ca : consultation : annual/triennial vac rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after ca : consultation : annual/triennial vac update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 CA : CONSULTATION : ANNUAL/TRIENNIAL VAC MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"CA : Consultation : Annual/Triennial Vac rows flagged: {mask_annual_vac_full.sum():,}")
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
# LA - CONSUM - NEEDLE INJECTOR (PER PACK ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "LA - Consum - Needle Injector (Per pack" in description column
mask_needle = remaining_unmatched_df['description'].str.lower() == 'la - consum - needle injector (per pack'

needle_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_needle, 
    '13.4_filtered_mask_WIP.csv',
    'la - consum - needle injector (per pack',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum - needle injector (per pack':")
print(needle_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - CONSUM - NEEDLE INJECTOR (PER PACK FLAGS
# =========================================================
# Update the la - consum - needle injector (per pack rows in the FULL dataframe
mask_needle_full = (
    (full_df['description'].str.lower() == 'la - consum - needle injector (per pack') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_needle_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_needle_full.sum():,} la - consum - needle injector (per pack rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - consum - needle injector (per pack update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 LA - CONSUM - NEEDLE INJECTOR (PER PACK MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"LA - Consum - Needle Injector (Per pack rows flagged: {mask_needle_full.sum():,}")
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
# CONTINUATION - TOILET HIRE ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "CONTINUATION - TOILET HIRE" in description column
mask_toilet = remaining_unmatched_df['description'].str.lower() == 'continuation - toilet hire'

toilet_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_toilet, 
    '13.4_filtered_mask_WIP.csv',
    'continuation - toilet hire',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'continuation - toilet hire':")
print(toilet_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH CONTINUATION - TOILET HIRE FLAGS
# =========================================================
# Update the continuation - toilet hire rows in the FULL dataframe
mask_toilet_full = (
    (full_df['description'].str.lower() == 'continuation - toilet hire') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_toilet_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_toilet_full.sum():,} continuation - toilet hire rows to 'L7_equipment_hire'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after continuation - toilet hire update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 CONTINUATION - TOILET HIRE MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"CONTINUATION - TOILET HIRE rows flagged: {mask_toilet_full.sum():,}")
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
# BOSS TRIPLE MINERALISED 20L ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Boss Triple Mineralised 20L" in description column
mask_boss = remaining_unmatched_df['description'].str.lower() == 'boss triple mineralised 20l'

boss_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_boss, 
    '13.4_filtered_mask_WIP.csv',
    'boss triple mineralised 20l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'boss triple mineralised 20l':")
print(boss_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH BOSS TRIPLE MINERALISED 20L FLAGS
# =========================================================
# Update the boss triple mineralised 20l rows in the FULL dataframe
mask_boss_full = (
    (full_df['description'].str.lower() == 'boss triple mineralised 20l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_boss_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_boss_full.sum():,} boss triple mineralised 20l rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after boss triple mineralised 20l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 BOSS TRIPLE MINERALISED 20L MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Boss Triple Mineralised 20L rows flagged: {mask_boss_full.sum():,}")
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
# VETRIMOXIN LA 250ML ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Vetrimoxin LA 250ml" in description column
mask_vetrimoxin = remaining_unmatched_df['description'].str.lower() == 'vetrimoxin la 250ml'

vetrimoxin_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_vetrimoxin, 
    '13.4_filtered_mask_WIP.csv',
    'vetrimoxin la 250ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vetrimoxin la 250ml':")
print(vetrimoxin_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH VETRIMOXIN LA 250ML FLAGS
# =========================================================
# Update the vetrimoxin la 250ml rows in the FULL dataframe
mask_vetrimoxin_full = (
    (full_df['description'].str.lower() == 'vetrimoxin la 250ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_vetrimoxin_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_vetrimoxin_full.sum():,} vetrimoxin la 250ml rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after vetrimoxin la 250ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 VETRIMOXIN LA 250ML MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Vetrimoxin LA 250ml rows flagged: {mask_vetrimoxin_full.sum():,}")
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
# XYLAZINE 2% ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Xylazine 2%" in description column
mask_xylazine = remaining_unmatched_df['description'].str.lower() == 'xylazine 2%'

xylazine_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_xylazine, 
    '13.4_filtered_mask_WIP.csv',
    'xylazine 2%',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'xylazine 2%':")
print(xylazine_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH XYLAZINE 2% FLAGS
# =========================================================
# Update the xylazine 2% rows in the FULL dataframe
mask_xylazine_full = (
    (full_df['description'].str.lower() == 'xylazine 2%') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_xylazine_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_xylazine_full.sum():,} xylazine 2% rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after xylazine 2% update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 XYLAZINE 2% MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Xylazine 2% rows flagged: {mask_xylazine_full.sum():,}")
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
# TECHNICIAN FEE ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "technician fee" in description column
mask_tech_fee = remaining_unmatched_df['description'].str.lower() == 'technician fee'

tech_fee_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_tech_fee, 
    '13.4_filtered_mask_WIP.csv',
    'technician fee',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'technician fee':")
print(tech_fee_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH TECHNICIAN FEE FLAGS
# =========================================================
# Update the technician fee rows in the FULL dataframe
mask_tech_fee_full = (
    (full_df['description'].str.lower() == 'technician fee') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_tech_fee_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_tech_fee_full.sum():,} technician fee rows to 'L7_mechanic'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after technician fee update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 TECHNICIAN FEE MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Technician fee rows flagged: {mask_tech_fee_full.sum():,}")
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
# CONSUMABLES ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "consumables" in description column
mask_consumables = remaining_unmatched_df['description'].str.lower() == 'consumables'

consumables_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_consumables, 
    '13.4_filtered_mask_WIP.csv',
    'consumables',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'consumables':")
print(consumables_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH CONSUMABLES FLAGS
# =========================================================
# Update the consumables rows in the FULL dataframe
mask_consumables_full = (
    (full_df['description'].str.lower() == 'consumables') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_consumables_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_consumables_full.sum():,} consumables rows to 'L7_mechanic'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after consumables update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 CONSUMABLES MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Consumables rows flagged: {mask_consumables_full.sum():,}")
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
# ADMINISTRATION ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "administration" in description column
mask_admin = remaining_unmatched_df['description'].str.lower() == 'administration'

admin_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_admin, 
    '13.4_filtered_mask_WIP.csv',
    'administration',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'administration':")
print(admin_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH ADMINISTRATION FLAGS
# =========================================================
# Update the administration rows in the FULL dataframe
mask_admin_full = (
    (full_df['description'].str.lower() == 'administration') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_admin_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_admin_full.sum():,} administration rows to 'L7_mechanic'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.4_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.4_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.4_invoice_line_items_still_unmatched.csv',
    'Still unmatched after administration update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 ADMINISTRATION MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Administration rows flagged: {mask_admin_full.sum():,}")
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