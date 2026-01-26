'''
Docstring for 13.5_L8_matching_continued
This script continues on from the work done in the 13.4_L7_matching_continued script.
L1 and L2 matching was done based on merchant id numbers and parsing text fields in invoice data
L3 matching method was done on merchant_branch
L4 matching in 13.1 was done on description (freight, petrol, diesel, shop, bookkeeping artefacts, null descriptions)
L5 matching in 13.2 was done on description (gas products, vet services, car wash, rounding)
L6 matching in 13.3 was done on description (fuel, vet products, shop items, freight)
L7 matching in 13.4 was done on description (mechanic services, vet products, equipment hire)
L8 matching in 13.5 continues with additional description-based matching

Inputs:
- 13.4_matching_progress.csv
- 13.4_invoice_line_items_still_unmatched.csv
- Merchant Discount Detail.xlsx

Outputs:
- 13.5_matching_progress.csv
- 13.5_invoice_line_items_still_unmatched.csv
- 13.5_filtered_mask_WIP.csv (temporary working file)
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
full_df = pd.read_csv(merchant_folder_dir / '13.4_matching_progress.csv')

# Load merchant reference data
MERCHANT_PATH = base_dir.parent / "Merchant Discount Detail.xlsx"
merchant_df = pd.read_excel(MERCHANT_PATH)

# Load remaining unmatched items from previous script
unmatched_df = pd.read_csv(merchant_folder_dir / '13.4_invoice_line_items_still_unmatched.csv')

print(f"\n{'='*70}")
print(f"STARTING 13.5 L8 MATCHING")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Currently unmatched rows: {len(unmatched_df):,}")
print(f"\nCurrent match_layer distribution:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}\n")

# =========================================================
# CONTINUE L8 MATCHING
# =========================================================

# =========================================================================================================================
# VETSERVE DRONTAL MAILOUT - 35KG DOG ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "VetServe Drontal Mailout - 35kg Dog" in description column
mask_drontal = unmatched_df['description'].str.lower() == 'vetserve drontal mailout - 35kg dog'

drontal_df = save_and_summarize2(
    unmatched_df, 
    mask_drontal, 
    '13.5_filtered_mask_WIP.csv',
    'vetserve drontal mailout - 35kg dog',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vetserve drontal mailout - 35kg dog':")
print(drontal_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH VETSERVE DRONTAL MAILOUT - 35KG DOG FLAGS
# =========================================================
# Update the vetserve drontal mailout - 35kg dog rows in the FULL dataframe
mask_drontal_full = (
    (full_df['description'].str.lower() == 'vetserve drontal mailout - 35kg dog') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_drontal_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_drontal_full.sum():,} vetserve drontal mailout - 35kg dog rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after vetserve drontal mailout - 35kg dog update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 VETSERVE DRONTAL MAILOUT - 35KG DOG MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"VetServe Drontal Mailout - 35kg Dog rows flagged: {mask_drontal_full.sum():,}")
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
# PALM KERNEL (KG) ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "PALM KERNEL  (KG)" in description column
mask_palm_kernel = remaining_unmatched_df['description'].str.lower() == 'palm kernel  (kg)'

palm_kernel_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_palm_kernel, 
    '13.5_filtered_mask_WIP.csv',
    'palm kernel  (kg)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'palm kernel  (kg)':")
print(palm_kernel_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH PALM KERNEL (KG) FLAGS
# =========================================================
# Update the palm kernel  (kg) rows in the FULL dataframe
mask_palm_kernel_full = (
    (full_df['description'].str.lower() == 'palm kernel  (kg)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_palm_kernel_full, 'match_layer'] = 'L7_cattle_feed'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_palm_kernel_full.sum():,} palm kernel  (kg) rows to 'L7_cattle_feed'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after palm kernel  (kg) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 PALM KERNEL (KG) MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"PALM KERNEL  (KG) rows flagged: {mask_palm_kernel_full.sum():,}")
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
# LUBES ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "LUBES" in description column
mask_lubes = remaining_unmatched_df['description'].str.lower() == 'lubes'

lubes_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_lubes, 
    '13.5_filtered_mask_WIP.csv',
    'lubes',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'lubes':")
print(lubes_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LUBES FLAGS
# =========================================================
# Update the lubes rows in the FULL dataframe
mask_lubes_full = (
    (full_df['description'].str.lower() == 'lubes') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_lubes_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_lubes_full.sum():,} lubes rows to 'L7_mechanic'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after lubes update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 LUBES MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"LUBES rows flagged: {mask_lubes_full.sum():,}")
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
# VACCINE : VANGUARD 5 PLUS ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Vaccine : Vanguard 5 Plus" in description column
mask_vanguard = remaining_unmatched_df['description'].str.lower() == 'vaccine : vanguard 5 plus'

vanguard_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_vanguard, 
    '13.5_filtered_mask_WIP.csv',
    'vaccine : vanguard 5 plus',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vaccine : vanguard 5 plus':")
print(vanguard_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH VACCINE : VANGUARD 5 PLUS FLAGS
# =========================================================
# Update the vaccine : vanguard 5 plus rows in the FULL dataframe
mask_vanguard_full = (
    (full_df['description'].str.lower() == 'vaccine : vanguard 5 plus') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_vanguard_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_vanguard_full.sum():,} vaccine : vanguard 5 plus rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after vaccine : vanguard 5 plus update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 VACCINE : VANGUARD 5 PLUS MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Vaccine : Vanguard 5 Plus rows flagged: {mask_vanguard_full.sum():,}")
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
# LA - CONSUM REPRO & SCANNING ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "LA - Consum Repro & Scanning" in description column
mask_repro = remaining_unmatched_df['description'].str.lower() == 'la - consum repro & scanning'

repro_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_repro, 
    '13.5_filtered_mask_WIP.csv',
    'la - consum repro & scanning',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum repro & scanning':")
print(repro_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - CONSUM REPRO & SCANNING FLAGS
# =========================================================
# Update the la - consum repro & scanning rows in the FULL dataframe
mask_repro_full = (
    (full_df['description'].str.lower() == 'la - consum repro & scanning') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_repro_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_repro_full.sum():,} la - consum repro & scanning rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - consum repro & scanning update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 LA - CONSUM REPRO & SCANNING MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"LA - Consum Repro & Scanning rows flagged: {mask_repro_full.sum():,}")
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
# Q-SCOOP-SOIL- SCREENED ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Q-Scoop-Soil- Screened" in description column
mask_qscoop = remaining_unmatched_df['description'].str.lower() == 'q-scoop-soil- screened'

qscoop_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_qscoop, 
    '13.5_filtered_mask_WIP.csv',
    'q-scoop-soil- screened',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-scoop-soil- screened':")
print(qscoop_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH Q-SCOOP-SOIL- SCREENED FLAGS
# =========================================================
# Update the q-scoop-soil- screened rows in the FULL dataframe
mask_qscoop_full = (
    (full_df['description'].str.lower() == 'q-scoop-soil- screened') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_qscoop_full, 'match_layer'] = 'L8_Landscaping'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_qscoop_full.sum():,} q-scoop-soil- screened rows to 'L8_Landscaping'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after q-scoop-soil- screened update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L8 Q-SCOOP-SOIL- SCREENED MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Q-Scoop-Soil- Screened rows flagged: {mask_qscoop_full.sum():,}")
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
# PROLAJECT B12 2000 + SELENIUM 500ML ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Prolaject B12 2000 + Selenium 500ml" in description column
mask_prolaject = remaining_unmatched_df['description'].str.lower() == 'prolaject b12 2000 + selenium 500ml'

prolaject_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_prolaject, 
    '13.5_filtered_mask_WIP.csv',
    'prolaject b12 2000 + selenium 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'prolaject b12 2000 + selenium 500ml':")
print(prolaject_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH PROLAJECT B12 2000 + SELENIUM 500ML FLAGS
# =========================================================
# Update the prolaject b12 2000 + selenium 500ml rows in the FULL dataframe
mask_prolaject_full = (
    (full_df['description'].str.lower() == 'prolaject b12 2000 + selenium 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_prolaject_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_prolaject_full.sum():,} prolaject b12 2000 + selenium 500ml rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after prolaject b12 2000 + selenium 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 PROLAJECT B12 2000 + SELENIUM 500ML MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Prolaject B12 2000 + Selenium 500ml rows flagged: {mask_prolaject_full.sum():,}")
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
# NEXGARD SPECTRA FOR CATS (2.5-7.4KG) - S ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Nexgard Spectra for Cats (2.5-7.4kg) - S" in description column
mask_nexgard = remaining_unmatched_df['description'].str.lower() == 'nexgard spectra for cats (2.5-7.4kg) - s'

nexgard_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_nexgard, 
    '13.5_filtered_mask_WIP.csv',
    'nexgard spectra for cats (2.5-7.4kg) - s',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'nexgard spectra for cats (2.5-7.4kg) - s':")
print(nexgard_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH NEXGARD SPECTRA FOR CATS (2.5-7.4KG) - S FLAGS
# =========================================================
# Update the nexgard spectra for cats (2.5-7.4kg) - s rows in the FULL dataframe
mask_nexgard_full = (
    (full_df['description'].str.lower() == 'nexgard spectra for cats (2.5-7.4kg) - s') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_nexgard_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_nexgard_full.sum():,} nexgard spectra for cats (2.5-7.4kg) - s rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after nexgard spectra for cats (2.5-7.4kg) - s update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 NEXGARD SPECTRA FOR CATS (2.5-7.4KG) - S MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Nexgard Spectra for Cats (2.5-7.4kg) - S rows flagged: {mask_nexgard_full.sum():,}")
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
# CA : ANAESTHESIA : PATIENT MONITORING : ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "CA : Anaesthesia : Patient Monitoring :" in description column
mask_anaesthesia = remaining_unmatched_df['description'].str.lower() == 'ca : anaesthesia : patient monitoring :'

anaesthesia_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_anaesthesia, 
    '13.5_filtered_mask_WIP.csv',
    'ca : anaesthesia : patient monitoring :',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : anaesthesia : patient monitoring :':")
print(anaesthesia_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH CA : ANAESTHESIA : PATIENT MONITORING : FLAGS
# =========================================================
# Update the ca : anaesthesia : patient monitoring : rows in the FULL dataframe
mask_anaesthesia_full = (
    (full_df['description'].str.lower() == 'ca : anaesthesia : patient monitoring :') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_anaesthesia_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_anaesthesia_full.sum():,} ca : anaesthesia : patient monitoring : rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after ca : anaesthesia : patient monitoring : update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 CA : ANAESTHESIA : PATIENT MONITORING : MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"CA : Anaesthesia : Patient Monitoring : rows flagged: {mask_anaesthesia_full.sum():,}")
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
# LA - EXAMINATION ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "LA - Examination" in description column
mask_exam = remaining_unmatched_df['description'].str.lower() == 'la - examination'

exam_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_exam, 
    '13.5_filtered_mask_WIP.csv',
    'la - examination',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - examination':")
print(exam_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - EXAMINATION FLAGS
# =========================================================
# Update the la - examination rows in the FULL dataframe
mask_exam_full = (
    (full_df['description'].str.lower() == 'la - examination') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_exam_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_exam_full.sum():,} la - examination rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - examination update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 LA - EXAMINATION MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"LA - Examination rows flagged: {mask_exam_full.sum():,}")
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
# LUBE (PER LITRE) ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_lube = remaining_unmatched_df['description'].str.lower() == 'lube (per litre)'

lube_df = save_and_summarize2(
    remaining_unmatched_df, 
    mask_lube, 
    '13.5_filtered_mask_WIP.csv',
    'lube (per litre)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'lube (per litre)':")
print(lube_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LUBE (PER LITRE) FLAGS
# =========================================================
# Update the lube (per litre) rows in the FULL dataframe
mask_lube_full = (
    (full_df['description'].str.lower() == 'lube (per litre)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_lube_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_lube_full.sum():,} lube (per litre) rows to 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.5_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.5_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.5_invoice_line_items_still_unmatched.csv',
    'Still unmatched after lube (per litre) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 LUBE (PER LITRE) MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Lube (Per Litre) rows flagged: {mask_lube_full.sum():,}")
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