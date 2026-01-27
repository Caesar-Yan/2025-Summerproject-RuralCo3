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

# =========================================================================================================================
# EXCAVATOR - 2.5tonne  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'excavator - 2.5tonne '

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'excavator - 2.5tonne',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'excavator - 2.5tonne':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'excavator - 2.5tonne') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L6_equipment_hire_fuel'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} excavator - 2.5tonne rows to 'L6_equipment_hire_fuel'")
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
    'Still unmatched after excavator - 2.5tonne update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L6 excavator - 2.5tonne MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"excavator - 2.5tonne rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# EXCAVATOR * 5 - 1.7tonne  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'excavator * 5 - 1.7tonne'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'excavator * 5 - 1.7tonne',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'excavator * 5 - 1.7tonne':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'excavator * 5 - 1.7tonne') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L6_equipment_hire_fuel'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} excavator * 5 - 1.7tonne rows to 'L6_equipment_hire_fuel'")
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
    'Still unmatched after excavator * 5 - 1.7tonne update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L6 excavator * 5 - 1.7tonne MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"excavator * 5 - 1.7tonne rows flagged: {mask_excavator_full.sum():,}")
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


# =========================================================================================================================
# W  Vehicle COF Truck 2 axle  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'w  vehicle cof truck 2 axle'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'w  vehicle cof truck 2 axle',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'w  vehicle cof truck 2 axle':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'w  vehicle cof truck 2 axle') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} w  vehicle cof truck 2 axle rows to 'L7_mechanic'")
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
    'Still unmatched after w  vehicle cof truck 2 axle update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 w  vehicle cof truck 2 axle MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"w  vehicle cof truck 2 axle rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Q-Scoop-Basecourse- AP20- Dobson St  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'q-scoop-basecourse- ap20- dobson st'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'q-scoop-basecourse- ap20- dobson st',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-scoop-basecourse- ap20- dobson st':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'q-scoop-basecourse- ap20- dobson st') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L9_infrastructure_consumables'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} q-scoop-basecourse- ap20- dobson st rows to 'L9_infrastructure_consumables'")
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
    'Still unmatched after q-scoop-basecourse- ap20- dobson st update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L9 q-scoop-basecourse- ap20- dobson st MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q-scoop-basecourse- ap20- dobson st rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# CA : Anaesthesia : Sedation : Standard  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : anaesthesia : sedation : standard'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : anaesthesia : sedation : standard',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : anaesthesia : sedation : standard':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : anaesthesia : sedation : standard') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : anaesthesia : sedation : standard rows to 'L5_Vet'")
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
    'Still unmatched after ca : anaesthesia : sedation : standard update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : anaesthesia : sedation : standard MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : anaesthesia : sedation : standard rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# 10.4 Semi Syn  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == '10.4 semi syn'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    '10.4 semi syn',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching '10.4 semi syn':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == '10.4 semi syn') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} 10.4 semi syn rows to 'L7_mechanic'")
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
    'Still unmatched after 10.4 semi syn update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7  10.4 semi syn MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"10.4 semi syn rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Torbugesic 10ml  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'torbugesic 10ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'torbugesic 10ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'torbugesic 10ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'torbugesic 10ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} torbugesic 10ml rows to 'L5_Vet'")
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
    'Still unmatched after torbugesic 10ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 torbugesic 10ml MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"torbugesic 10ml rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# VetServe Drontal Mailout - 10kg Dog ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'vetserve drontal mailout - 10kg dog'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'vetserve drontal mailout - 10kg dog',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vetserve drontal mailout - 10kg dog':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'vetserve drontal mailout - 10kg dog') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} vetserve drontal mailout - 10kg dog rows to 'L5_Vet'")
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
    'Still unmatched after vetserve drontal mailout - 10kg dog update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 vetserve drontal mailout - 10kg dog MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"vetserve drontal mailout - 10kg dog rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Bomacaine 500ml (Per Pack)  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'bomacaine 500ml (per pack)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'bomacaine 500ml (per pack)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'bomacaine 500ml (per pack)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'bomacaine 500ml (per pack)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} bomacaine 500ml (per pack) rows to 'L5_Vet'")
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
    'Still unmatched after bomacaine 500ml (per pack) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 bomacaine 500ml (per pack)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"bomacaine 500ml (per pack) rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Domitor  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'domitor'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'domitor',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'domitor':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'domitor') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} domitor rows to 'L5_Vet'")
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
    'Still unmatched after domitor update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 domitor  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"domitor rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Ultravac 5in1 B12 + Selenium 500ml   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ultravac 5in1 b12 + selenium 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ultravac 5in1 b12 + selenium 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ultravac 5in1 b12 + selenium 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ultravac 5in1 b12 + selenium 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ultravac 5in1 b12 + selenium 500ml rows to 'L5_Vet'")
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
    'Still unmatched after ultravac 5in1 b12 + selenium 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ultravac 5in1 b12 + selenium 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ultravac 5in1 b12 + selenium 500ml rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Q-Scoop-Premix- Shingle- Washed- Dobson   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'q-scoop-premix- shingle- washed- dobson'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'q-scoop-premix- shingle- washed- dobson',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-scoop-premix- shingle- washed- dobson':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'q-scoop-premix- shingle- washed- dobson') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L9_infrastructure_consumables'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} q-scoop-premix- shingle- washed- dobson rows to 'L9_infrastructure_consumables'")
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
    'Still unmatched after q-scoop-premix- shingle- washed- dobson update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L9 q-scoop-premix- shingle- washed- dobson  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q-scoop-premix- shingle- washed- dobson rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Rimadyl Chewable Tablets 100mg   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'rimadyl chewable tablets 100mg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'rimadyl chewable tablets 100mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'rimadyl chewable tablets 100mg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'rimadyl chewable tablets 100mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} rimadyl chewable tablets 100mg rows to 'L5_Vet'")
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
    'Still unmatched after rimadyl chewable tablets 100mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 rimadyl chewable tablets 100mg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"rimadyl chewable tablets 100mg rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Drontal Allwormer for Large Dogs (35kg T   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'drontal allwormer for large dogs (35kg t'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'drontal allwormer for large dogs (35kg t',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'drontal allwormer for large dogs (35kg t':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'drontal allwormer for large dogs (35kg t') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} drontal allwormer for large dogs (35kg t rows to 'L5_Vet'")
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
    'Still unmatched after drontal allwormer for large dogs (35kg t update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 drontal allwormer for large dogs (35kg t  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"drontal allwormer for large dogs (35kg t rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# CA : Prep/Non Theatre Fee (Non Sterile/T   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : prep/non theatre fee (non sterile/t'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : prep/non theatre fee (non sterile/t',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : prep/non theatre fee (non sterile/t':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : prep/non theatre fee (non sterile/t') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : prep/non theatre fee (non sterile/t rows to 'L5_Vet'")
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
    'Still unmatched after ca : prep/non theatre fee (non sterile/t update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : prep/non theatre fee (non sterile/t  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : prep/non theatre fee (non sterile/t rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Vaccine : Vanguard CC3 Intranasal   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'vaccine : vanguard cc3 intranasal'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'vaccine : vanguard cc3 intranasal',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vaccine : vanguard cc3 intranasal':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'vaccine : vanguard cc3 intranasal') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} vaccine : vanguard cc3 intranasal rows to 'L5_Vet'")
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
    'Still unmatched after vaccine : vanguard cc3 intranasal update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 vaccine : vanguard cc3 intranasal  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"vaccine : vanguard cc3 intranasal rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# BlackHawk Working Dog (Dry Food) - Per   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'blackhawk working dog (dry food) 2 - per'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'blackhawk working dog (dry food) - per 2',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'blackhawk working dog (dry food) - per 2':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'blackhawk working dog (dry food) - per 2') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} blackhawk working dog (dry food) - per 2 rows to 'L5_Vet'")
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
    'Still unmatched after blackhawk working dog (dry food) - per 2 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 blackhawk working dog (dry food) - per 2  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"blackhawk working dog (dry food) - per 2 rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# LA - Prescription Fee   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - prescription fee'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - prescription fee',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - prescription fee':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - prescription fee') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - prescription fee rows to 'L5_Vet'")
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
    'Still unmatched after la - prescription fee update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - rescription fee  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - prescription fee rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# la - dry cow or teatseal administration   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - dry cow or teatseal administration'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - dry cow or teatseal administration',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - dry cow or teatseal administration':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - dry cow or teatseal administration') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - dry cow or teatseal administration rows to 'L5_Vet'")
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
    'Still unmatched after la - dry cow or teatseal administration update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - dry cow or teatseal administration  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - dry cow or teatseal administration rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Nexgard Spectra for Cats (2.5-7.4kg) - 3  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'nexgard spectra for cats (2.5-7.4kg) - 3'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'nexgard spectra for cats (2.5-7.4kg) - 3',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'nexgard spectra for cats (2.5-7.4kg) - 3':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'nexgard spectra for cats (2.5-7.4kg) - 3') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} nexgard spectra for cats (2.5-7.4kg) - 3 rows to 'L5_Vet'")
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
    'Still unmatched after nexgard spectra for cats (2.5-7.4kg) - 3 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 nexgard spectra for cats (2.5-7.4kg) - 3  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"nexgard spectra for cats (2.5-7.4kg) - 3 rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Antisedan  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'antisedan'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'antisedan',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'antisedan':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'antisedan') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} antisedan rows to 'L5_Vet'")
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
    'Still unmatched after antisedan update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 antisedan  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"antisedan rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# LOGSPLITTER * VERTICAL   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'logsplitter * vertical'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'logsplitter * vertical',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'logsplitter * vertical':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'logsplitter * vertical') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} logsplitter * vertical rows to 'L5_Vet'")
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
    'Still unmatched after logsplitter * vertical update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 logsplitter * vertical  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"logsplitter * vertical rows flagged: {mask_excavator_full.sum():,}")
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


# =========================================================================================================================
# Q-Scoop-Zland- Compost- Supagrow   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'q-scoop-zland- compost- supagrow'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'q-scoop-zland- compost- supagrow',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-scoop-zland- compost- supagrow':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'q-scoop-zland- compost- supagrow') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} q-scoop-zland- compost- supagrow rows to 'L5_Vet'")
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
    'Still unmatched after q-scoop-zland- compost- supagrow update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 q-scoop-zland- compost- supagrow  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q-scoop-zland- compost- supagrow rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Tetravet Blue Aerosol 200g   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'tetravet blue aerosol 200g'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'tetravet blue aerosol 200g',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'tetravet blue aerosol 200g':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'tetravet blue aerosol 200g') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} tetravet blue aerosol 200g rows to 'L5_Vet'")
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
    'Still unmatched after tetravet blue aerosol 200g update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 tetravet blue aerosol 200g  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"tetravet blue aerosol 200g rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# LA - Consum Disbudding (Per Calf)   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - consum disbudding (per calf)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - consum disbudding (per calf)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum disbudding (per calf)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - consum disbudding (per calf)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - consum disbudding (per calf) rows to 'L5_Vet'")
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
    'Still unmatched after la - consum disbudding (per calf) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - consum disbudding (per calf)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - consum disbudding (per calf) rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Tenaline LA 250ml   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'tenaline la 250ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'tenaline la 250ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'tenaline la 250ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'tenaline la 250ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} tenaline la 250ml rows to 'L5_Vet'")
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
    'Still unmatched after tenaline la 250ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 tenaline la 250ml MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"tenaline la 250ml rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Tenaline LA 250ml   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'tenaline la 250ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'tenaline la 250ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'tenaline la 250ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'tenaline la 250ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} tenaline la 250ml rows to 'L5_Vet'")
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
    'Still unmatched after tenaline la 250ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 tenaline la 250ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"tenaline la 250ml rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Reversal 10mg 50ml   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'reversal 10mg 50ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'reversal 10mg 50ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'reversal 10mg 50ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'reversal 10mg 50ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} reversal 10mg 50ml rows to 'L5_Vet'")
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
    'Still unmatched after reversal 10mg 50ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 reversal 10mg 50ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"reversal 10mg 50ml rows flagged: {mask_excavator_full.sum():,}")
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

# =========================================================================================================================
# Multimin 500ml   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'multimin 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'multimin 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'multimin 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'multimin 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} multimin 500ml rows to 'L5_Vet'")
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
    'Still unmatched after multimin 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 multimin 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"multimin 500ml rows flagged: {mask_excavator_full.sum():,}")
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