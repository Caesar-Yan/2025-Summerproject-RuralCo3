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
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# package charge            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'package charge'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'package charge',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'package charge':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'package charge') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L4_no_discount_freight'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} package charge rows to 'L4_no_discount_freight'")
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
    'Still unmatched after package charge update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 package charge  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"package charge rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# w  consumable                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'w  consumable'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'w  consumable',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'w  consumable':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'w  consumable') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L4_no_discount_freight'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} w  consumable rows to 'L4_no_discount_freight'")
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
    'Still unmatched after w  consumable update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 w  consumable  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"w  consumable rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# ketomax 15% 250ml                                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ketomax 15% 250ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ketomax 15% 250ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ketomax 15% 250ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ketomax 15% 250ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ketomax 15% 250ml rows to 'L5_Vet'")
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
    'Still unmatched after ketomax 15% 250ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ketomax 15% 250ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ketomax 15% 250ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# vetserve milbemax mailout - 25kg tablet                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'vetserve milbemax mailout - 25kg tablet'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'vetserve milbemax mailout - 25kg tablet',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vetserve milbemax mailout - 25kg tablet':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'vetserve milbemax mailout - 25kg tablet') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} vetserve milbemax mailout - 25kg tablet rows to 'L5_Vet'")
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
    'Still unmatched after vetserve milbemax mailout - 25kg tablet update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 vetserve milbemax mailout - 25kg tablet  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"vetserve milbemax mailout - 25kg tablet rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# engemycin 250ml                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'engemycin 250ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'engemycin 250ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'engemycin 250ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'engemycin 250ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} engemycin 250ml rows to 'L5_Vet'")
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
    'Still unmatched after engemycin 250ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 engemycin 250ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"engemycin 250ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# covexin 10 500ml                             ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'covexin 10 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'covexin 10 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'covexin 10 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'covexin 10 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} covexin 10 500ml rows to 'L5_Vet'")
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
    'Still unmatched after covexin 10 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 covexin 10 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"covexin 10 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# zolvix plus 5l                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'zolvix plus 5l'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'zolvix plus 5l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'zolvix plus 5l':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'zolvix plus 5l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} zolvix plus 5l rows to 'L5_Vet'")
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
    'Still unmatched after zolvix plus 5l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 zolvix plus 5l  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"zolvix plus 5l rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# royal canin veterinary hypoallergenic do                               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'royal canin veterinary hypoallergenic do'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'royal canin veterinary hypoallergenic do',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'royal canin veterinary hypoallergenic do':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'royal canin veterinary hypoallergenic do') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} royal canin veterinary hypoallergenic do rows to 'L5_Vet'")
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
    'Still unmatched after royal canin veterinary hypoallergenic do update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 royal canin veterinary hypoallergenic do  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"royal canin veterinary hypoallergenic do rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# ovurelin  - per 1ml dose               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ovurelin  - per 1ml dose'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ovurelin  - per 1ml dose',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ovurelin  - per 1ml dose':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ovurelin  - per 1ml dose') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ovurelin  - per 1ml dose rows to 'L5_Vet'")
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
    'Still unmatched after ovurelin  - per 1ml dose update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ovurelin  - per 1ml dose  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ovurelin  - per 1ml dose rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# filter sleeve 230 x 850               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'filter sleeve 230 x 850'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'filter sleeve 230 x 850',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'filter sleeve 230 x 850':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'filter sleeve 230 x 850') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} filter sleeve 230 x 850 rows to 'L5_Vet'")
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
    'Still unmatched after filter sleeve 230 x 850 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 filter sleeve 230 x 850  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"filter sleeve 230 x 850 rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : iv catheter placement            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : iv catheter placement'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : iv catheter placement',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : iv catheter placement':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : iv catheter placement') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : iv catheter placement rows to 'L5_Vet'")
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
    'Still unmatched after ca : iv catheter placement update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : iv catheter placement  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : iv catheter placement rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# exlab : la courier & handling            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'exlab : la courier & handling'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'exlab : la courier & handling',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'exlab : la courier & handling':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'exlab : la courier & handling') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} exlab : la courier & handling rows to 'L5_Vet'")
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
    'Still unmatched after exlab : la courier & handling update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 exlab : la courier & handling  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"exlab : la courier & handling rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# rimadyl injection 20ml     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'rimadyl injection 20ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'rimadyl injection 20ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'rimadyl injection 20ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'rimadyl injection 20ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} rimadyl injection 20ml rows to 'L5_Vet'")
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
    'Still unmatched after rimadyl injection 20ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 rimadyl injection 20ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"rimadyl injection 20ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# ovuprost - per 2ml dose          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ovuprost - per 2ml dose'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ovuprost - per 2ml dose',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ovuprost - per 2ml dose':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ovuprost - per 2ml dose') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ovuprost - per 2ml dose rows to 'L5_Vet'")
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
    'Still unmatched after ovuprost - per 2ml dose update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ovuprost - per 2ml dose  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ovuprost - per 2ml dose rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : professional time : vet : per minut     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : professional time : vet : per minut'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : professional time : vet : per minut',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : professional time : vet : per minut':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : professional time : vet : per minut') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : professional time : vet : per minut rows to 'L5_Vet'")
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
    'Still unmatched after ca : professional time : vet : per minut update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : professional time : vet : per minut  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : professional time : vet : per minut rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# milpro dog tablets (over 5kg)  - per tab   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'milpro dog tablets (over 5kg)  - per tab'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'milpro dog tablets (over 5kg)  - per tab',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'milpro dog tablets (over 5kg)  - per tab':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'milpro dog tablets (over 5kg)  - per tab') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} milpro dog tablets (over 5kg)  - per tab rows to 'L5_Vet'")
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
    'Still unmatched after milpro dog tablets (over 5kg)  - per tab update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 milpro dog tablets (over 5kg)  - per tab  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"milpro dog tablets (over 5kg)  - per tab rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# drontal allwormer for dogs (up to 10kg t    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'drontal allwormer for dogs (up to 10kg t'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'drontal allwormer for dogs (up to 10kg t',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'drontal allwormer for dogs (up to 10kg t':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'drontal allwormer for dogs (up to 10kg t') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} drontal allwormer for dogs (up to 10kg t rows to 'L5_Vet'")
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
    'Still unmatched after drontal allwormer for dogs (up to 10kg t update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 drontal allwormer for dogs (up to 10kg t  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"drontal allwormer for dogs (up to 10kg t rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# groom : nail clip : standard        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'groom : nail clip : standard'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'groom : nail clip : standard',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'groom : nail clip : standard':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'groom : nail clip : standard') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} groom : nail clip : standard rows to 'L5_Vet'")
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
    'Still unmatched after groom : nail clip : standard update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 groom : nail clip : standard  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"groom : nail clip : standard rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# duomax 4000 b12 + sel 1l       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'duomax 4000 b12 + sel 1l'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'duomax 4000 b12 + sel 1l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'duomax 4000 b12 + sel 1l':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'duomax 4000 b12 + sel 1l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} duomax 4000 b12 + sel 1l rows to 'L5_Vet'")
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
    'Still unmatched after duomax 4000 b12 + sel 1l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 duomax 4000 b12 + sel 1l  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"duomax 4000 b12 + sel 1l rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : consultation : puppy/restart vaccin     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : consultation : puppy/restart vaccin'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : consultation : puppy/restart vaccin',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : consultation : puppy/restart vaccin':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : consultation : puppy/restart vaccin') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : consultation : puppy/restart vaccin rows to 'L5_Vet'")
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
    'Still unmatched after ca : consultation : puppy/restart vaccin update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : consultation : puppy/restart vaccin  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : consultation : puppy/restart vaccin rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - scan yes/no (>42 days)              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - scan yes/no (>42 days)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - scan yes/no (>42 days)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - scan yes/no (>42 days)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - scan yes/no (>42 days)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - scan yes/no (>42 days) rows to 'L5_Vet'")
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
    'Still unmatched after la - scan yes/no (>42 days) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - scan yes/no (>42 days)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - scan yes/no (>42 days) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - calving                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - calving'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - calving',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - calving':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - calving') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - calving rows to 'L5_Vet'")
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
    'Still unmatched after la - calving update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - calving  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - calving rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - repro vet dirty cow/metricheck per     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - repro vet dirty cow/metricheck per'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - repro vet dirty cow/metricheck per',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - repro vet dirty cow/metricheck per':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - repro vet dirty cow/metricheck per') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - repro vet dirty cow/metricheck per rows to 'L5_Vet'")
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
    'Still unmatched after la - repro vet dirty cow/metricheck per update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - repro vet dirty cow/metricheck per  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - repro vet dirty cow/metricheck per rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : radiography : initial (inc upto 2 v   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : radiography : initial (inc upto 2 v'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : radiography : initial (inc upto 2 v',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : radiography : initial (inc upto 2 v':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : radiography : initial (inc upto 2 v') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : radiography : initial (inc upto 2 v rows to 'L5_Vet'")
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
    'Still unmatched after ca : radiography : initial (inc upto 2 v update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : radiography : initial (inc upto 2 v  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : radiography : initial (inc upto 2 v rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - consum calving (per calving)       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - consum calving (per calving)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - consum calving (per calving)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum calving (per calving)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - consum calving (per calving)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - consum calving (per calving) rows to 'L5_Vet'")
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
    'Still unmatched after la - consum calving (per calving) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - consum calving (per calving)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - consum calving (per calving) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# covexin 10 100ml          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'covexin 10 100ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'covexin 10 100ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'covexin 10 100ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'covexin 10 100ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} covexin 10 100ml rows to 'L5_Vet'")
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
    'Still unmatched after covexin 10 100ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 covexin 10 100ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"covexin 10 100ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# gasket 13.8x18.8x4.8    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'gasket 13.8x18.8x4.8'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'gasket 13.8x18.8x4.8',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'gasket 13.8x18.8x4.8':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'gasket 13.8x18.8x4.8') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} gasket 13.8x18.8x4.8 rows to 'L7_mechanic'")
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
    'Still unmatched after gasket 13.8x18.8x4.8 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 gasket 13.8x18.8x4.8  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"gasket 13.8x18.8x4.8 rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# w oil & filter disposal           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'w oil & filter disposal'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'w oil & filter disposal',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'w oil & filter disposal':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'w oil & filter disposal') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} w oil & filter disposal rows to 'L7_mechanic'")
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
    'Still unmatched after w oil & filter disposal update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 w oil & filter disposal  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"w oil & filter disposal rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# lab technician (nett)        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'lab technician (nett)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'lab technician (nett)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'lab technician (nett)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'lab technician (nett)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} lab technician (nett) rows to 'L7_mechanic'")
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
    'Still unmatched after lab technician (nett) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 lab technician (nett)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"lab technician (nett) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 30 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(30))
print(f"{'='*70}\n")

# =========================================================================================================================
# oil filter         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'oil filter'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'oil filter',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'oil filter':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'oil filter') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} oil filter rows to 'L7_mechanic'")
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
    'Still unmatched after oil filter update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 oil filter  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"oil filter rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# fire & emergency levy              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'fire & emergency levy'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'fire & emergency levy',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'fire & emergency levy':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'fire & emergency levy') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L4_no_discount_freight'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} fire & emergency levy rows to 'L4_no_discount_freight'")
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
    'Still unmatched after fire & emergency levy update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 fire & emergency levy  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"fire & emergency levy rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# natural disaster charge                   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'natural disaster charge'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'natural disaster charge',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'natural disaster charge':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'natural disaster charge') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L4_no_discount_freight'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} natural disaster charge rows to 'L4_no_discount_freight'")
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
    'Still unmatched after natural disaster charge update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 natural disaster charge  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"natural disaster charge rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# q.delivery charge area 1 < 5 tonnes       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'q.delivery charge area 1 < 5 tonnes'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'q.delivery charge area 1 < 5 tonnes',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q.delivery charge area 1 < 5 tonnes':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'q.delivery charge area 1 < 5 tonnes') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L4_no_discount_freight'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} q.delivery charge area 1 < 5 tonnes rows to 'L4_no_discount_freight'")
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
    'Still unmatched after q.delivery charge area 1 < 5 tonnes update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 q.delivery charge area 1 < 5 tonnes  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q.delivery charge area 1 < 5 tonnes rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# q-scoop-zland- bark nuggets                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'q-scoop-zland- bark nuggets'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'q-scoop-zland- bark nuggets',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-scoop-zland- bark nuggets':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'q-scoop-zland- bark nuggets') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L4_no_discount_freight'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} q-scoop-zland- bark nuggets rows to 'L4_no_discount_freight'")
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
    'Still unmatched after q-scoop-zland- bark nuggets update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L4 q-scoop-zland- bark nuggets  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q-scoop-zland- bark nuggets rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# bivatop 250ml    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'bivatop 250ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'bivatop 250ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'bivatop 250ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'bivatop 250ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} bivatop 250ml rows to 'L5_Vet'")
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
    'Still unmatched after bivatop 250ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 bivatop 250ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"bivatop 250ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# buprelieve injection 10ml    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'buprelieve injection 10ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'buprelieve injection 10ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'buprelieve injection 10ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'buprelieve injection 10ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} buprelieve injection 10ml rows to 'L5_Vet'")
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
    'Still unmatched after buprelieve injection 10ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 buprelieve injection 10ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"buprelieve injection 10ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : consultation : repeat injection   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : consultation : repeat injection'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : consultation : repeat injection',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : consultation : repeat injection':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : consultation : repeat injection') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : consultation : repeat injection rows to 'L5_Vet'")
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
    'Still unmatched after ca : consultation : repeat injection update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : consultation : repeat injection  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : consultation : repeat injection rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : surgical instrument pack : each      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : surgical instrument pack : each'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : surgical instrument pack : each',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : surgical instrument pack : each':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : surgical instrument pack : each') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : surgical instrument pack : each rows to 'L5_Vet'")
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
    'Still unmatched after ca : surgical instrument pack : each update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : surgical instrument pack : each  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : surgical instrument pack : each rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : treatment consumables 1 (upto 5mins      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : treatment consumables 1 (upto 5mins'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : treatment consumables 1 (upto 5mins',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : treatment consumables 1 (upto 5mins':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : treatment consumables 1 (upto 5mins') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : treatment consumables 1 (upto 5mins rows to 'L5_Vet'")
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
    'Still unmatched after ca : treatment consumables 1 (upto 5mins update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : treatment consumables 1 (upto 5mins  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : treatment consumables 1 (upto 5mins rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# calcium hypochlorite (hth) - 25 kg      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'calcium hypochlorite (hth) - 25 kg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'calcium hypochlorite (hth) - 25 kg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'calcium hypochlorite (hth) - 25 kg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'calcium hypochlorite (hth) - 25 kg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} calcium hypochlorite (hth) - 25 kg rows to 'L5_Vet'")
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
    'Still unmatched after calcium hypochlorite (hth) - 25 kg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 calcium hypochlorite (hth) - 25 kg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"calcium hypochlorite (hth) - 25 kg rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# cidr-b - per cidr                          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'cidr-b - per cidr'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'cidr-b - per cidr',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'cidr-b - per cidr':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'cidr-b - per cidr') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} cidr-b - per cidr rows to 'L5_Vet'")
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
    'Still unmatched after cidr-b - per cidr update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 cidr-b - per cidr  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"cidr-b - per cidr rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# clavulox injection rtu per ml            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'clavulox injection rtu per ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'clavulox injection rtu per ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'clavulox injection rtu per ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'clavulox injection rtu per ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} clavulox injection rtu per ml rows to 'L5_Vet'")
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
    'Still unmatched after clavulox injection rtu per ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 clavulox injection rtu per ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"clavulox injection rtu per ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# clavulox tablets 250mg       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'clavulox tablets 250mg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'clavulox tablets 250mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'clavulox tablets 250mg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'clavulox tablets 250mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} clavulox tablets 250mg rows to 'L5_Vet'")
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
    'Still unmatched after clavulox tablets 250mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 clavulox tablets 250mg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"clavulox tablets 250mg rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# coppermax copper capsules 20g (elanco)   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'coppermax copper capsules 20g (elanco)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'coppermax copper capsules 20g (elanco)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'coppermax copper capsules 20g (elanco)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'coppermax copper capsules 20g (elanco)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} coppermax copper capsules 20g (elanco) rows to 'L5_Vet'")
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
    'Still unmatched after coppermax copper capsules 20g (elanco) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 coppermax copper capsules 20g (elanco)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"coppermax copper capsules 20g (elanco) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# coppermax copper capsules 30g (elanco)   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'coppermax copper capsules 30g (elanco)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'coppermax copper capsules 30g (elanco)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'coppermax copper capsules 30g (elanco)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'coppermax copper capsules 30g (elanco)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} coppermax copper capsules 30g (elanco) rows to 'L5_Vet'")
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
    'Still unmatched after coppermax copper capsules 30g (elanco) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 coppermax copper capsules 30g (elanco)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"coppermax copper capsules 30g (elanco) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# covexin 10 (per 2ml dose)         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'covexin 10 (per 2ml dose)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'covexin 10 (per 2ml dose)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'covexin 10 (per 2ml dose)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'covexin 10 (per 2ml dose)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} covexin 10 (per 2ml dose) rows to 'L5_Vet'")
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
    'Still unmatched after covexin 10 (per 2ml dose) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 covexin 10 (per 2ml dose)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"covexin 10 (per 2ml dose) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# duomax 4000 b12 + sel 500ml          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'duomax 4000 b12 + sel 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'duomax 4000 b12 + sel 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'duomax 4000 b12 + sel 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'duomax 4000 b12 + sel 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} duomax 4000 b12 + sel 500ml rows to 'L5_Vet'")
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
    'Still unmatched after duomax 4000 b12 + sel 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 duomax 4000 b12 + sel 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"duomax 4000 b12 + sel 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# exlab - la nutritional chemistry b12 (co    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'exlab - la nutritional chemistry b12 (co'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'exlab - la nutritional chemistry b12 (co',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'exlab - la nutritional chemistry b12 (co':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'exlab - la nutritional chemistry b12 (co') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} exlab - la nutritional chemistry b12 (co rows to 'L5_Vet'")
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
    'Still unmatched after exlab - la nutritional chemistry b12 (co update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 exlab - la nutritional chemistry b12 (co  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"exlab - la nutritional chemistry b12 (co rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# exlab - la nutritional chemistry copper    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'exlab - la nutritional chemistry copper'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'exlab - la nutritional chemistry copper',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'exlab - la nutritional chemistry copper':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'exlab - la nutritional chemistry copper') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} exlab - la nutritional chemistry copper rows to 'L5_Vet'")
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
    'Still unmatched after exlab - la nutritional chemistry copper update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 exlab - la nutritional chemistry copper  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"exlab - la nutritional chemistry copper rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# exlab - la nutritional chemistry seleniu   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'exlab - la nutritional chemistry seleniu'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'exlab - la nutritional chemistry seleniu',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'exlab - la nutritional chemistry seleniu':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'exlab - la nutritional chemistry seleniu') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} exlab - la nutritional chemistry seleniu rows to 'L5_Vet'")
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
    'Still unmatched after exlab - la nutritional chemistry seleniu update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 exlab - la nutritional chemistry seleniu  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"exlab - la nutritional chemistry seleniu rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# glucalphos 500ml (bomaflex)                ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'glucalphos 500ml (bomaflex)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'glucalphos 500ml (bomaflex)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'glucalphos 500ml (bomaflex)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'glucalphos 500ml (bomaflex)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} glucalphos 500ml (bomaflex) rows to 'L5_Vet'")
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
    'Still unmatched after glucalphos 500ml (bomaflex) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 glucalphos 500ml (bomaflex)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"glucalphos 500ml (bomaflex) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# inlab - la parasight faecal egg count -         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'inlab - la parasight faecal egg count -'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'inlab - la parasight faecal egg count -',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'inlab - la parasight faecal egg count -':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'inlab - la parasight faecal egg count -') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} inlab - la parasight faecal egg count - rows to 'L5_Vet'")
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
    'Still unmatched after inlab - la parasight faecal egg count - update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 inlab - la parasight faecal egg count -  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"inlab - la parasight faecal egg count - rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# inlab : urine : dipstick & specific grav   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'inlab : urine : dipstick & specific grav'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'inlab : urine : dipstick & specific grav',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'inlab : urine : dipstick & specific grav':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'inlab : urine : dipstick & specific grav') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} inlab : urine : dipstick & specific grav rows to 'L5_Vet'")
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
    'Still unmatched after inlab : urine : dipstick & specific grav update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 inlab : urine : dipstick & specific grav  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"inlab : urine : dipstick & specific grav rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - epidural                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - epidural'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - epidural',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - epidural':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - epidural') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - epidural rows to 'L5_Vet'")
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
    'Still unmatched after la - epidural update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - epidural  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - epidural rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - local anaesthesia fee                   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - local anaesthesia fee'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - local anaesthesia fee',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - local anaesthesia fee':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - local anaesthesia fee') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - local anaesthesia fee rows to 'L5_Vet'")
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
    'Still unmatched after la - local anaesthesia fee update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - local anaesthesia fee  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - local anaesthesia fee rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - velveting                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - velveting'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - velveting',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - velveting':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - velveting') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - velveting rows to 'L5_Vet'")
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
    'Still unmatched after la - velveting update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - velveting  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - velveting rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# mamyzin inj 5gm                               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'mamyzin inj 5gm'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'mamyzin inj 5gm',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'mamyzin inj 5gm':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'mamyzin inj 5gm') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} mamyzin inj 5gm rows to 'L5_Vet'")
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
    'Still unmatched after mamyzin inj 5gm update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 mamyzin inj 5gm  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"mamyzin inj 5gm rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# mastatest p2 (10 tests)                               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'mastatest p2 (10 tests)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'mastatest p2 (10 tests)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'mastatest p2 (10 tests)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'mastatest p2 (10 tests)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} mastatest p2 (10 tests) rows to 'L5_Vet'")
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
    'Still unmatched after mastatest p2 (10 tests) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 mastatest p2 (10 tests)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"mastatest p2 (10 tests) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# orbenin eye ointment               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'orbenin eye ointment'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'orbenin eye ointment',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'orbenin eye ointment':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'orbenin eye ointment') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} orbenin eye ointment rows to 'L5_Vet'")
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
    'Still unmatched after orbenin eye ointment update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 orbenin eye ointment  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"orbenin eye ointment rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# oxytocin 100ml            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'oxytocin 100ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'oxytocin 100ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'oxytocin 100ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'oxytocin 100ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} oxytocin 100ml rows to 'L5_Vet'")
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
    'Still unmatched after oxytocin 100ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 oxytocin 100ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"oxytocin 100ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# penclox 1200 mc (pack of 24)              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'penclox 1200 mc (pack of 24)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'penclox 1200 mc (pack of 24)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'penclox 1200 mc (pack of 24)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'penclox 1200 mc (pack of 24)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} penclox 1200 mc (pack of 24) rows to 'L5_Vet'")
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
    'Still unmatched after penclox 1200 mc (pack of 24) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 penclox 1200 mc (pack of 24)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"penclox 1200 mc (pack of 24) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# pentobarb 300 250ml (250)                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'pentobarb 300 250ml (250)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'pentobarb 300 250ml (250)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'pentobarb 300 250ml (250)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'pentobarb 300 250ml (250)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} pentobarb 300 250ml (250) rows to 'L5_Vet'")
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
    'Still unmatched after pentobarb 300 250ml (250) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 pentobarb 300 250ml (250)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"pentobarb 300 250ml (250) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# rimadyl chewable tablets 75mg          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'rimadyl chewable tablets 75mg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'rimadyl chewable tablets 75mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'rimadyl chewable tablets 75mg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'rimadyl chewable tablets 75mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} rimadyl chewable tablets 75mg rows to 'L5_Vet'")
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
    'Still unmatched after rimadyl chewable tablets 75mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 rimadyl chewable tablets 75mg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"rimadyl chewable tablets 75mg rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# selekt fresh cow                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'selekt fresh cow'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'selekt fresh cow',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'selekt fresh cow':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'selekt fresh cow') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} selekt fresh cow rows to 'L5_Vet'")
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
    'Still unmatched after selekt fresh cow update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 selekt fresh cow  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"selekt fresh cow rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# selovin la injection 500ml               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'selovin la injection 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'selovin la injection 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'selovin la injection 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'selovin la injection 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} selovin la injection 500ml rows to 'L5_Vet'")
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
    'Still unmatched after selovin la injection 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 selovin la injection 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"selovin la injection 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# teatseal                                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'teatseal'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'teatseal',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'teatseal':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'teatseal') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} teatseal rows to 'L5_Vet'")
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
    'Still unmatched after teatseal update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 teatseal  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"teatseal rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# turbo injection 500ml                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'turbo injection 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'turbo injection 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'turbo injection 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'turbo injection 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} turbo injection 500ml rows to 'L5_Vet'")
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
    'Still unmatched after turbo injection 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 turbo injection 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"turbo injection 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ultravac 5in1 500ml                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ultravac 5in1 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ultravac 5in1 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ultravac 5in1 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ultravac 5in1 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ultravac 5in1 500ml rows to 'L5_Vet'")
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
    'Still unmatched after ultravac 5in1 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ultravac 5in1 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ultravac 5in1 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ultravac 7in1 - 500ml                                   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ultravac 7in1 - 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ultravac 7in1 - 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ultravac 7in1 - 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ultravac 7in1 - 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ultravac 7in1 - 500ml rows to 'L5_Vet'")
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
    'Still unmatched after ultravac 7in1 - 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ultravac 7in1 - 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ultravac 7in1 - 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# vaccine : felocell 3                             ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'vaccine : felocell 3'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'vaccine : felocell 3',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vaccine : felocell 3':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'vaccine : felocell 3') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} vaccine : felocell 3 rows to 'L5_Vet'")
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
    'Still unmatched after vaccine : felocell 3 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 vaccine : felocell 3  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"vaccine : felocell 3 rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# veltrak tags                                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'veltrak tags'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'veltrak tags',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'veltrak tags':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'veltrak tags') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} veltrak tags rows to 'L5_Vet'")
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
    'Still unmatched after veltrak tags update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 veltrak tags  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"veltrak tags rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# vibrostrep 100ml                                ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'vibrostrep 100ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'vibrostrep 100ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vibrostrep 100ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'vibrostrep 100ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} vibrostrep 100ml rows to 'L5_Vet'")
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
    'Still unmatched after vibrostrep 100ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 vibrostrep 100ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"vibrostrep 100ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# xylazine 10%                                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'xylazine 10%'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'xylazine 10%',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'xylazine 10%':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'xylazine 10%') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} xylazine 10% rows to 'L5_Vet'")
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
    'Still unmatched after xylazine 10% update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 xylazine 10%  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"xylazine 10% rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# zoletil 100                                             ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'zoletil 100'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'zoletil 100',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'zoletil 100':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'zoletil 100') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} zoletil 100 rows to 'L5_Vet'")
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
    'Still unmatched after zoletil 100 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 zoletil 100  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"zoletil 100 rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# blade wear                                        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'blade wear'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'blade wear',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'blade wear':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'blade wear') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} blade wear rows to 'L7_mechanic'")
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
    'Still unmatched after blade wear update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 blade wear  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"blade wear rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# break roller tester                     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'break roller tester'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'break roller tester',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'break roller tester':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'break roller tester') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} break roller tester rows to 'L7_mechanic'")
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
    'Still unmatched after break roller tester update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 break roller tester  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"break roller tester rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# filter oil  25c00                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'filter oil  25c00'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'filter oil  25c00',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'filter oil  25c00':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'filter oil  25c00') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} filter oil  25c00 rows to 'L7_mechanic'")
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
    'Still unmatched after filter oil  25c00 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 filter oil  25c00  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"filter oil  25c00 rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# spark plug ngk dr8ea dont a/ft              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'spark plug ngk dr8ea dont a/ft'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'spark plug ngk dr8ea dont a/ft',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'spark plug ngk dr8ea dont a/ft':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'spark plug ngk dr8ea dont a/ft') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} spark plug ngk dr8ea dont a/ft rows to 'L7_mechanic'")
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
    'Still unmatched after spark plug ngk dr8ea dont a/ft update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 spark plug ngk dr8ea dont a/ft  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"spark plug ngk dr8ea dont a/ft rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# w  vehicle cof truck 4 axle                   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'w  vehicle cof truck 4 axle'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'w  vehicle cof truck 4 axle',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'w  vehicle cof truck 4 axle':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'w  vehicle cof truck 4 axle') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} w  vehicle cof truck 4 axle rows to 'L7_mechanic'")
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
    'Still unmatched after w  vehicle cof truck 4 axle update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 w  vehicle cof truck 4 axle  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"w  vehicle cof truck 4 axle rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# q-basecourse- ap20- dobson st                         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'q-basecourse- ap20- dobson st'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'q-basecourse- ap20- dobson st',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-basecourse- ap20- dobson st':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'q-basecourse- ap20- dobson st') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L9_infrastructure_consumables'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} q-basecourse- ap20- dobson st rows to 'L9_infrastructure_consumables'")
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
    'Still unmatched after q-basecourse- ap20- dobson st update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L9 q-basecourse- ap20- dobson st  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q-basecourse- ap20- dobson st rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# bandage : per roll : cohesive/coflex/fun           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'bandage : per roll : cohesive/coflex/fun'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'bandage : per roll : cohesive/coflex/fun',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'bandage : per roll : cohesive/coflex/fun':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'bandage : per roll : cohesive/coflex/fun') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} bandage : per roll : cohesive/coflex/fun rows to 'L5_Vet'")
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
    'Still unmatched after bandage : per roll : cohesive/coflex/fun update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 bandage : per roll : cohesive/coflex/fun  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"bandage : per roll : cohesive/coflex/fun rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# blackhawk original adult dog - lamb & ri       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'blackhawk original adult dog - lamb & ri'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'blackhawk original adult dog - lamb & ri',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'blackhawk original adult dog - lamb & ri':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'blackhawk original adult dog - lamb & ri') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} blackhawk original adult dog - lamb & ri rows to 'L5_Vet'")
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
    'Still unmatched after blackhawk original adult dog - lamb & ri update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 blackhawk original adult dog - lamb & ri  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"blackhawk original adult dog - lamb & ri rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# boss pour-on 5l                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'boss pour-on 5l'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'boss pour-on 5l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'boss pour-on 5l':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'boss pour-on 5l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} boss pour-on 5l rows to 'L5_Vet'")
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
    'Still unmatched after boss pour-on 5l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 boss pour-on 5l  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"boss pour-on 5l rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : anaesthesia : sedation : euthanasia                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : anaesthesia : sedation : euthanasia'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : anaesthesia : sedation : euthanasia',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : anaesthesia : sedation : euthanasia':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : anaesthesia : sedation : euthanasia') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : anaesthesia : sedation : euthanasia rows to 'L5_Vet'")
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
    'Still unmatched after ca : anaesthesia : sedation : euthanasia update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : anaesthesia : sedation : euthanasia  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : anaesthesia : sedation : euthanasia rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : consultation : euthanasia                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : consultation : euthanasia'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : consultation : euthanasia',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : consultation : euthanasia':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : consultation : euthanasia') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : consultation : euthanasia rows to 'L5_Vet'")
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
    'Still unmatched after ca : consultation : euthanasia update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : consultation : euthanasia  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : consultation : euthanasia rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : consultation : kitten/restart vacci         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : consultation : kitten/restart vacci'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : consultation : kitten/restart vacci',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : consultation : kitten/restart vacci':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : consultation : kitten/restart vacci') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : consultation : kitten/restart vacci rows to 'L5_Vet'")
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
    'Still unmatched after ca : consultation : kitten/restart vacci update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : consultation : kitten/restart vacci  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : consultation : kitten/restart vacci rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : consultation : repeat injection : v       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : consultation : repeat injection : v'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : consultation : repeat injection : v',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : consultation : repeat injection : v':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : consultation : repeat injection : v') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : consultation : repeat injection : v rows to 'L5_Vet'")
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
    'Still unmatched after ca : consultation : repeat injection : v update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : consultation : repeat injection : v  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : consultation : repeat injection : v rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : patient preparation : non sterile :       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : patient preparation : non sterile :'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : patient preparation : non sterile :',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : patient preparation : non sterile :':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : patient preparation : non sterile :') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : patient preparation : non sterile : rows to 'L5_Vet'")
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
    'Still unmatched after ca : patient preparation : non sterile : update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : patient preparation : non sterile :  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : patient preparation : non sterile : rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : radiography : additional images (ea        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : radiography : additional images (ea'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : radiography : additional images (ea',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : radiography : additional images (ea':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : radiography : additional images (ea') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : radiography : additional images (ea rows to 'L5_Vet'")
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
    'Still unmatched after ca : radiography : additional images (ea update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : radiography : additional images (ea  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : radiography : additional images (ea rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# campyvax 4 500ml                                ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'campyvax 4 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'campyvax 4 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'campyvax 4 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'campyvax 4 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} campyvax 4 500ml rows to 'L5_Vet'")
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
    'Still unmatched after campyvax 4 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 campyvax 4 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"campyvax 4 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# catosal 100ml                                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'catosal 100ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'catosal 100ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'catosal 100ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'catosal 100ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} catosal 100ml rows to 'L5_Vet'")
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
    'Still unmatched after catosal 100ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 catosal 100ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"catosal 100ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# clavulox tablets 500mg                                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'clavulox tablets 500mg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'clavulox tablets 500mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'clavulox tablets 500mg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'clavulox tablets 500mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} clavulox tablets 500mg rows to 'L5_Vet'")
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
    'Still unmatched after clavulox tablets 500mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 clavulox tablets 500mg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"clavulox tablets 500mg rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# depodine injection 500ml                             ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'depodine injection 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'depodine injection 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'depodine injection 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'depodine injection 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} depodine injection 500ml rows to 'L5_Vet'")
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
    'Still unmatched after depodine injection 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 depodine injection 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"depodine injection 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# duoject b                                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'duoject b'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'duoject b',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'duoject b':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'duoject b') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} duoject b rows to 'L5_Vet'")
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
    'Still unmatched after duoject b update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 duoject b  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"duoject b rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# eclipse pour-on 5.5l                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'eclipse pour-on 5.5l'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'eclipse pour-on 5.5l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'eclipse pour-on 5.5l':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'eclipse pour-on 5.5l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} eclipse pour-on 5.5l rows to 'L5_Vet'")
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
    'Still unmatched after eclipse pour-on 5.5l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 eclipse pour-on 5.5l  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"eclipse pour-on 5.5l rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# eclipse pour-on 5l                                   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'eclipse pour-on 5l'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'eclipse pour-on 5l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'eclipse pour-on 5l':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'eclipse pour-on 5l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} eclipse pour-on 5l rows to 'L5_Vet'")
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
    'Still unmatched after eclipse pour-on 5l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 eclipse pour-on 5l  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"eclipse pour-on 5l rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# exlab - la nutritional chemistry inorgan                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'exlab - la nutritional chemistry inorgan'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'exlab - la nutritional chemistry inorgan',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'exlab - la nutritional chemistry inorgan':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'exlab - la nutritional chemistry inorgan') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} exlab - la nutritional chemistry inorgan rows to 'L5_Vet'")
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
    'Still unmatched after exlab - la nutritional chemistry inorgan update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 exlab - la nutritional chemistry inorgan  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"exlab - la nutritional chemistry inorgan rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# fluids : per bag : hartmanns : 1000mls                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'fluids : per bag : hartmanns : 1000mls'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'fluids : per bag : hartmanns : 1000mls',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'fluids : per bag : hartmanns : 1000mls':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'fluids : per bag : hartmanns : 1000mls') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} fluids : per bag : hartmanns : 1000mls rows to 'L5_Vet'")
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
    'Still unmatched after fluids : per bag : hartmanns : 1000mls update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 fluids : per bag : hartmanns : 1000mls  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"fluids : per bag : hartmanns : 1000mls rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# gun injector 5ml stv simcro (ea=1)                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'gun injector 5ml stv simcro (ea=1)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'gun injector 5ml stv simcro (ea=1)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'gun injector 5ml stv simcro (ea=1)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'gun injector 5ml stv simcro (ea=1)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} gun injector 5ml stv simcro (ea=1) rows to 'L5_Vet'")
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
    'Still unmatched after gun injector 5ml stv simcro (ea=1) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 gun injector 5ml stv simcro (ea=1)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"gun injector 5ml stv simcro (ea=1) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# inlab : cytology : ear smear : initial                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'inlab : cytology : ear smear : initial'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'inlab : cytology : ear smear : initial',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'inlab : cytology : ear smear : initial':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'inlab : cytology : ear smear : initial') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} inlab : cytology : ear smear : initial rows to 'L5_Vet'")
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
    'Still unmatched after inlab : cytology : ear smear : initial update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 inlab : cytology : ear smear : initial  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"inlab : cytology : ear smear : initial rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - calf disbudding incl. pain relief (         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - calf disbudding incl. pain relief ('

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - calf disbudding incl. pain relief (',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - calf disbudding incl. pain relief (':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - calf disbudding incl. pain relief (') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - calf disbudding incl. pain relief ( rows to 'L5_Vet'")
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
    'Still unmatched after la - calf disbudding incl. pain relief ( update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - calf disbudding incl. pain relief (  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - calf disbudding incl. pain relief ( rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - consult rvm                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - consult rvm'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - consult rvm',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consult rvm':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - consult rvm') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - consult rvm rows to 'L5_Vet'")
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
    'Still unmatched after la - consult rvm update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - consult rvm  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - consult rvm rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - consum - needles box of 100 (all si               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - consum - needles box of 100 (all si'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - consum - needles box of 100 (all si',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum - needles box of 100 (all si':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - consum - needles box of 100 (all si') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - consum - needles box of 100 (all si rows to 'L5_Vet'")
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
    'Still unmatched after la - consum - needles box of 100 (all si update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - consum - needles box of 100 (all si  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - consum - needles box of 100 (all si rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - consum minor surgery                          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - consum minor surgery'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - consum minor surgery',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum minor surgery':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - consum minor surgery') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - consum minor surgery rows to 'L5_Vet'")
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
    'Still unmatched after la - consum minor surgery update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - consum minor surgery  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - consum minor surgery rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - discount rvm only                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - discount rvm only'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - discount rvm only',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - discount rvm only':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - discount rvm only') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - discount rvm only rows to 'L5_Vet'")
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
    'Still unmatched after la - discount rvm only update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - discount rvm only  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - discount rvm only rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# lepto 4-way vaccine (per 2ml dose)                   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'lepto 4-way vaccine (per 2ml dose)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'lepto 4-way vaccine (per 2ml dose)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'lepto 4-way vaccine (per 2ml dose)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'lepto 4-way vaccine (per 2ml dose)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} lepto 4-way vaccine (per 2ml dose) rows to 'L5_Vet'")
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
    'Still unmatched after lepto 4-way vaccine (per 2ml dose) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 lepto 4-way vaccine (per 2ml dose)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"lepto 4-way vaccine (per 2ml dose) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# mamyzin inj 10gm                                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'mamyzin inj 10gm'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'mamyzin inj 10gm',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'mamyzin inj 10gm':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'mamyzin inj 10gm') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} mamyzin inj 10gm rows to 'L5_Vet'")
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
    'Still unmatched after mamyzin inj 10gm update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 mamyzin inj 10gm  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"mamyzin inj 10gm rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# metacam 40 inj 100ml (ea=1ml)                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'metacam 40 inj 100ml (ea=1ml)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'metacam 40 inj 100ml (ea=1ml)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'metacam 40 inj 100ml (ea=1ml)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'metacam 40 inj 100ml (ea=1ml)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} metacam 40 inj 100ml (ea=1ml) rows to 'L5_Vet'")
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
    'Still unmatched after metacam 40 inj 100ml (ea=1ml) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 metacam 40 inj 100ml (ea=1ml)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"metacam 40 inj 100ml (ea=1ml) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# metacam inj cat/dog 5mg                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'metacam inj cat/dog 5mg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'metacam inj cat/dog 5mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'metacam inj cat/dog 5mg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'metacam inj cat/dog 5mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} metacam inj cat/dog 5mg rows to 'L5_Vet'")
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
    'Still unmatched after metacam inj cat/dog 5mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 metacam inj cat/dog 5mg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"metacam inj cat/dog 5mg rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# metacam oral cat 0.5mg 3ml                                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'metacam oral cat 0.5mg 3ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'metacam oral cat 0.5mg 3ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'metacam oral cat 0.5mg 3ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'metacam oral cat 0.5mg 3ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} metacam oral cat 0.5mg 3ml rows to 'L5_Vet'")
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
    'Still unmatched after metacam oral cat 0.5mg 3ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 metacam oral cat 0.5mg 3ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"metacam oral cat 0.5mg 3ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# multine 5 b12 + sel 500ml                                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'multine 5 b12 + sel 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'multine 5 b12 + sel 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'multine 5 b12 + sel 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'multine 5 b12 + sel 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} multine 5 b12 + sel 500ml rows to 'L5_Vet'")
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
    'Still unmatched after multine 5 b12 + sel 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 multine 5 b12 + sel 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"multine 5 b12 + sel 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# prolaject b12 2000 500ml                                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'prolaject b12 2000 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'prolaject b12 2000 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'prolaject b12 2000 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'prolaject b12 2000 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} prolaject b12 2000 500ml rows to 'L5_Vet'")
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
    'Still unmatched after prolaject b12 2000 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 prolaject b12 2000 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"prolaject b12 2000 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# propercillin 500ml                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'propercillin 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'propercillin 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'propercillin 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'propercillin 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} propercillin 500ml rows to 'L5_Vet'")
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
    'Still unmatched after propercillin 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 propercillin 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"propercillin 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# rimadyl chewable tablets 25mg                                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'rimadyl chewable tablets 25mg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'rimadyl chewable tablets 25mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'rimadyl chewable tablets 25mg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'rimadyl chewable tablets 25mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} rimadyl chewable tablets 25mg rows to 'L5_Vet'")
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
    'Still unmatched after rimadyl chewable tablets 25mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 rimadyl chewable tablets 25mg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"rimadyl chewable tablets 25mg rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# royal canin neutered adult small dog (dr                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'royal canin neutered adult small dog (dr'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'royal canin neutered adult small dog (dr',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'royal canin neutered adult small dog (dr':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'royal canin neutered adult small dog (dr') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} royal canin neutered adult small dog (dr rows to 'L5_Vet'")
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
    'Still unmatched after royal canin neutered adult small dog (dr update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 royal canin neutered adult small dog (dr  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"royal canin neutered adult small dog (dr rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# sample jar - milk 35ml                                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'sample jar - milk 35ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'sample jar - milk 35ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'sample jar - milk 35ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'sample jar - milk 35ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} sample jar - milk 35ml rows to 'L5_Vet'")
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
    'Still unmatched after sample jar - milk 35ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 sample jar - milk 35ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"sample jar - milk 35ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# ultravac bvd 250ml                                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ultravac bvd 250ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ultravac bvd 250ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ultravac bvd 250ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ultravac bvd 250ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ultravac bvd 250ml rows to 'L5_Vet'")
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
    'Still unmatched after ultravac bvd 250ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ultravac bvd 250ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ultravac bvd 250ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# vaccine : canigen kc                                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'vaccine : canigen kc'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'vaccine : canigen kc',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vaccine : canigen kc':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'vaccine : canigen kc') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} vaccine : canigen kc rows to 'L5_Vet'")
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
    'Still unmatched after vaccine : canigen kc update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 vaccine : canigen kc  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"vaccine : canigen kc rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# xylazine 5%                                                                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'xylazine 5%'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'xylazine 5%',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'xylazine 5%':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'xylazine 5%') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} xylazine 5% rows to 'L5_Vet'")
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
    'Still unmatched after xylazine 5% update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 xylazine 5%  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"xylazine 5% rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# castrol vecton 15w 40 ck-4/e9 bulk                                                     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'castrol vecton 15w 40 ck-4/e9 bulk'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'castrol vecton 15w 40 ck-4/e9 bulk',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'castrol vecton 15w 40 ck-4/e9 bulk':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'castrol vecton 15w 40 ck-4/e9 bulk') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} castrol vecton 15w 40 ck-4/e9 bulk rows to 'L7_mechanic'")
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
    'Still unmatched after castrol vecton 15w 40 ck-4/e9 bulk update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 castrol vecton 15w 40 ck-4/e9 bulk  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"castrol vecton 15w 40 ck-4/e9 bulk rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# filter fuel                                                        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'filter fuel'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'filter fuel',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'filter fuel':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'filter fuel') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} filter fuel rows to 'L7_mechanic'")
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
    'Still unmatched after filter fuel update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 filter fuel  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"filter fuel rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# filter sleeve 230 x 1100                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'filter sleeve 230 x 1100'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'filter sleeve 230 x 1100',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'filter sleeve 230 x 1100':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'filter sleeve 230 x 1100') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} filter sleeve 230 x 1100 rows to 'L7_mechanic'")
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
    'Still unmatched after filter sleeve 230 x 1100 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 filter sleeve 230 x 1100  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"filter sleeve 230 x 1100 rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# gearcase oil                                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'gearcase oil'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'gearcase oil',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'gearcase oil':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'gearcase oil') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} gearcase oil rows to 'L7_mechanic'")
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
    'Still unmatched after gearcase oil update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 gearcase oil  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"gearcase oil rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# glycerine bp - 20 ltr                                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'glycerine bp - 20 ltr'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'glycerine bp - 20 ltr',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'glycerine bp - 20 ltr':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'glycerine bp - 20 ltr') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} glycerine bp - 20 ltr rows to 'L7_mechanic'")
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
    'Still unmatched after glycerine bp - 20 ltr update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 glycerine bp - 20 ltr  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"glycerine bp - 20 ltr rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# sundries (nett)                                               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'sundries (nett)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'sundries (nett)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'sundries (nett)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'sundries (nett)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} sundries (nett) rows to 'L7_mechanic'")
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
    'Still unmatched after sundries (nett) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 sundries (nett)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"sundries (nett) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# w  vehicle cof truck 3 axle                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'w  vehicle cof truck 3 axle'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'w  vehicle cof truck 3 axle',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'w  vehicle cof truck 3 axle':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'w  vehicle cof truck 3 axle') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} w  vehicle cof truck 3 axle rows to 'L7_mechanic'")
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
    'Still unmatched after w  vehicle cof truck 3 axle update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 w  vehicle cof truck 3 axle  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"w  vehicle cof truck 3 axle rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# molasses feedgrade bulk (kg)                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'molasses feedgrade bulk (kg)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'molasses feedgrade bulk (kg)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'molasses feedgrade bulk (kg)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'molasses feedgrade bulk (kg)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_cattle_feed'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} molasses feedgrade bulk (kg) rows to 'L7_cattle_feed'")
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
    'Still unmatched after molasses feedgrade bulk (kg) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 molasses feedgrade bulk (kg)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"molasses feedgrade bulk (kg) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# toilet - standard on trailer                                ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'toilet - standard on trailer'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'toilet - standard on trailer',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'toilet - standard on trailer':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'toilet - standard on trailer') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} toilet - standard on trailer rows to 'L7_equipment_hire'")
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
    'Still unmatched after toilet - standard on trailer update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 toilet - standard on trailer  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"toilet - standard on trailer rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 100 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(100))
print(f"{'='*70}\n")

# =========================================================================================================================
# q-soil- screened                                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'q-soil- screened'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'q-soil- screened',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-soil- screened':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'q-soil- screened') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L8_Landscaping'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} q-soil- screened rows to 'L8_Landscaping'")
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
    'Still unmatched after q-soil- screened update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L8 q-soil- screened  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q-soil- screened rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# q-zland- cement-ultrachem 20kg                                                ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'q-zland- cement-ultrachem 20kg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'q-zland- cement-ultrachem 20kg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-zland- cement-ultrachem 20kg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'q-zland- cement-ultrachem 20kg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L9_infrastructure_consumables'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} q-zland- cement-ultrachem 20kg rows to 'L9_infrastructure_consumables'")
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
    'Still unmatched after q-zland- cement-ultrachem 20kg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L9 q-zland- cement-ultrachem 20kg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q-zland- cement-ultrachem 20kg rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")


# =========================================================================================================================
# ongas forklift cylinder rental                                        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ongas forklift cylinder rental'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ongas forklift cylinder rental',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ongas forklift cylinder rental':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ongas forklift cylinder rental') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ongas forklift cylinder rental rows to 'L7_equipment_hire'")
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
    'Still unmatched after ongas forklift cylinder rental update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 ongas forklift cylinder rental  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ongas forklift cylinder rental rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# compactor *  reversible 300kg                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'compactor *  reversible 300kg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'compactor *  reversible 300kg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'compactor *  reversible 300kg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'compactor *  reversible 300kg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} compactor *  reversible 300kg rows to 'L7_equipment_hire'")
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
    'Still unmatched after compactor *  reversible 300kg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 compactor *  reversible 300kg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"compactor *  reversible 300kg rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# flat bed trailer                                               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'flat bed trailer'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'flat bed trailer',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'flat bed trailer':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'flat bed trailer') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} flat bed trailer rows to 'L7_equipment_hire'")
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
    'Still unmatched after flat bed trailer update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 flat bed trailer  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"flat bed trailer rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# trailer * tandem crate (8x5)                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'trailer * tandem crate (8x5)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'trailer * tandem crate (8x5)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'trailer * tandem crate (8x5)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'trailer * tandem crate (8x5)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} trailer * tandem crate (8x5) rows to 'L7_equipment_hire'")
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
    'Still unmatched after trailer * tandem crate (8x5) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 trailer * tandem crate (8x5)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"trailer * tandem crate (8x5) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# filter sleeve 150 x 610                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'filter sleeve 150 x 610'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'filter sleeve 150 x 610',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'filter sleeve 150 x 610':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'filter sleeve 150 x 610') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} filter sleeve 150 x 610 rows to 'L7_mechanic'")
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
    'Still unmatched after filter sleeve 150 x 610 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 filter sleeve 150 x 610  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"filter sleeve 150 x 610 rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# fuel filter                                     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'fuel filter'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'fuel filter',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'fuel filter':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'fuel filter') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} fuel filter rows to 'L7_mechanic'")
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
    'Still unmatched after fuel filter update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 fuel filter  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"fuel filter rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# sundry                                          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'sundry'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'sundry',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'sundry':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'sundry') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} sundry rows to 'L7_mechanic'")
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
    'Still unmatched after sundry update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 sundry  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"sundry rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# w  vehicle cof trailer 4 axle                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'w  vehicle cof trailer 4 axle'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'w  vehicle cof trailer 4 axle',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'w  vehicle cof trailer 4 axle':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'w  vehicle cof trailer 4 axle') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} w  vehicle cof trailer 4 axle rows to 'L7_mechanic'")
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
    'Still unmatched after w  vehicle cof trailer 4 axle update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 w  vehicle cof trailer 4 axle  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"w  vehicle cof trailer 4 axle rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# w  vehicle pre cof truck 4 axle                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'w  vehicle pre cof truck 4 axle'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'w  vehicle pre cof truck 4 axle',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'w  vehicle pre cof truck 4 axle':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'w  vehicle pre cof truck 4 axle') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} w  vehicle pre cof truck 4 axle rows to 'L7_mechanic'")
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
    'Still unmatched after w  vehicle pre cof truck 4 axle update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 w  vehicle pre cof truck 4 axle  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"w  vehicle pre cof truck 4 axle rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# bandage : per roll : padding : soffban :        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'bandage : per roll : padding : soffban :'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'bandage : per roll : padding : soffban :',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'bandage : per roll : padding : soffban :':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'bandage : per roll : padding : soffban :') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} bandage : per roll : padding : soffban : rows to 'L5_Vet'")
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
    'Still unmatched after bandage : per roll : padding : soffban : update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 bandage : per roll : padding : soffban :  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"bandage : per roll : padding : soffban : rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# blackhawk original adult dog - fish & po      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'blackhawk original adult dog - fish & po'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'blackhawk original adult dog - fish & po',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'blackhawk original adult dog - fish & po':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'blackhawk original adult dog - fish & po') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} blackhawk original adult dog - fish & po rows to 'L5_Vet'")
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
    'Still unmatched after blackhawk original adult dog - fish & po update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 blackhawk original adult dog - fish & po  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"blackhawk original adult dog - fish & po rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# bovatec - 23.5 ltr                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'bovatec - 23.5 ltr'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'bovatec - 23.5 ltr',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'bovatec - 23.5 ltr':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'bovatec - 23.5 ltr') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} bovatec - 23.5 ltr rows to 'L5_Vet'")
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
    'Still unmatched after bovatec - 23.5 ltr update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 bovatec - 23.5 ltr  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"bovatec - 23.5 ltr rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : anaesthesia : gaseous                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : anaesthesia : gaseous'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : anaesthesia : gaseous',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : anaesthesia : gaseous':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : anaesthesia : gaseous') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : anaesthesia : gaseous rows to 'L5_Vet'")
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
    'Still unmatched after ca : anaesthesia : gaseous update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : anaesthesia : gaseous  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : anaesthesia : gaseous rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : consultation : post operative : sta    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : consultation : post operative : sta'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : consultation : post operative : sta',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : consultation : post operative : sta':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : consultation : post operative : sta') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : consultation : post operative : sta rows to 'L5_Vet'")
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
    'Still unmatched after ca : consultation : post operative : sta update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : consultation : post operative : sta  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : consultation : post operative : sta rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : consultation : well pet : standard     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : consultation : well pet : standard'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : consultation : well pet : standard',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : consultation : well pet : standard':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : consultation : well pet : standard') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : consultation : well pet : standard rows to 'L5_Vet'")
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
    'Still unmatched after ca : consultation : well pet : standard update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : consultation : well pet : standard  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : consultation : well pet : standard rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : professional time : surgical : per       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : professional time : surgical : per'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : professional time : surgical : per',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : professional time : surgical : per':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : professional time : surgical : per') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : professional time : surgical : per rows to 'L5_Vet'")
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
    'Still unmatched after ca : professional time : surgical : per update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : professional time : surgical : per  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : professional time : surgical : per rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ca : treatment consumables 2 (upto 15min      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ca : treatment consumables 2 (upto 15min'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ca : treatment consumables 2 (upto 15min',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : treatment consumables 2 (upto 15min':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ca : treatment consumables 2 (upto 15min') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ca : treatment consumables 2 (upto 15min rows to 'L5_Vet'")
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
    'Still unmatched after ca : treatment consumables 2 (upto 15min update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : treatment consumables 2 (upto 15min  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : treatment consumables 2 (upto 15min rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# calpro 375 500ml (bomaflex)                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'calpro 375 500ml (bomaflex)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'calpro 375 500ml (bomaflex)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'calpro 375 500ml (bomaflex)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'calpro 375 500ml (bomaflex)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} calpro 375 500ml (bomaflex) rows to 'L5_Vet'")
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
    'Still unmatched after calpro 375 500ml (bomaflex) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 calpro 375 500ml (bomaflex)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"calpro 375 500ml (bomaflex) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# copaject inj. 200ml                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'copaject inj. 200ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'copaject inj. 200ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'copaject inj. 200ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'copaject inj. 200ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} copaject inj. 200ml rows to 'L5_Vet'")
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
    'Still unmatched after copaject inj. 200ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 copaject inj. 200ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"copaject inj. 200ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# corporal oral drench 5l                                         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'corporal oral drench 5l'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'corporal oral drench 5l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'corporal oral drench 5l':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'corporal oral drench 5l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} corporal oral drench 5l rows to 'L5_Vet'")
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
    'Still unmatched after corporal oral drench 5l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 corporal oral drench 5l  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"corporal oral drench 5l rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# droncit tablet for cats & dogs  - per ta            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'droncit tablet for cats & dogs  - per ta'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'droncit tablet for cats & dogs  - per ta',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'droncit tablet for cats & dogs  - per ta':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'droncit tablet for cats & dogs  - per ta') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} droncit tablet for cats & dogs  - per ta rows to 'L5_Vet'")
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
    'Still unmatched after droncit tablet for cats & dogs  - per ta update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 droncit tablet for cats & dogs  - per ta  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"droncit tablet for cats & dogs  - per ta rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# drontal allwormer for cats (6kg tablet)         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'drontal allwormer for cats (6kg tablet)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'drontal allwormer for cats (6kg tablet)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'drontal allwormer for cats (6kg tablet)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'drontal allwormer for cats (6kg tablet)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} drontal allwormer for cats (6kg tablet) rows to 'L5_Vet'")
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
    'Still unmatched after drontal allwormer for cats (6kg tablet) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 drontal allwormer for cats (6kg tablet)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"drontal allwormer for cats (6kg tablet) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")


# =========================================================================================================================
# excede la                                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'excede la'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'excede la',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'excede la':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'excede la') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} excede la rows to 'L5_Vet'")
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
    'Still unmatched after excede la update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 excede la  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"excede la rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# glycerine bp - 200 ltr                               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'glycerine bp - 200 ltr'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'glycerine bp - 200 ltr',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'glycerine bp - 200 ltr':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'glycerine bp - 200 ltr') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} glycerine bp - 200 ltr rows to 'L5_Vet'")
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
    'Still unmatched after glycerine bp - 200 ltr update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 glycerine bp - 200 ltr  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"glycerine bp - 200 ltr rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# hills prescription diet metabolic + mobi                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'hills prescription diet metabolic + mobi'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'hills prescription diet metabolic + mobi',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'hills prescription diet metabolic + mobi':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'hills prescription diet metabolic + mobi') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} hills prescription diet metabolic + mobi rows to 'L5_Vet'")
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
    'Still unmatched after hills prescription diet metabolic + mobi update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 hills prescription diet metabolic + mobi  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"hills prescription diet metabolic + mobi rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# hills prescription diet t/d cat (dry foo         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'hills prescription diet t/d cat (dry foo'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'hills prescription diet t/d cat (dry foo',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'hills prescription diet t/d cat (dry foo':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'hills prescription diet t/d cat (dry foo') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} hills prescription diet t/d cat (dry foo rows to 'L5_Vet'")
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
    'Still unmatched after hills prescription diet t/d cat (dry foo update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 hills prescription diet t/d cat (dry foo  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"hills prescription diet t/d cat (dry foo rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# iv fluids : initial therapeutic : add fl        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'iv fluids : initial therapeutic : add fl'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'iv fluids : initial therapeutic : add fl',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'iv fluids : initial therapeutic : add fl':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'iv fluids : initial therapeutic : add fl') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} iv fluids : initial therapeutic : add fl rows to 'L5_Vet'")
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
    'Still unmatched after iv fluids : initial therapeutic : add fl update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 iv fluids : initial therapeutic : add fl  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"iv fluids : initial therapeutic : add fl rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# iv fluids : intraoperative : routine pro        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'iv fluids : intraoperative : routine pro'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'iv fluids : intraoperative : routine pro',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'iv fluids : intraoperative : routine pro':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'iv fluids : intraoperative : routine pro') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} iv fluids : intraoperative : routine pro rows to 'L5_Vet'")
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
    'Still unmatched after iv fluids : intraoperative : routine pro update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 iv fluids : intraoperative : routine pro  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"iv fluids : intraoperative : routine pro rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ketamine injection 10% (100mg/ml)               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ketamine injection 10% (100mg/ml)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ketamine injection 10% (100mg/ml)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ketamine injection 10% (100mg/ml)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ketamine injection 10% (100mg/ml)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ketamine injection 10% (100mg/ml) rows to 'L5_Vet'")
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
    'Still unmatched after ketamine injection 10% (100mg/ml) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ketamine injection 10% (100mg/ml)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ketamine injection 10% (100mg/ml) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# ketomax 15% cow care pack                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'ketomax 15% cow care pack'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'ketomax 15% cow care pack',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ketomax 15% cow care pack':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'ketomax 15% cow care pack') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} ketomax 15% cow care pack rows to 'L5_Vet'")
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
    'Still unmatched after ketomax 15% cow care pack update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ketomax 15% cow care pack  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ketomax 15% cow care pack rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - body condition score                             ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - body condition score'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - body condition score',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - body condition score':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - body condition score') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - body condition score rows to 'L5_Vet'")
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
    'Still unmatched after la - body condition score update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - body condition score  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - body condition score rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - calf disbudding base (no metacam)            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - calf disbudding base (no metacam)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - calf disbudding base (no metacam)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - calf disbudding base (no metacam)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - calf disbudding base (no metacam)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - calf disbudding base (no metacam) rows to 'L5_Vet'")
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
    'Still unmatched after la - calf disbudding base (no metacam) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - calf disbudding base (no metacam)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - calf disbudding base (no metacam) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - consum sterile kit                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - consum sterile kit'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - consum sterile kit',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum sterile kit':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - consum sterile kit') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - consum sterile kit rows to 'L5_Vet'")
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
    'Still unmatched after la - consum sterile kit update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - consum sterile kit  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - consum sterile kit rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# la - discount fee only                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'la - discount fee only'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'la - discount fee only',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - discount fee only':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'la - discount fee only') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} la - discount fee only rows to 'L5_Vet'")
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
    'Still unmatched after la - discount fee only update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - discount fee only  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - discount fee only rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# lepto 4-way vaccine 500ml (per 2ml dose)                ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'lepto 4-way vaccine 500ml (per 2ml dose)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'lepto 4-way vaccine 500ml (per 2ml dose)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'lepto 4-way vaccine 500ml (per 2ml dose)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'lepto 4-way vaccine 500ml (per 2ml dose)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} lepto 4-way vaccine 500ml (per 2ml dose) rows to 'L5_Vet'")
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
    'Still unmatched after lepto 4-way vaccine 500ml (per 2ml dose) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 lepto 4-way vaccine 500ml (per 2ml dose)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"lepto 4-way vaccine 500ml (per 2ml dose) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# leptoshield 2 500ml (per 2ml dose)                   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'leptoshield 2 500ml (per 2ml dose)'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'leptoshield 2 500ml (per 2ml dose)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'leptoshield 2 500ml (per 2ml dose)':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'leptoshield 2 500ml (per 2ml dose)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} leptoshield 2 500ml (per 2ml dose) rows to 'L5_Vet'")
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
    'Still unmatched after leptoshield 2 500ml (per 2ml dose) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 leptoshield 2 500ml (per 2ml dose)  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"leptoshield 2 500ml (per 2ml dose) rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# lypor 5l                                          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'lypor 5l'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'lypor 5l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'lypor 5l':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'lypor 5l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} lypor 5l rows to 'L5_Vet'")
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
    'Still unmatched after lypor 5l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 lypor 5l  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"lypor 5l rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# metrivet syringes                                                ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'metrivet syringes'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'metrivet syringes',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'metrivet syringes':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'metrivet syringes') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} metrivet syringes rows to 'L5_Vet'")
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
    'Still unmatched after metrivet syringes update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 metrivet syringes  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"metrivet syringes rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# microchip : implantation : standard                                   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'microchip : implantation : standard'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'microchip : implantation : standard',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'microchip : implantation : standard':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'microchip : implantation : standard') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} microchip : implantation : standard rows to 'L5_Vet'")
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
    'Still unmatched after microchip : implantation : standard update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 microchip : implantation : standard  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"microchip : implantation : standard rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# nexgard spectra for large dogs (15.1-30k               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'nexgard spectra for large dogs (15.1-30k'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'nexgard spectra for large dogs (15.1-30k',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'nexgard spectra for large dogs (15.1-30k':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'nexgard spectra for large dogs (15.1-30k') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} nexgard spectra for large dogs (15.1-30k rows to 'L5_Vet'")
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
    'Still unmatched after nexgard spectra for large dogs (15.1-30k update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 nexgard spectra for large dogs (15.1-30k  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"nexgard spectra for large dogs (15.1-30k rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# orbenin dc                                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'orbenin dc'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'orbenin dc',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'orbenin dc':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'orbenin dc') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} orbenin dc rows to 'L5_Vet'")
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
    'Still unmatched after orbenin dc update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 orbenin dc  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"orbenin dc rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# royal canin mini puppy (dry food) - per                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'royal canin mini puppy (dry food) - per'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'royal canin mini puppy (dry food) - per',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'royal canin mini puppy (dry food) - per':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'royal canin mini puppy (dry food) - per') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} royal canin mini puppy (dry food) - per rows to 'L5_Vet'")
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
    'Still unmatched after royal canin mini puppy (dry food) - per update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 royal canin mini puppy (dry food) - per  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"royal canin mini puppy (dry food) - per rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# royal canin neutered adult large dog (dr           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'royal canin neutered adult large dog (dr'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'royal canin neutered adult large dog (dr',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'royal canin neutered adult large dog (dr':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'royal canin neutered adult large dog (dr') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} royal canin neutered adult large dog (dr rows to 'L5_Vet'")
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
    'Still unmatched after royal canin neutered adult large dog (dr update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 royal canin neutered adult large dog (dr  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"royal canin neutered adult large dog (dr rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# royal canin veterinary gastro intestinal          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'royal canin veterinary gastro intestinal'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'royal canin veterinary gastro intestinal',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'royal canin veterinary gastro intestinal':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'royal canin veterinary gastro intestinal') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} royal canin veterinary gastro intestinal rows to 'L5_Vet'")
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
    'Still unmatched after royal canin veterinary gastro intestinal update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 royal canin veterinary gastro intestinal  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"royal canin veterinary gastro intestinal rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# rumenox 12kg                                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'rumenox 12kg'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'rumenox 12kg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'rumenox 12kg':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'rumenox 12kg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} rumenox 12kg rows to 'L5_Vet'")
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
    'Still unmatched after rumenox 12kg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 rumenox 12kg  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"rumenox 12kg rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# salvexin-b 500ml                                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'salvexin-b 500ml'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'salvexin-b 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'salvexin-b 500ml':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'salvexin-b 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} salvexin-b 500ml rows to 'L5_Vet'")
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
    'Still unmatched after salvexin-b 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 salvexin-b 500ml  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"salvexin-b 500ml rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# selekt off feed                                                   ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'selekt off feed'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'selekt off feed',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'selekt off feed':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'selekt off feed') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} selekt off feed rows to 'L5_Vet'")
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
    'Still unmatched after selekt off feed update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 selekt off feed  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"selekt off feed rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# soft seal                                                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'soft seal'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'soft seal',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'soft seal':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'soft seal') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} soft seal rows to 'L5_Vet'")
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
    'Still unmatched after soft seal update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 soft seal  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"soft seal rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")

# =========================================================================================================================
# vaccine : vanguard ccb oral                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_excavator = unmatched_df['description'].str.lower() == 'vaccine : vanguard ccb oral'

excavator_df = save_and_summarize2(
    unmatched_df, 
    mask_excavator, 
    '13.6_filtered_mask_WIP.csv',
    'vaccine : vanguard ccb oral',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'vaccine : vanguard ccb oral':")
print(excavator_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la vet per min charge rows in the FULL dataframe
mask_excavator_full = (
    (full_df['description'].str.lower() == 'vaccine : vanguard ccb oral') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_excavator_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_excavator_full.sum():,} vaccine : vanguard ccb oral rows to 'L5_Vet'")
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
    'Still unmatched after vaccine : vanguard ccb oral update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 vaccine : vanguard ccb oral  MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"vaccine : vanguard ccb oral rows flagged: {mask_excavator_full.sum():,}")
print(f"Remaining unmatched rows: {len(unmatched_df):,}")
print(f"{'='*70}")

# =========================================================
# ANALYZE REMAINING UNMATCHED DESCRIPTIONS
# =========================================================
print(f"\n{'='*70}")
print(f"TOP 50 MOST COMMON DESCRIPTION VALUES IN REMAINING UNMATCHED")
print(f"{'='*70}")
print(unmatched_df['description'].value_counts().head(50))
print(f"{'='*70}\n")




