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
full_df = pd.read_csv(merchant_folder_dir / '13.6_matching_progress.csv')

# Load merchant reference data
MERCHANT_PATH = base_dir.parent / "Merchant Discount Detail.xlsx"
merchant_df = pd.read_excel(MERCHANT_PATH)

# Load remaining unmatched items from previous script
unmatched_df = pd.read_csv(merchant_folder_dir / '13.6_invoice_line_items_still_unmatched.csv')

print(f"\n{'='*70}")
print(f"STARTING 13.7 L9 MATCHING")
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
# urea ammonium nitrate solution  (ltr)       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'urea ammonium nitrate solution  (ltr)'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'urea ammonium nitrate solution  (ltr)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'urea ammonium nitrate solution  (ltr)':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the urea ammonium nitrate solution  (ltr) rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'urea ammonium nitrate solution  (ltr)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} urea ammonium nitrate solution  (ltr) 'L7_mechanic'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after urea ammonium nitrate solution  (ltr) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 urea ammonium nitrate solution  (ltr) MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"urea ammonium nitrate solution  (ltr) rows flagged: {mask_Charge_full.sum():,}")
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
# heart children donation $2                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'heart children donation $2'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'heart children donation $2',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'heart children donation $2':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the heart children donation $2 rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'heart children donation $2') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} heart children donation $2 'L7_mechanic'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after heart children donation $2 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 heart children donation $2 MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"heart children donation $2 rows flagged: {mask_Charge_full.sum():,}")
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
# glycerine bp - 100 ltr                                ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'glycerine bp - 100 ltr'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'glycerine bp - 100 ltr',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'glycerine bp - 100 ltr':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the glycerine bp - 100 ltr rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'glycerine bp - 100 ltr') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} glycerine bp - 100 ltr 'L7_mechanic'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after glycerine bp - 100 ltr update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 glycerine bp - 100 ltr MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"glycerine bp - 100 ltr rows flagged: {mask_Charge_full.sum():,}")
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
# gear oil 80/90                                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'gear oil 80/90'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'gear oil 80/90',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'gear oil 80/90':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the gear oil 80/90 rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'gear oil 80/90') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_mechanic'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} gear oil 80/90 'L7_mechanic'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after gear oil 80/90 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 gear oil 80/90 MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"gear oil 80/90 rows flagged: {mask_Charge_full.sum():,}")
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
# trailer * tilt bed                                                        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'trailer * tilt bed'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'trailer * tilt bed',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'trailer * tilt bed':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the trailer * tilt bed rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'trailer * tilt bed') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} trailer * tilt bed 'L7_equipment_hire'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after trailer * tilt bed update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 trailer * tilt bed MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"trailer * tilt bed rows flagged: {mask_Charge_full.sum():,}")
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
# trailer * crate (8x4)                                                               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'trailer * crate (8x4)'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'trailer * crate (8x4)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'trailer * crate (8x4)':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the trailer * crate (8x4) rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'trailer * crate (8x4)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} trailer * crate (8x4) 'L7_equipment_hire'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after trailer * crate (8x4) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 trailer * crate (8x4) MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"trailer * crate (8x4) rows flagged: {mask_Charge_full.sum():,}")
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
# trailer * 7x4                                                                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'trailer * 7x4'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'trailer * 7x4',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'trailer * 7x4':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the trailer * 7x4 rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'trailer * 7x4') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} trailer * 7x4 'L7_equipment_hire'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after trailer * 7x4 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 trailer * 7x4 MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"trailer * 7x4 rows flagged: {mask_Charge_full.sum():,}")
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
# toilet - tandem flush                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'toilet - tandem flush'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'toilet - tandem flush',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'toilet - tandem flush':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the toilet - tandem flush rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'toilet - tandem flush') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} toilet - tandem flush 'L7_equipment_hire'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after toilet - tandem flush update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 toilet - tandem flush MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"toilet - tandem flush rows flagged: {mask_Charge_full.sum():,}")
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
# toilet - standard                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'toilet - standard'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'toilet - standard',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'toilet - standard':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the toilet - standard rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'toilet - standard') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} toilet - standard 'L7_equipment_hire'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after toilet - standard update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 toilet - standard MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"toilet - standard rows flagged: {mask_Charge_full.sum():,}")
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
# tip truck (blue)                                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'tip truck (blue)'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'tip truck (blue)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'tip truck (blue)':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the tip truck (blue) rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'tip truck (blue)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} tip truck (blue) 'L7_equipment_hire'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after tip truck (blue) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 tip truck (blue) MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"tip truck (blue) rows flagged: {mask_Charge_full.sum():,}")
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
# cherrypicker - 13m                                        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'cherrypicker - 13m'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'cherrypicker - 13m',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'cherrypicker - 13m':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the cherrypicker - 13m rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'cherrypicker - 13m') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L7_equipment_hire'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} cherrypicker - 13m 'L7_equipment_hire'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after cherrypicker - 13m update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L7 cherrypicker - 13m MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"cherrypicker - 13m rows flagged: {mask_Charge_full.sum():,}")
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
# acezine 2 10ml                                                     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'acezine 2 10ml'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'acezine 2 10ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'acezine 2 10ml':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the acezine 2 10ml rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'acezine 2 10ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} acezine 2 10ml 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after acezine 2 10ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 acezine 2 10ml MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"acezine 2 10ml rows flagged: {mask_Charge_full.sum():,}")
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
# albiotic (lincocin forte)                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'albiotic (lincocin forte)'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'albiotic (lincocin forte)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'albiotic (lincocin forte)':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the albiotic (lincocin forte) rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'albiotic (lincocin forte)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} albiotic (lincocin forte) 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after albiotic (lincocin forte) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 albiotic (lincocin forte) MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"albiotic (lincocin forte) rows flagged: {mask_Charge_full.sum():,}")
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
# aquafol 100ml                                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'aquafol 100ml'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'aquafol 100ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'aquafol 100ml':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the aquafol 100ml rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'aquafol 100ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} aquafol 100ml 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after aquafol 100ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 aquafol 100ml MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"aquafol 100ml rows flagged: {mask_Charge_full.sum():,}")
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
# bearing, 2 rubber seal                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'bearing, 2 rubber seal'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'bearing, 2 rubber seal',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'bearing, 2 rubber seal':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the bearing, 2 rubber seal rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'bearing, 2 rubber seal') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} bearing, 2 rubber seal 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after bearing, 2 rubber seal update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 bearing, 2 rubber seal MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"bearing, 2 rubber seal rows flagged: {mask_Charge_full.sum():,}")
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
# boss pour-on 2.5l                                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'boss pour-on 2.5l'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'boss pour-on 2.5l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'boss pour-on 2.5l':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the boss pour-on 2.5l rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'boss pour-on 2.5l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} boss pour-on 2.5l 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after boss pour-on 2.5l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 boss pour-on 2.5l MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"boss pour-on 2.5l rows flagged: {mask_Charge_full.sum():,}")
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
# bravecto plus spot-on for medium cats (2             ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'bravecto plus spot-on for medium cats (2'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'bravecto plus spot-on for medium cats (2',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'bravecto plus spot-on for medium cats (2':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the bravecto plus spot-on for medium cats (2 rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'bravecto plus spot-on for medium cats (2') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} bravecto plus spot-on for medium cats (2 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after bravecto plus spot-on for medium cats (2 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 bravecto plus spot-on for medium cats (2 MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"bravecto plus spot-on for medium cats (2 rows flagged: {mask_Charge_full.sum():,}")
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
# ca : theatre fee (sterile)                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'ca : theatre fee (sterile)'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'ca : theatre fee (sterile)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ca : theatre fee (sterile)':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the ca : theatre fee (sterile) rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'ca : theatre fee (sterile)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} ca : theatre fee (sterile) 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after ca : theatre fee (sterile) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ca : theatre fee (sterile) MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ca : theatre fee (sterile) rows flagged: {mask_Charge_full.sum():,}")
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
# carafate 1g tablets                                     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'carafate 1g tablets'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'carafate 1g tablets',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'carafate 1g tablets':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the carafate 1g tablets rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'carafate 1g tablets') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} carafate 1g tablets 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after carafate 1g tablets update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 carafate 1g tablets MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"carafate 1g tablets rows flagged: {mask_Charge_full.sum():,}")
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
# cerenia injection 20ml                                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'cerenia injection 20ml'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'cerenia injection 20ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'cerenia injection 20ml':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the cerenia injection 20ml rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'cerenia injection 20ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} cerenia injection 20ml 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after cerenia injection 20ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 cerenia injection 20ml MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"cerenia injection 20ml rows flagged: {mask_Charge_full.sum():,}")
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
# clavulox tablets 50mg                                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'clavulox tablets 50mg'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'clavulox tablets 50mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'clavulox tablets 50mg':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the clavulox tablets 50mg rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'clavulox tablets 50mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} clavulox tablets 50mg 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after clavulox tablets 50mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 clavulox tablets 50mg MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"clavulox tablets 50mg rows flagged: {mask_Charge_full.sum():,}")
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
# coglavax 8in1 500ml                                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'coglavax 8in1 500ml'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'coglavax 8in1 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'coglavax 8in1 500ml':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the coglavax 8in1 500ml rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'coglavax 8in1 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} coglavax 8in1 500ml 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after coglavax 8in1 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 coglavax 8in1 500ml MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"coglavax 8in1 500ml rows flagged: {mask_Charge_full.sum():,}")
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
# drontal allwormer for cats (4kg tablet)                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'drontal allwormer for cats (4kg tablet)'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'drontal allwormer for cats (4kg tablet)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'drontal allwormer for cats (4kg tablet)':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the drontal allwormer for cats (4kg tablet) rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'drontal allwormer for cats (4kg tablet)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} drontal allwormer for cats (4kg tablet) 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after drontal allwormer for cats (4kg tablet) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 drontal allwormer for cats (4kg tablet) MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"drontal allwormer for cats (4kg tablet) rows flagged: {mask_Charge_full.sum():,}")
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
# eukanuba puppy large breed 17kg                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'eukanuba puppy large breed 17kg'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'eukanuba puppy large breed 17kg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'eukanuba puppy large breed 17kg':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the eukanuba puppy large breed 17kg rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'eukanuba puppy large breed 17kg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} eukanuba puppy large breed 17kg 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after eukanuba puppy large breed 17kg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 eukanuba puppy large breed 17kg MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"eukanuba puppy large breed 17kg rows flagged: {mask_Charge_full.sum():,}")
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
# exlab � la lic individual milk johnes te               ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'exlab � la lic individual milk johnes te'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'exlab � la lic individual milk johnes te',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'exlab � la lic individual milk johnes te':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the exlab � la lic individual milk johnes te rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'exlab � la lic individual milk johnes te') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} exlab � la lic individual milk johnes te 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after exlab � la lic individual milk johnes te update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 exlab � la lic individual milk johnes te MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"exlab � la lic individual milk johnes te rows flagged: {mask_Charge_full.sum():,}")
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
# fibor 500g                                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'fibor 500g'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'fibor 500g',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'fibor 500g':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the fibor 500g rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'fibor 500g') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} fibor 500g 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after fibor 500g update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 fibor 500g MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"fibor 500g rows flagged: {mask_Charge_full.sum():,}")
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
# flexidine inj 250ml                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'flexidine inj 250ml'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'flexidine inj 250ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'flexidine inj 250ml':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the flexidine inj 250ml rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'flexidine inj 250ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} flexidine inj 250ml 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after flexidine inj 250ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 flexidine inj 250ml MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"flexidine inj 250ml rows flagged: {mask_Charge_full.sum():,}")
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
# frudix tablet 40mg                                                     ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'frudix tablet 40mg'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'frudix tablet 40mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'frudix tablet 40mg':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the frudix tablet 40mg rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'frudix tablet 40mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} frudix tablet 40mg 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after frudix tablet 40mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 frudix tablet 40mg MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"frudix tablet 40mg rows flagged: {mask_Charge_full.sum():,}")
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
# gasket 12.1x16.2x3.7                                                           ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'gasket 12.1x16.2x3.7'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'gasket 12.1x16.2x3.7',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'gasket 12.1x16.2x3.7':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the gasket 12.1x16.2x3.7 rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'gasket 12.1x16.2x3.7') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} gasket 12.1x16.2x3.7 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after gasket 12.1x16.2x3.7 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 gasket 12.1x16.2x3.7 MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"gasket 12.1x16.2x3.7 rows flagged: {mask_Charge_full.sum():,}")
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
# la - consum - syringe & needle small (25                         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'la - consum - syringe & needle small (25'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'la - consum - syringe & needle small (25',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum - syringe & needle small (25':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la - consum - syringe & needle small (25 rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'la - consum - syringe & needle small (25') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} la - consum - syringe & needle small (25 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - consum - syringe & needle small (25 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - consum - syringe & needle small (25 MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - consum - syringe & needle small (25 rows flagged: {mask_Charge_full.sum():,}")
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
# la - consum farm visit                                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'la - consum farm visit'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'la - consum farm visit',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum farm visit':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la - consum farm visit rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'la - consum farm visit') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} la - consum farm visit 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - consum farm visit update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - consum farm visit MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - consum farm visit rows flagged: {mask_Charge_full.sum():,}")
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
# la - consum intramammary administration            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'la - consum intramammary administration'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'la - consum intramammary administration',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - consum intramammary administration':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la - consum intramammary administration rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'la - consum intramammary administration') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} la - consum intramammary administration 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - consum intramammary administration update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - consum intramammary administration MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - consum intramammary administration rows flagged: {mask_Charge_full.sum():,}")
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
# la - scan ageing                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'la - scan ageing'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'la - scan ageing',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - scan ageing':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la - scan ageing rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'la - scan ageing') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} la - scan ageing 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - scan ageing update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - scan ageing MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - scan ageing rows flagged: {mask_Charge_full.sum():,}")
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
# la - scan deer                                              ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'la - scan deer'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'la - scan deer',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - scan deer':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la - scan deer rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'la - scan deer') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} la - scan deer 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - scan deer update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - scan deer MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - scan deer rows flagged: {mask_Charge_full.sum():,}")
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
# la - teatseal administration per head                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'la - teatseal administration per head'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'la - teatseal administration per head',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - teatseal administration per head':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la - teatseal administration per head rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'la - teatseal administration per head') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} la - teatseal administration per head 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - teatseal administration per head update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - teatseal administration per head MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - teatseal administration per head rows flagged: {mask_Charge_full.sum():,}")
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
# la - teatseal trailer hire per heifer            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'la - teatseal trailer hire per heifer'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'la - teatseal trailer hire per heifer',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'la - teatseal trailer hire per heifer':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the la - teatseal trailer hire per heifer rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'la - teatseal trailer hire per heifer') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} la - teatseal trailer hire per heifer 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after la - teatseal trailer hire per heifer update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 la - teatseal trailer hire per heifer MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"la - teatseal trailer hire per heifer rows flagged: {mask_Charge_full.sum():,}")
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
# nexgard spectra for extra large dogs (30        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'nexgard spectra for extra large dogs (30'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'nexgard spectra for extra large dogs (30',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'nexgard spectra for extra large dogs (30':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the nexgard spectra for extra large dogs (30 rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'nexgard spectra for extra large dogs (30') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} nexgard spectra for extra large dogs (30 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after nexgard spectra for extra large dogs (30 update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 nexgard spectra for extra large dogs (30 MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"nexgard spectra for extra large dogs (30 rows flagged: {mask_Charge_full.sum():,}")
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
# prednisone 5mg                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'prednisone 5mg'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'prednisone 5mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'prednisone 5mg':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the prednisone 5mg rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'prednisone 5mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} prednisone 5mg 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after prednisone 5mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 prednisone 5mg MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"prednisone 5mg rows flagged: {mask_Charge_full.sum():,}")
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
# previcox tablets 227mg                                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'previcox tablets 227mg'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'previcox tablets 227mg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'previcox tablets 227mg':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the previcox tablets 227mg rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'previcox tablets 227mg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} previcox tablets 227mg 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after previcox tablets 227mg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 previcox tablets 227mg MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"previcox tablets 227mg rows flagged: {mask_Charge_full.sum():,}")
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
# q-scoop-sand- bedding sand                            ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'q-scoop-sand- bedding sand'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'q-scoop-sand- bedding sand',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-scoop-sand- bedding sand':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the q-scoop-sand- bedding sand rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'q-scoop-sand- bedding sand') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} q-scoop-sand- bedding sand 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after q-scoop-sand- bedding sand update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 q-scoop-sand- bedding sand MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q-scoop-sand- bedding sand rows flagged: {mask_Charge_full.sum():,}")
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
# q-scoop-zland- pine-mulch                          ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'q-scoop-zland- pine-mulch'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'q-scoop-zland- pine-mulch',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'q-scoop-zland- pine-mulch':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the q-scoop-zland- pine-mulch rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'q-scoop-zland- pine-mulch') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} q-scoop-zland- pine-mulch 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after q-scoop-zland- pine-mulch update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 q-scoop-zland- pine-mulch MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"q-scoop-zland- pine-mulch rows flagged: {mask_Charge_full.sum():,}")
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
# revive sachet 10l                                        ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'revive sachet 10l'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'revive sachet 10l',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'revive sachet 10l':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the revive sachet 10l rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'revive sachet 10l') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} revive sachet 10l 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after revive sachet 10l update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 revive sachet 10l MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"revive sachet 10l rows flagged: {mask_Charge_full.sum():,}")
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
# royal canin intense hairball 2kg                                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'royal canin intense hairball 2kg'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'royal canin intense hairball 2kg',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'royal canin intense hairball 2kg':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the royal canin intense hairball 2kg rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'royal canin intense hairball 2kg') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} royal canin intense hairball 2kg 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after royal canin intense hairball 2kg update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 royal canin intense hairball 2kg MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"royal canin intense hairball 2kg rows flagged: {mask_Charge_full.sum():,}")
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
# royal canin maxi puppy (dry food) - per                                ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'royal canin maxi puppy (dry food) - per'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'royal canin maxi puppy (dry food) - per',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'royal canin maxi puppy (dry food) - per':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the royal canin maxi puppy (dry food) - per rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'royal canin maxi puppy (dry food) - per') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} royal canin maxi puppy (dry food) - per 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after royal canin maxi puppy (dry food) - per update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 royal canin maxi puppy (dry food) - per MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"royal canin maxi puppy (dry food) - per rows flagged: {mask_Charge_full.sum():,}")
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
# royal canin veterinary urinary moderate                       ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'royal canin veterinary urinary moderate'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'royal canin veterinary urinary moderate',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'royal canin veterinary urinary moderate':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the royal canin veterinary urinary moderate  rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'royal canin veterinary urinary moderate') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} royal canin veterinary urinary moderate 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after royal canin veterinary urinary moderate update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 royal canin veterinary urinary moderate MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"royal canin veterinary urinary moderate rows flagged: {mask_Charge_full.sum():,}")
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
# ultravac 5in1 sel 500ml                                 ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'ultravac 5in1 sel 500ml'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'ultravac 5in1 sel 500ml',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'ultravac 5in1 sel 500ml':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the ultravac 5in1 sel 500ml  rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'ultravac 5in1 sel 500ml') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} ultravac 5in1 sel 500ml 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after ultravac 5in1 sel 500ml update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 ultravac 5in1 sel 500ml MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"ultravac 5in1 sel 500ml rows flagged: {mask_Charge_full.sum():,}")
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
# u-seal (200)                                                  ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'u-seal (200)'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'u-seal (200)',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'u-seal (200)':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the u-seal (200)  rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'u-seal (200)') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} u-seal (200) 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after u-seal (200) update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 u-seal (200) MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"u-seal (200) rows flagged: {mask_Charge_full.sum():,}")
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
# w 5th wheel/skid plate certification                      ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'w 5th wheel/skid plate certification'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'w 5th wheel/skid plate certification',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'w 5th wheel/skid plate certification':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the w 5th wheel/skid plate certification  rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'w 5th wheel/skid plate certification') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} w 5th wheel/skid plate certification 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after w 5th wheel/skid plate certification update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 w 5th wheel/skid plate certification MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"w 5th wheel/skid plate certification rows flagged: {mask_Charge_full.sum():,}")
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
# waterpump impeller                                    ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'waterpump impeller'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'waterpump impeller',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'waterpump impeller':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the waterpump impeller  rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'waterpump impeller') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} waterpump impeller 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after waterpump impeller update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 waterpump impeller MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"waterpump impeller rows flagged: {mask_Charge_full.sum():,}")
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
# yersiniavax - 50 dose                                                         ANALYSIS
# =========================================================================================================================
# Filter for rows with exact "Lube (Per Litre)" in description column
mask_Charge = unmatched_df['description'].str.lower() == 'yersiniavax - 50 dose'

Charge_df = save_and_summarize2(
    unmatched_df, 
    mask_Charge, 
    '13.7_filtered_mask_WIP.csv',
    'yersiniavax - 50 dose',
    output_dir=output_dir
)

# Print unique values
print("\nUnique description values matching 'yersiniavax - 50 dose':")
print(Charge_df['description'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH LA - Vet Per Min Charge FLAGS
# =========================================================
# Update the yersiniavax - 50 dose  rows in the FULL dataframe
mask_Charge_full = (
    (full_df['description'].str.lower() == 'yersiniavax - 50 dose') & 
    (full_df['match_layer'] == 'unmatched')
)

full_df.loc[mask_Charge_full, 'match_layer'] = 'L5_Vet'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_Charge_full.sum():,} yersiniavax - 50 dose 'L5_Vet'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts().sort_index())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13.7_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13.7_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

unmatched_df = save_and_summarize2(
    full_df, 
    unmatched_mask, 
    '13.7_invoice_line_items_still_unmatched.csv',
    'Still unmatched after yersiniavax - 50 dose update',
    output_dir=output_dir
)

print(f"\n{'='*70}")
print(f"L5 yersiniavax - 50 dose MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"yersiniavax - 50 dose rows flagged: {mask_Charge_full.sum():,}")
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

11