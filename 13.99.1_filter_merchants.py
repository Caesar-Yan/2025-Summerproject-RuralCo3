"""
Script: 13.99.1_filter_merchants.py

Purpose:
    Filter merchant discount detail table to:
    1. Keep only rows with Invoicing method = 'Swipe'
    2. Exclude merchants whose ATS numbers are in the matched merchants list
    3. Exclude rows with statement discount text in Discount Offered 2
    4. Create cleaned_discount column (converting 'convenience' to 0, leaving others blank)

Input:
    - Merchant Discount Detail.xlsx
    - 13.99_matched_merchants.csv

Output:
    - 13.99.1_all_merchants_with_cleaned_discount.csv (all merchants with cleaned_discount)
    - 13.99.1_filtered_merchants_swipe_unmatched.csv (filtered merchants)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Set up paths
data_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202')
output_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3/merchant')

# =========================================================
# LOAD MATCHED MERCHANTS LIST
# =========================================================
print(f"\n{'='*70}")
print(f"LOADING MATCHED MERCHANTS LIST")
print(f"{'='*70}")

matched_merchants_file = output_dir / '13.99_matched_merchants.csv'
matched_merchants_df = pd.read_csv(matched_merchants_file)
matched_ats_numbers = set(matched_merchants_df['matched_ats_number'].values)

print(f"Loaded {len(matched_ats_numbers):,} matched ATS numbers to exclude")

# =========================================================
# LOAD MERCHANT DISCOUNT DATA
# =========================================================
print("\n" + "="*70)
print("LOADING MERCHANT DISCOUNT DATA")
print("="*70)

merchant_df = pd.read_excel(data_dir / 'Merchant Discount Detail.xlsx')

print(f"Total rows in merchant discount detail: {len(merchant_df):,}")
print(f"\nColumns in dataset:")
print(merchant_df.columns.tolist())

# =========================================================
# CREATE CLEANED_DISCOUNT FOR ALL MERCHANTS
# =========================================================
print(f"\n{'='*70}")
print(f"CREATING CLEANED_DISCOUNT COLUMN FOR ALL MERCHANTS")
print(f"{'='*70}")

def extract_percentage(value):
    """Extract percentage value from string like '2.5%' or '10%'"""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Check for 'convenience' first
        if 'convenience' in value.lower():
            return 0
        # Search for percentage pattern (e.g., "2.5%", "10%")
        match = re.search(r'(\d+\.?\d*)\s*%', str(value))
        if match:
            return float(match.group(1))
    return None

def get_lowest_discount(row):
    """Get the lowest discount percentage between the two discount columns"""
    # Get percentages from both columns
    pct1 = row['discount_1_pct']
    pct2 = row['discount_2_pct']
    
    # If both are available, return the minimum
    if pd.notna(pct1) and pd.notna(pct2):
        return min(pct1, pct2)
    # If only one is available, return that one
    elif pd.notna(pct1):
        return pct1
    elif pd.notna(pct2):
        return pct2
    # If neither is available, return NaN
    else:
        return np.nan

# Extract percentages from both columns for ALL merchants
merchant_df['discount_1_pct'] = merchant_df['Discount Offered'].apply(extract_percentage)
merchant_df['discount_2_pct'] = merchant_df['Discount Offered 2'].apply(extract_percentage)

# Create cleaned_discount column
merchant_df['cleaned_discount'] = merchant_df.apply(get_lowest_discount, axis=1)

# Drop temporary columns
merchant_df = merchant_df.drop(columns=['discount_1_pct', 'discount_2_pct'])

print(f"\nCleaned discount created for all {len(merchant_df):,} merchants")
print(f"Rows with values in cleaned_discount: {merchant_df['cleaned_discount'].notna().sum():,}")
print(f"Rows with blank in cleaned_discount: {merchant_df['cleaned_discount'].isna().sum():,}")

# Count convenience (0 values)
convenience_count = (merchant_df['cleaned_discount'] == 0).sum()
print(f"Rows with 'convenience' (cleaned_discount = 0): {convenience_count:,}")

print(f"\nCleaned discount distribution (top 20 values):")
print(merchant_df['cleaned_discount'].value_counts().dropna().sort_index().head(20))

# =========================================================
# SAVE ALL MERCHANTS WITH CLEANED DISCOUNT
# =========================================================
print(f"\n{'='*70}")
print(f"SAVING ALL MERCHANTS WITH CLEANED DISCOUNT")
print(f"{'='*70}")

all_merchants_output = output_dir / '13.99.1_all_merchants_with_cleaned_discount.csv'
merchant_df.to_csv(all_merchants_output, index=False)

print(f"Saved: {all_merchants_output.name} ({len(merchant_df):,} rows)")

# =========================================================
# FILTER FOR SWIPE INVOICE METHOD
# =========================================================
print(f"\n{'='*70}")
print(f"FILTERING FOR SWIPE INVOICING METHOD")
print(f"{'='*70}")

# Check unique Invoicing method values
print(f"\nUnique Invoicing method values:")
print(merchant_df['Invoicing method'].value_counts())

# Filter for Swipe only
df_swipe = merchant_df[merchant_df['Invoicing method'] == 'Swipe'].copy()

print(f"\nRows after filtering for Swipe: {len(df_swipe):,}")
print(f"Reduction: {len(merchant_df) - len(df_swipe):,} rows removed")

# =========================================================
# EXCLUDE MATCHED MERCHANTS
# =========================================================
print(f"\n{'='*70}")
print(f"EXCLUDING MATCHED MERCHANTS")
print(f"{'='*70}")

# Filter out matched merchants using 'ATS Number' column
df_filtered = df_swipe[~df_swipe['ATS Number'].isin(matched_ats_numbers)].copy()

print(f"\nRows after excluding matched merchants: {len(df_filtered):,}")
print(f"Reduction: {len(df_swipe) - len(df_filtered):,} rows removed")

# =========================================================
# EXCLUDE STATEMENT DISCOUNT ROWS
# =========================================================
print(f"\n{'='*70}")
print(f"EXCLUDING STATEMENT DISCOUNT ROWS")
print(f"{'='*70}")

# Define the texts to exclude (both variations)
statement_discount_texts = [
    "The discount will appear on your statement, not at point of sale",
    "Discount will appear on your statement, not at point of sale",
    "Statement"
]

# Count rows with either text variation
statement_discount_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
for text in statement_discount_texts:
    mask = df_filtered['Discount Offered 2'].astype(str).str.contains(
        text, case=False, na=False, regex=False
    )
    statement_discount_mask = statement_discount_mask | mask
    count = mask.sum()
    print(f"Rows with '{text}': {count:,}")

total_statement_discount_count = statement_discount_mask.sum()
print(f"\nTotal rows with statement discount text: {total_statement_discount_count:,}")

# Filter out rows with either text
df_filtered = df_filtered[~statement_discount_mask].copy()

print(f"Rows after excluding statement discount text: {len(df_filtered):,}")

# =========================================================
# EXCLUDE ROWS WITH BLANK DISCOUNTS
# =========================================================
print(f"\n{'='*70}")
print(f"EXCLUDING ROWS WITH BLANK DISCOUNTS")
print(f"{'='*70}")

# Count rows with blank values in both columns
blank_both = (df_filtered['Discount Offered'].isna()) & (df_filtered['Discount Offered 2'].isna())
blank_both_count = blank_both.sum()

print(f"\nRows with blank values in both Discount Offered and Discount Offered 2: {blank_both_count:,}")

# Filter out rows where both discount columns are blank
df_filtered = df_filtered[~blank_both].copy()

print(f"Rows after excluding blank discounts: {len(df_filtered):,}")

# =========================================================
# SUMMARY STATISTICS FOR FILTERED DATA
# =========================================================
print(f"\n{'='*70}")
print(f"SUMMARY STATISTICS - FILTERED DATA")
print(f"{'='*70}")

print(f"\nOriginal dataset: {len(merchant_df):,} rows")
print(f"After Swipe filter: {len(df_swipe):,} rows ({len(df_swipe)/len(merchant_df)*100:.1f}%)")
print(f"Final filtered dataset: {len(df_filtered):,} rows ({len(df_filtered)/len(merchant_df)*100:.1f}%)")

print(f"\nUnique merchants in filtered dataset: {df_filtered['ATS Number'].nunique():,}")

# Discount statistics for filtered data
print(f"\nFiltered data - Cleaned discount statistics:")
print(f"Rows with values in cleaned_discount: {df_filtered['cleaned_discount'].notna().sum():,}")
print(f"Rows with blank in cleaned_discount: {df_filtered['cleaned_discount'].isna().sum():,}")
print(f"Rows with 'convenience' (cleaned_discount = 0): {(df_filtered['cleaned_discount'] == 0).sum():,}")

print(f"\nFiltered data - Cleaned discount distribution:")
print(df_filtered['cleaned_discount'].value_counts().dropna().sort_index())

# Show sample of filtered data
print(f"\nFirst 20 rows of filtered dataset:")
print(df_filtered[['ATS Number', 'Account Name', 'Invoicing method', 
                   'Discount Offered', 'Discount Offered 2', 'cleaned_discount']].head(20).to_string(index=False))

# =========================================================
# SAVE FILTERED DATASET
# =========================================================
print(f"\n{'='*70}")
print(f"SAVING FILTERED DATASET")
print(f"{'='*70}")

filtered_output = output_dir / '13.99.1_filtered_merchants_swipe_unmatched.csv'
df_filtered.to_csv(filtered_output, index=False)

print(f"Saved: {filtered_output.name} ({len(df_filtered):,} rows)")

# =========================================================
# FINAL SUMMARY
# =========================================================
print(f"\n{'='*70}")
print(f"PROCESS COMPLETE!")
print(f"{'='*70}")

print(f"\n📁 OUTPUT FILES:")
print(f"  1. {all_merchants_output.name}")
print(f"     - All {len(merchant_df):,} merchants with cleaned_discount column")
print(f"  2. {filtered_output.name}")
print(f"     - Filtered {len(df_filtered):,} merchants (Swipe, unmatched, with discounts)")

print(f"\n📊 CLEANED DISCOUNT SUMMARY:")
print(f"  • All merchants with cleaned_discount: {merchant_df['cleaned_discount'].notna().sum():,}")
print(f"  • Filtered merchants with cleaned_discount: {df_filtered['cleaned_discount'].notna().sum():,}")

print(f"\n{'='*70}\n")