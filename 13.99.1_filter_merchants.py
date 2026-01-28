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
    - 13.99.1_filtered_merchants_swipe_unmatched.csv
"""

import pandas as pd
import numpy as np
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
# CREATE AND POPULATE CLEANED_DISCOUNT COLUMN
# =========================================================
print(f"\n{'='*70}")
print(f"CREATING AND POPULATING CLEANED_DISCOUNT COLUMN")
print(f"{'='*70}")

import re

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

# Extract percentages from both columns
df_filtered['discount_1_pct'] = df_filtered['Discount Offered'].apply(extract_percentage)
df_filtered['discount_2_pct'] = df_filtered['Discount Offered 2'].apply(extract_percentage)

# Create cleaned_discount column by finding the lowest percentage between the two columns
def get_lowest_discount(row):
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

# Create cleaned_discount with parsed percentages
df_filtered['cleaned_discount'] = df_filtered.apply(get_lowest_discount, axis=1)

# Drop temporary columns
df_filtered = df_filtered.drop(columns=['discount_1_pct', 'discount_2_pct'])

# Count convenience (0 values)
convenience_count = (df_filtered['cleaned_discount'] == 0).sum()

print(f"\nRows with 'convenience' (cleaned_discount = 0): {convenience_count:,}")
print(f"\nDiscount percentages parsed and cleaned_discount column created")
print(f"\nCleaned discount distribution:")
print(df_filtered['cleaned_discount'].value_counts().dropna().sort_index())
print(f"\nRows with values in cleaned_discount: {df_filtered['cleaned_discount'].notna().sum():,}")
print(f"Rows with blank in cleaned_discount: {df_filtered['cleaned_discount'].isna().sum():,}")

# =========================================================
# SUMMARY STATISTICS
# =========================================================
print(f"\n{'='*70}")
print(f"SUMMARY STATISTICS")
print(f"{'='*70}")

print(f"\nOriginal dataset: {len(merchant_df):,} rows")
print(f"After Swipe filter: {len(df_swipe):,} rows ({len(df_swipe)/len(merchant_df)*100:.1f}%)")
print(f"After excluding matched merchants: {len(df_filtered):,} rows ({len(df_filtered)/len(merchant_df)*100:.1f}%)")

print(f"\nUnique merchants in filtered dataset: {df_filtered['ATS Number'].nunique():,}")

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

output_file = output_dir / '13.99.1_filtered_merchants_swipe_unmatched.csv'
df_filtered.to_csv(output_file, index=False)

print(f"\nSaved filtered merchant data to: {output_file.name}")
print(f"Total rows saved: {len(df_filtered):,}")
print(f"{'='*70}\n")