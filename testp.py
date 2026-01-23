'''
Docstring for 12.1_try_to_match

this script attempts to match undiscounted invoice line items to merchants in the
Merchant Discount Detail reference file.
merchant_identifier values are normalized (lowercased, punctuation removed, common
suffixes removed) and then matched conservatively using exact / substring containment.
matched merchant metadata and discount fields are appended to each invoice line.

inputs:
- 12_invoice_line_items_undiscounted_only.csv
- Merchant Discount Detail.xlsx

outputs:
- 12_invoice_line_items_undiscounted_matched_merchant.csv
    undiscounted invoice line items with additional columns for matched merchant
    details (ats number, account name) and discount reference fields
'''

import pandas as pd
import os
import re
from tqdm import tqdm  # Optional: for progress bar

# =========================================================
# PATH CONFIGURATION
# =========================================================
BASE_ROOT = r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202"

# Input A: undiscounted invoice line items
INVOICE_PATH = os.path.join(
    BASE_ROOT,
    r"RuralCo3\data_cleaning\12_invoice_line_items_undiscounted_only.csv"
)

# Input B: merchant discount detail (Excel)
MERCHANT_PATH = os.path.join(
    BASE_ROOT,
    "Merchant Discount Detail.xlsx"
)

# Output (keep prefix "12_")
OUTPUT_PATH = os.path.join(
    BASE_ROOT,
    r"RuralCo3\data_cleaning\12_invoice_line_items_undiscounted_matched_merchant.csv"
)

# =========================================================
# 1. Load data
# =========================================================
print("Loading data...")
invoice_df = pd.read_csv(INVOICE_PATH, low_memory=False)
merchant_df = pd.read_excel(MERCHANT_PATH)

print(f"Loaded {len(invoice_df):,} invoice lines")
print(f"Loaded {len(merchant_df):,} merchant records")

# =========================================================
# 2. Helper: clean merchant names
# =========================================================
def clean_merchant_name(x):
    """
    Normalize merchant/account names for conservative matching:
    - lowercase
    - remove punctuation
    - remove common legal suffixes / generic words
    - collapse whitespace
    """
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(
        r"\b(ltd|limited|company|co|nz|head office|holdings|group)\b",
        "",
        x
    )
    x = re.sub(r"\s+", " ", x).strip()
    return x

print("Cleaning merchant names...")
invoice_df["merchant_clean"] = invoice_df["merchant_identifier"].apply(clean_merchant_name)
merchant_df["account_clean"] = merchant_df["Account Name"].apply(clean_merchant_name)

# =========================================================
# 3. Build merchant lookup table - OPTIMIZED
# =========================================================
merchant_lookup = merchant_df[
    ["ATS Number", "Account Name", "account_clean", "Discount Offered", "Discount Offered 2"]
].drop_duplicates()

# Convert to list of tuples for faster iteration
merchant_list = [
    (row["account_clean"], row["ATS Number"], row["Account Name"], 
     row["Discount Offered"], row["Discount Offered 2"])
    for _, row in merchant_lookup.iterrows()
    if pd.notna(row["account_clean"]) and row["account_clean"]
]

print(f"Built lookup with {len(merchant_list):,} unique merchants")

# =========================================================
# 4. Merchant matching (exact / contains, conservative) - OPTIMIZED
# =========================================================
def match_merchant_fast(merchant_name):
    """
    Match invoice merchant to discount merchant using:
    - exact / substring containment on cleaned names
    Returns merchant metadata if matched, else None.
    """
    if not merchant_name:
        return pd.Series([None, None, None, None])

    for key, ats, account, disc1, disc2 in merchant_list:
        if key in merchant_name or merchant_name in key:
            return pd.Series([ats, account, disc1, disc2])

    return pd.Series([None, None, None, None])

print("Matching merchants (this may take a few minutes)...")

# Use progress bar if tqdm is available
try:
    tqdm.pandas()
    invoice_df[
        [
            "matched_ats_number",
            "matched_account_name",
            "matched_discount_offered",
            "matched_discount_detail"
        ]
    ] = invoice_df["merchant_clean"].progress_apply(match_merchant_fast)
except:
    invoice_df[
        [
            "matched_ats_number",
            "matched_account_name",
            "matched_discount_offered",
            "matched_discount_detail"
        ]
    ] = invoice_df["merchant_clean"].apply(match_merchant_fast)

# =========================================================
# 5. Sanity checks / summary
# =========================================================
print("=" * 70)
print("MERCHANT MATCHING SUMMARY (UNDISCOUNTED INVOICE LINE ITEMS)")
print("=" * 70)

total = len(invoice_df)
matched = invoice_df["matched_account_name"].notna().sum()

print(f"Total undiscounted invoice lines: {total:,}")
print(f"Matched to discount merchants:    {matched:,}")
print(f"Match rate:                       {matched / total * 100:.2f}%")

print("\nTop 10 matched merchants (by line count):")
top_merchants = (
    invoice_df.loc[invoice_df["matched_account_name"].notna(), "matched_account_name"]
    .value_counts()
    .head(10)
)
print(top_merchants.to_string())

print("=" * 70)

# =========================================================
# 6. Save output
# =========================================================
print("Saving output...")
invoice_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved matched file to:\n{OUTPUT_PATH}")
print("=" * 70)