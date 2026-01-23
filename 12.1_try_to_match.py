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
invoice_df = pd.read_csv(INVOICE_PATH, low_memory=False)

# 默认读取第一个 sheet；如果后面发现不对，再显式指定 sheet_name
merchant_df = pd.read_excel(MERCHANT_PATH)

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

invoice_df["merchant_clean"] = invoice_df["merchant_identifier"].apply(clean_merchant_name)
merchant_df["account_clean"] = merchant_df["Account Name"].apply(clean_merchant_name)

# =========================================================
# 3. Build merchant lookup table
# =========================================================
merchant_lookup = merchant_df[
    ["ATS Number", "Account Name", "account_clean", "Discount Offered", "Discount Offered 2"]
].drop_duplicates()

# =========================================================
# 4. Merchant matching (exact / contains, conservative)
# =========================================================
def match_merchant(merchant_name):
    """
    Match invoice merchant to discount merchant using:
    - exact / substring containment on cleaned names
    Returns merchant metadata if matched, else None.
    """
    if not merchant_name:
        return pd.Series([None, None, None, None])

    for _, row in merchant_lookup.iterrows():
        key = row["account_clean"]
        if key and (key in merchant_name or merchant_name in key):
            return pd.Series([
                row["ATS Number"],
                row["Account Name"],
                row["Discount Offered"],
                row["Discount Offered 2"]
            ])

    return pd.Series([None, None, None, None])

invoice_df[
    [
        "matched_ats_number",
        "matched_account_name",
        "matched_discount_offered",
        "matched_discount_detail"
    ]
] = invoice_df["merchant_clean"].apply(match_merchant)

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
invoice_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved matched file to:\n{OUTPUT_PATH}")
print("=" * 70)

