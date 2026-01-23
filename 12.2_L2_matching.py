'''
Docstring for 12.2_L2_matching

this script performs a second-layer (L2) merchant matching for invoice line items
that were not matched in the first pass.
it uses description-based matching by normalizing the invoice description text and
searching for representative merchant tokens derived from Merchant Discount Detail
account names.
when a token is found in the description, the corresponding merchant metadata and
discount fields are appended, and match_source is set to "description".

inputs:
- 12_invoice_line_items_undiscounted_matched_merchant.csv
- Merchant Discount Detail.xlsx

outputs:
- 12_invoice_line_items_undiscounted_matched_merchant_L2.csv
    invoice line items with additional L2 matches applied for previously-unmatched
    rows, including matched merchant fields and a match_source indicator
'''

import pandas as pd
import os
import re

# =========================================================
# Paths
# =========================================================
BASE_ROOT = r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202"

IN_PATH = os.path.join(
    BASE_ROOT,
    r"RuralCo3\data_cleaning\12_invoice_line_items_undiscounted_matched_merchant.csv"
)

MERCHANT_PATH = os.path.join(
    BASE_ROOT,
    "Merchant Discount Detail.xlsx"
)

OUT_PATH = os.path.join(
    BASE_ROOT,
    r"RuralCo3\data_cleaning\12_invoice_line_items_undiscounted_matched_merchant_L2.csv"
)

# =========================================================
# Text normalization helpers
# =========================================================
def normalize_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(
        r"\b(ltd|limited|company|co|nz|head office|holdings|group)\b",
        " ",
        regex=True
    )
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def representative_token(name: str, min_len: int = 5):
    tokens = [t for t in name.split() if len(t) >= min_len]
    return max(tokens, key=len) if tokens else None

# =========================================================
# Load data (minimal columns only)
# =========================================================
invoice_cols = [
    "description",
    "matched_account_name",
    "matched_ats_number",
    "matched_discount_offered",
    "matched_discount_detail",
]

df = pd.read_csv(IN_PATH, usecols=invoice_cols, low_memory=False)
merchant_df = pd.read_excel(MERCHANT_PATH)

# =========================================================
# Prepare unmatched invoice descriptions
# =========================================================
unmatched_mask = df["matched_account_name"].isna()

df.loc[unmatched_mask, "description_clean"] = normalize_text(
    df.loc[unmatched_mask, "description"]
)

# =========================================================
# Prepare merchant tokens
# =========================================================
merchant_df["account_clean"] = normalize_text(merchant_df["Account Name"])
merchant_df["rep_token"] = merchant_df["account_clean"].apply(representative_token)

token_df = (
    merchant_df[
        ["ATS Number", "Account Name", "Discount Offered", "Discount Offered 2", "rep_token"]
    ]
    .dropna(subset=["rep_token"])
    .drop_duplicates(subset=["rep_token"])
)

# =========================================================
# Layer 2 matching: description-based
# =========================================================
still_unmatched = unmatched_mask.copy()
df["match_source"] = df.get("match_source")

for row in token_df.itertuples(index=False):
    token = row.rep_token

    hit_mask = (
        still_unmatched &
        df["description_clean"].str.contains(
            rf"\b{re.escape(token)}\b",
            regex=True,
            na=False
        )
    )

    if hit_mask.any():
        df.loc[hit_mask, "matched_ats_number"] = row[0]
        df.loc[hit_mask, "matched_account_name"] = row[1]
        df.loc[hit_mask, "matched_discount_offered"] = row[2]
        df.loc[hit_mask, "matched_discount_detail"] = row[3]
        df.loc[hit_mask, "match_source"] = "description"

        still_unmatched = still_unmatched & ~hit_mask

# =========================================================
# Save result
# =========================================================
df.to_csv(OUT_PATH, index=False)


