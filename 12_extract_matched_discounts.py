# temporary note:
# this script has not yet been tested due to temporary inability
# to connect to the remote host. this note will be removed after testing.

'''
Docstring for 12_extract_matched_discounts

this script extracts merchant discount information for invoice line items
that have been matched to known discount merchants (L1 and L2).
the prices in the invoice data are already discounted, so this script
parses discount text from Merchant Discount Detail and back-calculates
the implied undiscounted price and discount amount.

only lines matched via L1 or L2 are included in the calculation.
discount rates are parsed conservatively (lower bound for ranges,
explicit value for "up to", and no discount entries are ignored).

inputs:
- 12_invoice_line_items_undiscounted_matched_merchant_L1L2.csv

outputs:
- 12_matched_discount_extracted_line_level.csv
    line-level discount rate, implied undiscounted price,
    and implied discount amount for matched invoice lines
- 12_matched_discount_extracted_merchant_summary.csv
    merchant-level summary of implied discount amounts
'''


import pandas as pd
import os
import re
import numpy as np

BASE_ROOT = r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202"

IN_PATH = os.path.join(
    BASE_ROOT,
    r"RuralCo3\data_cleaning\12_invoice_line_items_undiscounted_matched_merchant_L1L2.csv"
)

OUT_LINE_PATH = os.path.join(
    BASE_ROOT,
    r"RuralCo3\data_cleaning\12_matched_discount_extracted_line_level.csv"
)

OUT_MERCHANT_PATH = os.path.join(
    BASE_ROOT,
    r"RuralCo3\data_cleaning\12_matched_discount_extracted_merchant_summary.csv"
)

def parse_conservative_rate(text):
    """
    Returns a conservative percent-off rate as a decimal (e.g., 0.05 for 5%).
    Rules:
      - '5-12.5%' -> 0.05
      - 'Up to 10%' -> 0.10
      - '25.25% average' -> 0.2525
      - 'No discount' -> NaN
    """
    if pd.isna(text):
        return np.nan

    s = str(text).lower()

    if "no discount" in s:
        return np.nan

    values = re.findall(r"(\d+(?:\.\d+)?)\s*%", s)
    if not values:
        return np.nan

    nums = [float(v) for v in values]

    if "-" in s or " to " in s:
        return min(nums) / 100.0

    return nums[0] / 100.0

df = pd.read_csv(IN_PATH, low_memory=False)

# Final matched fields (L1 + L2)
df["matched_account_final"] = df["matched_account_name"]
df.loc[df["matched_account_final"].isna(), "matched_account_final"] = df["matched_account_name_L2"]

df["matched_discount_text_final"] = df["matched_discount_offered"]
df.loc[df["matched_discount_text_final"].isna(), "matched_discount_text_final"] = df["matched_discount_offered_L2"]

eligible = df["match_layer"].isin(["L1", "L2"])

# Ensure numeric price
df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")

# Parse discount rate
df["discount_rate_conservative"] = np.nan
df.loc[eligible, "discount_rate_conservative"] = (
    df.loc[eligible, "matched_discount_text_final"].apply(parse_conservative_rate)
)

# Compute implied original (undiscounted) price and discount amount
# discounted_price is after discount: discounted = original * (1-r)
valid = eligible & df["discount_rate_conservative"].notna() & df["discounted_price"].notna()
r = df.loc[valid, "discount_rate_conservative"]

df["implied_undiscounted_price"] = np.nan
df["implied_discount_amount"] = np.nan

df.loc[valid, "implied_undiscounted_price"] = df.loc[valid, "discounted_price"] / (1.0 - r)
df.loc[valid, "implied_discount_amount"] = df.loc[valid, "implied_undiscounted_price"] - df.loc[valid, "discounted_price"]

# Save line-level output
df.to_csv(OUT_LINE_PATH, index=False)

# Merchant summary (matched only, where rate parsed)
summary = (
    df.loc[df["implied_discount_amount"].notna()]
    .groupby(["matched_account_final", "match_layer"], as_index=False)
    .agg(
        matched_lines=("implied_discount_amount", "count"),
        total_implied_discount=("implied_discount_amount", "sum"),
        avg_discount_rate=("discount_rate_conservative", "mean"),
        total_discounted_spend=("discounted_price", "sum"),
        total_implied_undiscounted=("implied_undiscounted_price", "sum"),
    )
    .sort_values("total_implied_discount", ascending=False)
)

summary.to_csv(OUT_MERCHANT_PATH, index=False)
