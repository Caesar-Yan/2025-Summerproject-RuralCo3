import pandas as pd
import os
import re

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
    r"RuralCo3\data_cleaning\12_invoice_line_items_undiscounted_matched_merchant_L1L2.csv"
)

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\b(ltd|limited|company|co|nz|head office|holdings|group)\b", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def representative_token(name: str, min_len: int = 5):
    tokens = [t for t in name.split() if len(t) >= min_len]
    return max(tokens, key=len) if tokens else None

df = pd.read_csv(IN_PATH, low_memory=False)
merchant_df = pd.read_excel(MERCHANT_PATH)

merchant_df["account_clean"] = normalize_series(merchant_df["Account Name"])
merchant_df["rep_token"] = merchant_df["account_clean"].apply(representative_token)

token_df = (
    merchant_df[["ATS Number", "Account Name", "Discount Offered", "Discount Offered 2", "rep_token"]]
    .dropna(subset=["rep_token"])
    .drop_duplicates(subset=["rep_token"])
)

l1_matched = df["matched_account_name"].notna()

df["description_clean"] = normalize_series(df.get("description"))

df["matched_ats_number_L2"] = pd.NA
df["matched_account_name_L2"] = pd.NA
df["matched_discount_offered_L2"] = pd.NA
df["matched_discount_detail_L2"] = pd.NA

still_unmatched = ~l1_matched

for row in token_df.itertuples(index=False):
    token = row.rep_token
    hit = still_unmatched & df["description_clean"].str.contains(rf"\b{re.escape(token)}\b", regex=True, na=False)

    if hit.any():
        df.loc[hit, "matched_ats_number_L2"] = row[0]
        df.loc[hit, "matched_account_name_L2"] = row[1]
        df.loc[hit, "matched_discount_offered_L2"] = row[2]
        df.loc[hit, "matched_discount_detail_L2"] = row[3]
        still_unmatched = still_unmatched & ~hit

df["match_layer"] = "unmatched"
df.loc[l1_matched, "match_layer"] = "L1"
df.loc[~l1_matched & df["matched_account_name_L2"].notna(), "match_layer"] = "L2"

df.to_csv(OUT_PATH, index=False)
