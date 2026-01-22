import pandas as pd
import os

# =========================================================
# PATH CONFIGURATION
# =========================================================
BASE_PATH = r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3\data_cleaning"

INPUT_FILE = "datetime_parsed_invoice_line_item_df_transformed.csv"
OUTPUT_FILE = "12_invoice_line_items_undiscounted_only.csv"

INPUT_PATH = os.path.join(BASE_PATH, INPUT_FILE)
OUTPUT_PATH = os.path.join(BASE_PATH, OUTPUT_FILE)

# =========================================================
# 1. Load data
# =========================================================
df = pd.read_csv(INPUT_PATH, low_memory=False)

# =========================================================
# 2. Basic cleaning
# =========================================================
df["undiscounted_price"] = pd.to_numeric(df["undiscounted_price"], errors="coerce")
df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")

# =========================================================
# 3. Filter undiscounted items
# =========================================================
undiscounted_df = df[
    df["undiscounted_price"].notna() &
    df["discounted_price"].notna() &
    (df["undiscounted_price"] == df["discounted_price"])
].copy()

# =========================================================
# 4. Select relevant columns
# =========================================================
cols_to_keep = [
    "invoice_id",
    "product_code",
    "description",
    "quantity",
    "undiscounted_price",
    "discounted_price",
    "merchant_identifier",
    "merchant_branch",
    "extras",
    "transaction_date",
    "invoice_period"
]

undiscounted_df = undiscounted_df[cols_to_keep]

# =========================================================
# 5. Sanity checks
# =========================================================
print("=" * 60)
print("UN-DISCOUNTED INVOICE LINE ITEMS")
print("=" * 60)
print(f"Total invoice line items: {len(df):,}")
print(f"Undiscounted line items: {len(undiscounted_df):,}")
print(f"Percentage undiscounted: {len(undiscounted_df) / len(df) * 100:.2f}%")

# =========================================================
# 6. Save output
# =========================================================
undiscounted_df.to_csv(OUTPUT_PATH, index=False)

print("=" * 60)
print(f"Saved file to:\n{OUTPUT_PATH}")
print("=" * 60)
