import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATA_PATH = "imputed_ats_invoice_line_item.csv"   # produced by 03_infer_values_ATS_line_item.py
ALL_DATA_PKL = "all_data.pkl"                     # contains ats_invoice table
RANDOM_STATE = 42
TEST_SIZE = 0.25

# If True, downsample line-items before monthly aggregation (helps speed)
USE_SAMPLE = False
SAMPLE_N = 200_000


# ------------------------------------------------------------
# Small debug helper
# ------------------------------------------------------------
def dbg(tag: str, df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[{tag}] rows={len(df):,} cols={df.shape[1]}")
    for c in ["cardholder_identifier", "member_id", "month", "inactive_next"]:
        if c in df.columns:
            if c in ["month"]:
                print(f"{c} non-null:", df[c].notna().sum())
                if df[c].notna().any():
                    print("month range:", df[c].min(), "to", df[c].max())
            elif c in ["inactive_next"]:
                print("inactive_next counts (incl NaN):")
                print(df[c].value_counts(dropna=False))
            else:
                print(f"{c} non-null:", df[c].notna().sum())
                print(f"{c} unique:", df[c].nunique(dropna=True))
    return df


def find_file_upwards(filename: str, start_dir: str = None, max_up: int = 6) -> str | None:
    """
    Find a file by walking upwards from start_dir (or current working dir).
    Useful when script is run from a different folder.
    """
    cur = Path(start_dir or os.getcwd()).resolve()
    for _ in range(max_up + 1):
        cand = cur / filename
        if cand.exists():
            return str(cand)
        cur = cur.parent
    return None


# ------------------------------------------------------------
# 0) Resolve paths robustly
# ------------------------------------------------------------
data_path_resolved = find_file_upwards(DATA_PATH)
if data_path_resolved is None:
    raise FileNotFoundError(
        f"Could not find {DATA_PATH} in current folder or parent folders. "
        f"Run the script from the repo folder that contains {DATA_PATH}, "
        f"or set DATA_PATH to an absolute path."
    )

pkl_path_resolved = find_file_upwards(ALL_DATA_PKL)
if pkl_path_resolved is None:
    raise FileNotFoundError(
        f"Could not find {ALL_DATA_PKL} in current folder or parent folders. "
        f"Run the script from the repo folder that contains {ALL_DATA_PKL}, "
        f"or set ALL_DATA_PKL to an absolute path."
    )

print("Using DATA_PATH:", data_path_resolved)
print("Using ALL_DATA_PKL:", pkl_path_resolved)


# ------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------
df = pd.read_csv(data_path_resolved, low_memory=False)
df = dbg("loaded raw line-items", df)

if USE_SAMPLE and len(df) > SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=RANDOM_STATE).copy()
    df = dbg(f"after sampling n={SAMPLE_N}", df)

required_cols = ["invoice_id", "discounted_price", "undiscounted_price"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {DATA_PATH}: {missing}")

# Basic numeric cleaning
for c in ["discounted_price", "undiscounted_price", "discount_offered", "quantity",
          "line_gross_amt_received", "line_net_amt_received", "unit_gross_amt_received"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# ------------------------------------------------------------
# 2) Map customer id (member_id) onto line-items using ats_invoice.id
#    This avoids a huge merge and is the correct relationship:
#    ats_invoice_line_item.invoice_id  ->  ats_invoice.id
# ------------------------------------------------------------
with open(pkl_path_resolved, "rb") as f:
    all_data = pickle.load(f)

ats_inv = all_data.get("ats_invoice")
if ats_inv is None:
    raise ValueError("ats_invoice not found inside all_data.pkl")

need_ats_cols = ["id", "member_id"]
missing_ats = [c for c in need_ats_cols if c not in ats_inv.columns]
if missing_ats:
    raise ValueError(f"ats_invoice missing columns: {missing_ats}")

ats_inv = ats_inv[["id", "member_id"]].copy()

# Normalize keys as strings to avoid dtype mismatch
df["invoice_id_norm"] = df["invoice_id"].astype(str)
ats_inv["id_norm"] = ats_inv["id"].astype(str)

# Build mapping (ats_invoice.id is unique)
invoice_to_member = pd.Series(ats_inv["member_id"].values, index=ats_inv["id_norm"])
df["member_id"] = df["invoice_id_norm"].map(invoice_to_member)

df = dbg("after mapping member_id", df)

# Keep only rows with customer id
df = df[df["member_id"].notna()].copy()
df = dbg("after dropping missing member_id", df)

# Use member_id as customer id for modelling
df["customer_id"] = df["member_id"].astype(str)


# ------------------------------------------------------------
# 3) Build a usable month variable
#    Prefer transaction_date, fallback to created_at
# ------------------------------------------------------------
date_col = None
if "transaction_date" in df.columns:
    date_col = "transaction_date"
elif "created_at" in df.columns:
    date_col = "created_at"
else:
    raise ValueError("No usable date column found. Need 'transaction_date' or 'created_at'.")

# Parse date
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
print(f"\nUsing date column: {date_col}")
print("date non-null after parse:", df[date_col].notna().sum())

# Keep only rows with usable date
df = df[df[date_col].notna()].copy()

# Month start timestamp (e.g., 2025-11-01)
df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
df = dbg("after month created", df)

if len(df) == 0:
    raise ValueError("No rows left after requiring both customer_id and a usable date.")


# ------------------------------------------------------------
# 4) Discount amount (line-level)
# ------------------------------------------------------------
df["discount_amount"] = (df["undiscounted_price"] - df["discounted_price"]).round(2)
df["discount_amount"] = df["discount_amount"].clip(lower=0)


# ------------------------------------------------------------
# 5) Aggregate to customer-month level (one row per customer per month)
# ------------------------------------------------------------
# Add transaction count and unique counts safely
df["txn_count"] = 1

agg_dict = {
    "discounted_price": "sum",
    "undiscounted_price": "sum",
    "discount_amount": "sum",
    "txn_count": "sum",
}

# Optional features if columns exist
if "merchant_identifier" in df.columns:
    agg_dict["merchant_identifier"] = pd.Series.nunique
if "invoice_id" in df.columns:
    agg_dict["invoice_id"] = pd.Series.nunique

df_month = (
    df.groupby(["customer_id", "month"], as_index=False)
      .agg(agg_dict)
      .rename(columns={
          "discounted_price": "monthly_spend_discounted",
          "undiscounted_price": "monthly_spend_undiscounted",
          "discount_amount": "monthly_discount_value",
          "txn_count": "monthly_txn_count",
          "merchant_identifier": "unique_merchants",
          "invoice_id": "unique_invoices",
      })
)

# Extra features
df_month["discount_rate"] = np.where(
    df_month["monthly_spend_undiscounted"] > 0,
    df_month["monthly_discount_value"] / df_month["monthly_spend_undiscounted"],
    0.0
)

df_month = dbg("after monthly aggregation", df_month)

if len(df_month) == 0:
    raise ValueError("Monthly aggregation produced 0 rows. Check customer_id/date parsing.")


# ------------------------------------------------------------
# 6) Create label inactive_next
#    Definition:
#    For each customer, if next observed month != current month + 1, label = 1 (inactive next month)
#    Last observed month per customer has unknown next-month outcome -> dropped
# ------------------------------------------------------------
df_month = df_month.sort_values(["customer_id", "month"]).copy()
df_month["next_month"] = df_month.groupby("customer_id")["month"].shift(-1)
df_month["month_plus_1"] = (df_month["month"].dt.to_period("M") + 1).dt.to_timestamp()

df_month["inactive_next"] = np.where(
    df_month["next_month"].notna() & (df_month["next_month"] != df_month["month_plus_1"]),
    1,
    np.where(df_month["next_month"].notna(), 0, np.nan)
)

df_month = dbg("after inactive_next computed", df_month)

df_model = df_month[df_month["inactive_next"].notna()].copy()
df_model["inactive_next"] = df_model["inactive_next"].astype(int)

df_model = dbg("after dropping NaN labels", df_model)

if len(df_model) == 0:
    raise ValueError(
        "No samples left after label construction. "
        "Most likely: each customer appears in only one month, or date parsing is too sparse."
    )

if df_model["inactive_next"].nunique() < 2:
    raise ValueError(
        f"Label has only one class (inactive_next unique={df_model['inactive_next'].nunique()}). "
        "Need both 0 and 1 to train."
    )


# ------------------------------------------------------------
# 7) Build features X and label y
# ------------------------------------------------------------
feature_cols = [
    "monthly_spend_discounted",
    "monthly_spend_undiscounted",
    "monthly_discount_value",
    "monthly_txn_count",
    "discount_rate",
]

if "unique_merchants" in df_model.columns:
    feature_cols.append("unique_merchants")
if "unique_invoices" in df_model.columns:
    feature_cols.append("unique_invoices")

X = df_model[feature_cols].copy()
y = df_model["inactive_next"].copy()

print("\nFinal modelling matrix:")
print("X shape:", X.shape)
print("y counts:")
print(y.value_counts(dropna=False))


# ------------------------------------------------------------
# 8) Train/test split (stratified)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ------------------------------------------------------------
# 9) Model pipeline (Logistic Regression baseline)
# ------------------------------------------------------------
numeric_features = feature_cols

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
    ],
    remainder="drop"
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

clf.fit(X_train, y_train)


# ------------------------------------------------------------
# 10) Evaluation
# ------------------------------------------------------------
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nModel evaluation:")
print("Accuracy:", round(acc, 4))
print("ROC-AUC:", round(auc, 4))
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# Save modelling table for audit
df_model.to_csv("model_table_customer_month.csv", index=False)
print("\nSaved: model_table_customer_month.csv")
print("Done.")