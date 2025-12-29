import pandas as pd
import numpy as np
import pickle

# ================================================================
# Load all_data.pkl so ATS invoice metadata (due_date, etc.) is available
# ================================================================
with open("all_data.pkl", "rb") as f:
    all_data = pickle.load(f)

# ================================================================
# 1. Load the imputed ATS line-item dataset
#    This file is produced by 03_infer_values_ATS_line_item.py
# ================================================================
ats_path = "imputed_ats_invoice_line_item.csv"
df = pd.read_csv(ats_path)

# Sanity check – ensure required columns exist
required_cols = ["undiscounted_price", "discounted_price", "flag"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in {ats_path}: {missing}")

print(f"Loaded dataset: {len(df):,} ATS line-items")

# ================================================================
# 2. Compute per-line discount amount
#    If undiscounted == discounted → the customer did not receive a discount
# ================================================================
df["discount_amount"] = df["undiscounted_price"] - df["discounted_price"]
df["discount_amount"] = df["discount_amount"].round(2)
df["discount_amount"] = df["discount_amount"].clip(lower=0)

benefit_rows = df[df["discount_amount"] > 0].copy()
print(f"Rows where discount_amount > 0: {len(benefit_rows):,}")

# ================================================================
# 3. Baseline scenario – Cancel ALL discounts (maximum theoretical benefit)
# ================================================================
total_benefit = benefit_rows["discount_amount"].sum()
avg_benefit_per_tx = benefit_rows["discount_amount"].mean()

print("\n=== DIRECT BENEFIT – Cancel ALL discounts (ATS customers) ===")
print(f"Total potential uplift: {total_benefit:,.2f}")
print(f"Avg uplift per affected line: {avg_benefit_per_tx:,.2f}")

# Optional breakdown exports
if "cardholder_identifier" in df.columns:
    df.groupby("cardholder_identifier")["discount_amount"].sum().sort_values(ascending=False).to_csv(
        "benefit_by_cardholder.csv"
    )
    print("Saved: benefit_by_cardholder.csv")

if "merchant_id" in df.columns:
    df.groupby("merchant_id")["discount_amount"].sum().sort_values(ascending=False).to_csv(
        "benefit_by_merchant.csv"
    )
    print("Saved: benefit_by_merchant.csv")

# ================================================================
# 4. Scenario-based calculation – discount cancellation only in selected cases
#    Requires a repayment variable: days_to_pay
# ================================================================
print("\n================= SCENARIO-BASED CALCULATION =================")

total_benefit_all = df["discount_amount"].sum()
print("Scenario A – Cancel ALL discounts:")
print(f"Total uplift: {total_benefit_all:,.2f}")
print(f"Average uplift per line: {df['discount_amount'].mean():,.2f}\n")

# Check if repayment timing exists
if "days_to_pay" not in df.columns:
    print("WARNING: 'days_to_pay' column not found — skipping repayment-based scenarios.\n")
else:
    # Scenario B – Cancel discount only where payment took more than 30 days
    late_mask = df["days_to_pay"] > 30
    df["benefit_late"] = 0.0
    df.loc[late_mask, "benefit_late"] = df.loc[late_mask, "discount_amount"]

    total_benefit_late = df["benefit_late"].sum()
    print("Scenario B – Cancel discount only where days_to_pay > 30:")
    print(f"Total uplift: {total_benefit_late:,.2f}")
    print(f"Share vs Scenario A: {(total_benefit_late / total_benefit_all) * 100:.2f}%\n")

    # Scenario C – Tiered cancellation (example strategy)
    # ≤14 days: recover 0%
    # 15–30 days: recover 50%
    # >30 days: recover 100%
    days = df["days_to_pay"]
    conditions = [
        (days <= 14),
        (days > 14) & (days <= 30),
        (days > 30)
    ]
    recovery_fraction = [0.0, 0.5, 1.0]

    df["recovery_rate"] = np.select(conditions, recovery_fraction, default=0.0)
    df["benefit_tiered"] = df["discount_amount"] * df["recovery_rate"]

    total_benefit_tiered = df["benefit_tiered"].sum()
    print("Scenario C – Tiered cancellation (0% ≤14d, 50% 15–30d, 100% >30d):")
    print(f"Total uplift: {total_benefit_tiered:,.2f}")
    print(f"Share vs Scenario A: {(total_benefit_tiered / total_benefit_all) * 100:.2f}%\n")

print("End of scenario-based calculation.")

# ================================================================
# 5. Extended scenario – Cancel discounts based on invoice due-date
#    Uses days_to_due instead of repayment (proxy measurement)
# ================================================================
print("\n================= EXTENDED – BASED ON DUE-DATE POLICY =================")

ats_inv = all_data.get("ats_invoice")
if ats_inv is None:
    print("ATS invoice table not found — due-date-based scenario skipped.\n")
else:
    # Parse dates
    ats_inv["date"] = pd.to_datetime(ats_inv["date"], errors="coerce")
    ats_inv["due_date"] = pd.to_datetime(ats_inv["due_date"], errors="coerce")

    # Days between invoice date and due date
    ats_inv["days_to_due"] = (ats_inv["due_date"] - ats_inv["date"]).dt.days

    # Prepare a subset with a renamed invoice id to avoid clashing with line-item id
    ats_inv_subset = ats_inv[["id", "days_to_due"]].rename(
        columns={"id": "ats_invoice_id"}
    )

    # Merge invoice-level timing to line-item DF using invoice_id
    df = df.merge(
        ats_inv_subset,
        left_on="invoice_id",
        right_on="ats_invoice_id",
        how="left"
    )

    print("\nDays_to_due summary:")
    print(df["days_to_due"].describe())

    # Tiered cancellation based on billing cycle length
    conditions = [
        (df["days_to_due"] <= 14),
        (df["days_to_due"] > 14) & (df["days_to_due"] <= 30),
        (df["days_to_due"] > 30)
    ]
    fraction = [0.0, 0.5, 1.0]

    df["recovery_due_rate"] = np.select(conditions, fraction, default=0.0)
    df["benefit_due_based"] = (df["discount_amount"] * df["recovery_due_rate"]).round(2)

    total_due_based = df["benefit_due_based"].sum()
    denom = total_benefit_all if total_benefit_all != 0 else 1.0

    print("\nScenario – Cancel discount based on billing cycle (due-date):")
    print(f"Total uplift: {total_due_based:,.2f}")
    print(f"Share vs Cancel-all: {(total_due_based / denom) * 100:.2f}%")

    df.to_csv("benefit_with_due_scenarios.csv", index=False)
    print("Saved: benefit_with_due_scenarios.csv\n")

print("================= END EXTENDED BENEFIT CALCULATION =================")

df = pd.read_csv("imputed_ats_invoice_line_item.csv")

print(df["discounted_price"].describe())
print(df["undiscounted_price"].describe())