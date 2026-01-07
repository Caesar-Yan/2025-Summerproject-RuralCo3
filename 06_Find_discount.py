import pandas as pd

# Load invoice data
df_invoice = pd.read_csv("invoice.csv", low_memory=False)

# Parse transaction date (use 'date' as transaction date)
df_invoice["transaction_date"] = pd.to_datetime(
    df_invoice["date"], errors="coerce"
)

# Parse candidate date columns
candidate_date_cols = [
    "updated_at",
    "created_at",
    "extracted_invoice_date"
]

results = []

for col in candidate_date_cols:
    if col not in df_invoice.columns:
        continue

    # Parse datetime
    df_invoice[col] = pd.to_datetime(df_invoice[col], errors="coerce")

    #  Remove timezone information if present
    if df_invoice[col].dt.tz is not None:
        df_invoice[col] = df_invoice[col].dt.tz_localize(None)

    # Also ensure transaction_date is tz-naive
    if df_invoice["transaction_date"].dt.tz is not None:
        df_invoice["transaction_date"] = df_invoice["transaction_date"].dt.tz_localize(None)

    # Calculate day difference
    delta_days = (df_invoice[col] - df_invoice["transaction_date"]).dt.days

    stats = {
        "candidate_column": col,
        "non_null_pairs": delta_days.notna().sum(),
        "negative_days_ratio": (delta_days < 0).mean(),
        "median_days": delta_days.median(),
        "p95_days": delta_days.quantile(0.95)
    }

    results.append(stats)

df_time_test = pd.DataFrame(results)

print("\n=== Payment date candidates vs transaction date ===")
print(df_time_test.sort_values("negative_days_ratio"))

# Ensure datetime and timezone consistency
df_invoice["transaction_date"] = pd.to_datetime(
    df_invoice["date"], errors="coerce"
)
df_invoice["payment_date"] = pd.to_datetime(
    df_invoice["updated_at"], errors="coerce"
)

# Remove timezone info if present
if df_invoice["payment_date"].dt.tz is not None:
    df_invoice["payment_date"] = df_invoice["payment_date"].dt.tz_localize(None)

if df_invoice["transaction_date"].dt.tz is not None:
    df_invoice["transaction_date"] = df_invoice["transaction_date"].dt.tz_localize(None)

# Calculate days to payment
df_invoice["days_to_payment"] = (
    df_invoice["payment_date"] - df_invoice["transaction_date"]
).dt.days

# Identify late payments based on 14-day policy
EARLY_PAYMENT_DAYS = 14

df_invoice["paid_late"] = df_invoice["days_to_payment"] > EARLY_PAYMENT_DAYS

# Identify invoices where discount should be cancelled
df_invoice["discount_violation"] = (
    (df_invoice["discount_delta"] < 0) &
    (df_invoice["paid_late"])
)

direct_revenue_uplift = (
    -df_invoice.loc[df_invoice["discount_violation"], "discount_delta"].sum()
)

print(f"Direct revenue uplift (NZD): {direct_revenue_uplift:,.2f}")

print(df_invoice["discount_violation"].value_counts())