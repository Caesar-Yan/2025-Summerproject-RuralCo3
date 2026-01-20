import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
output_dir = base_dir / "data_cleaning"
output_dir.mkdir(exist_ok=True)

# load data
df_invoice = pd.read_csv(output_dir / "invoice.csv", low_memory=False)
df_line = pd.read_csv(output_dir / "invoice_line_item.csv", low_memory=False)

print(df_line.columns.tolist())

# Select necessary columns for analysis
invoice_cols = [
    "id",
    "member_id",
    "date",
    "gross_transaction_amount",
    "discount_delta",
    "total_discount_delta",
    "process_status",
    "payment_processor"
]

line_cols = [
    "invoice_id",
    "line_gross_amt_received",   
    "quantity"
]

df_invoice = df_invoice[invoice_cols]
df_line = df_line[line_cols]

line_agg = (
    df_line
    .groupby("invoice_id", as_index=False)
    .agg(
        line_total_gross=("line_gross_amt_received", "sum"),
        line_count=("line_gross_amt_received", "count"),
        total_quantity=("quantity", "sum")
    )
)

df_merged = df_invoice.merge(
    line_agg,
    left_on="id",
    right_on="invoice_id",
    how="left"
)

df_merged.drop(columns=["invoice_id"], inplace=True)

df_merged["line_total_gross"] = df_merged["line_total_gross"].fillna(0)
df_merged["line_count"] = df_merged["line_count"].fillna(0)
df_merged["total_quantity"] = df_merged["total_quantity"].fillna(0)

# Missing discounts are treated as 0 (no discount)
df_merged["discount_delta"] = df_merged["discount_delta"].fillna(0)
df_merged["total_discount_delta"] = df_merged["total_discount_delta"].fillna(0)

# Date format
df_merged["date"] = pd.to_datetime(df_merged["date"], errors="coerce")

# Difference between line item amounts and invoice total
df_merged["amount_diff"] = (
    df_merged["gross_transaction_amount"] - df_merged["line_total_gross"]
)

df_merged["amount_diff"].describe()

# Save merged file
df_merged.to_csv(output_dir / "invoice_merged_clean.csv", index=False)

print(f"Merged file saved to: {output_dir / 'invoice_merged_clean.csv'}")