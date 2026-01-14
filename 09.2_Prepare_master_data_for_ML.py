"""
09.2_Prepare_master_data_for_ML.py.py
=====================
Saves the master dataset parquet file as CSV for inspection.

Author: Chris
Date: January 2026
"""

import pandas as pd
from pathlib import Path

# ================================================================
# Configuration
# ================================================================
INPUT_FILE = r"t:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo2\Clean Code\master_dataset_complete.parquet"
OUTPUT_DIR = Path("Payment Profile")
OUTPUT_FILE = OUTPUT_DIR / "master_dataset_complete.csv"

# ================================================================
# Create output directory
# ================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Created directory: {OUTPUT_DIR}")

# ================================================================
# Load and save
# ================================================================
print(f"\nLoading: {INPUT_FILE}")
df = pd.read_parquet(INPUT_FILE)

print(f"✓ Loaded {len(df):,} records")
# print(f"✓ Columns: {list(df.columns)}")
print(f"✓ Shape: {df.shape}")

# ================================================================
# Exclude people who have never made a payment, 
# create avg_time_between_payments variable
# derive any other variables we might need
# ================================================================

# Removes accounts where payment_days = 0 or null
df = df[df['payment_days'] > 0].copy()

print(f"✓ Loaded {len(df):,} records")
# print(f"✓ Columns: {list(df.columns)}")
print(f"✓ Shape: {df.shape}")

# Create new feature: average time between payments
print("\nCreating derived feature...")
if 'account_tenure_months' in df.columns and 'payment_days' in df.columns:
    df['avg_time_between_payments'] = df['account_tenure_months'] / df['payment_days']
    print(f"✓ Created 'avg_time_between_payments' feature")
else:
    print("⚠ Cannot create avg_time_between_payments - missing required columns")

# InvoiceAmount creation
print("\nCreating derived feature...")
if 'avg_spend_per_txn' in df.columns and 'txn_per_month' in df.columns:
    df['InvoiceAmount'] = df['avg_spend_per_txn'] * df['txn_per_month']
    print(f"✓ Created 'InvoiceAmount' feature")
else:
    print("⚠ Cannot create InvoiceAmount - missing required columns")




print(f"\nSaving to: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False)

print(f"✓ Saved successfully!")
print(f"✓ File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")