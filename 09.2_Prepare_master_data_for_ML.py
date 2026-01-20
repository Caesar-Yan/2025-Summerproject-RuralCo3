'''
Docstring for 09.2_Prepare_master_data_for_ML

this script reads in the group2 master data, which is on an account level.
it contains information on which accounts are late, how long it takes them to pay their bills, and how long they have been delinquent
this script handles processing this data, by excluding people who have never made a payment (unactiavted),
and makes some new variables which i don't think are getting used later to create the payment profile anyway

inputs:
- master_dataset_complete.parquet
    from group2

outputs:
- master_dataset_complete.csv
    into csv to be easier to inspect by eye
'''

import pandas as pd
from pathlib import Path

# ================================================================
# Configuration
# ================================================================

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
profile_dir.mkdir(exist_ok=True)
data_cleaning_dir = base_dir / "data_cleaning"

INPUT_FILE = r"t:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo2\Clean Code\master_dataset_complete.parquet"
OUTPUT_DIR = profile_dir
OUTPUT_FILE = profile_dir / "master_dataset_complete.csv"

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


# combined delinquent column
print("\nCreating derived feature...")
if 'is_delinquent' in df.columns and 'is_seriously_delinquent' in df.columns:
    df['Late'] = ((df['is_delinquent'] == 1) | (df['is_seriously_delinquent'] == 1)).astype(int)
    print(f"✓ Created 'Late' feature")
else:
    print("⚠ Cannot create Late - missing required columns")

print(f"\nSaving to: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False)

print(f"✓ Saved successfully!")
print(f"✓ File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")