'''
Docstring for 08.1_Debit_only

this script is just for filtering out negative undiscounted_price entries
basically a direct continuation of 08_, and will overwrite the same csv. 
if you want to switch, just run 08_ and overwrite it.

inputs: 
- ats_grouped_by_invoice_transformed.csv
- invoice_grouped_by_invoice_transformed.csv

outputs:
- ats_grouped_by_invoice_transformed.csv
- invoice_grouped_by_invoice_transformed.csv

'''

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
output_dir = base_dir / "data_cleaning"

# ================================================================
# Load the two imputed line-item datasets
# ================================================================
ats_path = "ats_grouped_by_invoice_transformed.csv"
invoice_path = "invoice_grouped_by_invoice_transformed.csv"

ats = pd.read_csv(output_dir / ats_path)
invoice = pd.read_csv(output_dir / invoice_path)

print(f"Loaded ATS dataset: {len(ats):,} line-items")
print(f"Loaded Invoice dataset: {len(invoice):,} line-items")

# filter out all rows where total_undiscounted_price < 0
# filter out all rows where total_undiscounted_price < 0

print("\n" + "="*80)
print("FILTERING OUT NEGATIVE UNDISCOUNTED_PRICE ENTRIES")
print("="*80)

# Filter ATS dataframe
print(f"\nATS DataFrame:")
print(f"  Before filtering: {len(ats):,} rows")
negative_price_ats = (ats['total_undiscounted_price'] < 0).sum()
print(f"  Rows with total_undiscounted_price < 0: {negative_price_ats:,}")

ats = ats[ats['total_undiscounted_price'] >= 0].copy()
print(f"  After filtering: {len(ats):,} rows")
print(f"  Removed: {negative_price_ats:,} rows")

# Filter Invoice dataframe
print(f"\nInvoice DataFrame:")
print(f"  Before filtering: {len(invoice):,} rows")
negative_price_invoice = (invoice['total_undiscounted_price'] < 0).sum()
print(f"  Rows with total_undiscounted_price < 0: {negative_price_invoice:,}")

invoice = invoice[invoice['total_undiscounted_price'] >= 0].copy()
print(f"  After filtering: {len(invoice):,} rows")
print(f"  Removed: {negative_price_invoice:,} rows")

print("="*80)

# Save the filtered dataframes (overwriting the original files)
ats.to_csv(output_dir / ats_path, index=False)
invoice.to_csv(output_dir / invoice_path, index=False)

print("\n" + "="*80)
print("FILTERING COMPLETE - FILES OVERWRITTEN")
print("="*80)
print(f"Updated ATS data saved to: {ats_path}")
print(f"  Final row count: {len(ats):,}")
print(f"Updated Invoice data saved to: {invoice_path}")
print(f"  Final row count: {len(invoice):,}")

# Show summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print("\nATS total_undiscounted_price distribution:")
print(ats['total_undiscounted_price'].describe())

print("\nInvoice total_undiscounted_price distribution:")
print(invoice['total_undiscounted_price'].describe())

print("\n" + "="*80)
print("ALL PROCESSING COMPLETE!")
print("="*80)