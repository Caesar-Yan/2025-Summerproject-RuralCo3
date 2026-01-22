'''
Docstring for 06_match_merchant_id_to_invoices

this script aims to match up the merchant IDs from the marchant table to the invoices table

inputs:
- datetime_parsed_ats_invoice_line_item_df_transformed.csv
- datetime_parsed_invoice_line_item_df_transformed.csv

outputs:
- 

'''

import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
merchant_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/imported 5-12-2025")
output_dir = base_dir / "data_cleaning"

# Read the final CSVs from your original code
ats_df = pd.read_csv(output_dir / 'datetime_parsed_ats_invoice_line_item_df_transformed.csv')
invoice_df = pd.read_csv(output_dir / 'datetime_parsed_invoice_line_item_df_transformed.csv')
merchant_df = pd.read_excel(merchant_dir / 'Merchant_Discount_Detail.xlsx')

# Display head of all dataframes
print("=" * 80)
print("ATS Invoice Line Item DataFrame")
print("=" * 80)
print(ats_df.head())
print(f"\nShape: {ats_df.shape}")
print(f"Columns: {list(ats_df.columns)}")

print("\n" + "=" * 80)
print("Invoice Line Item DataFrame")
print("=" * 80)
print(invoice_df.head())
print(f"\nShape: {invoice_df.shape}")
print(f"Columns: {list(invoice_df.columns)}")

print("\n" + "=" * 80)
print("Merchant DataFrame")
print("=" * 80)
print(merchant_df.head())
print(f"\nShape: {merchant_df.shape}")
print(f"Columns: {list(merchant_df.columns)}")
