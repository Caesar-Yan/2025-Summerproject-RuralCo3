'''
Docstring for 06_match_merchant_id_to_invoices

this script aims to match up the merchant IDs from the marchant table to the invoices table

inputs:
- datetime_parsed_ats_invoice_line_item_df_transformed.csv
- datetime_parsed_invoice_line_item_df_transformed.csv

outputs:
- 06_merchant_filtered_ids.csv
- 06_ats_invoice_filtered_ids.csv
- 06_ats_invoice_line_items_filtered_dfs.csv
- 06_invoice_invoice_filtered_ids.csv
- 06_invoice_line_items_filtered_dfs.csv
'''

import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
merchant_df_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/imported 5-12-2025")
data_cleaning_dir = base_dir / "data_cleaning"
parent_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/imported 5-12-2025/invoices_export/20251121")
payment_profile_dir = base_dir / "payment_profile"
merchant_folder_dir = base_dir / "merchant"
# Create merchant directory if it doesn't exist
merchant_folder_dir.mkdir(parents=True, exist_ok=True)

# Read the final CSVs from your original code
ats_df = pd.read_csv(data_cleaning_dir / 'datetime_parsed_ats_invoice_line_item_df_transformed.csv')
invoice_df = pd.read_csv(data_cleaning_dir / 'datetime_parsed_invoice_line_item_df_transformed.csv')
ats_invoice_df = pd.read_csv(parent_dir / 'ats_invoice.csv')
invoice_invoice_df = pd.read_csv(parent_dir / 'invoice.csv')

master_dataset_df = pd.read_csv(payment_profile_dir / 'master_dataset_complete.csv')
merchant_df = pd.read_excel(merchant_df_dir / 'Merchant_Discount_Detail.xlsx')

# Create a dictionary to iterate through both dataframes
all_invoices_df = {
    'ats_line_item': ats_df,
    'ats_invoice': ats_invoice_df,
    'invoice_line_item': invoice_df,
    'invoice_invoice': invoice_invoice_df,
}

all_dfs = {
    'ats_line_item': ats_df,
    'ats_invoice': ats_invoice_df,
    'invoice_line_item': invoice_df,
    'invoice_invoice': invoice_invoice_df,
    'master_data': master_dataset_df,
    'merchant': merchant_df
}

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

# Create filtered dataframes with only 'id' columns + "Unnamed: 0"
filtered_invoices_df = {}

for df_name, df in all_invoices_df.items():
    # Get columns with 'id' in the name (case-insensitive)
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    
    # Add "Unnamed: 0" if it exists
    columns_to_keep = []
    if "Unnamed: 0" in df.columns:
        columns_to_keep.append("Unnamed: 0")
    
    columns_to_keep.extend(id_columns)
    
    # Create filtered dataframe
    filtered_invoices_df[df_name] = df[columns_to_keep].copy()
    
    # Add dataframe name as first column
    filtered_invoices_df[df_name].insert(0, 'dataframe_source', df_name)
    
    print(f"\n{df_name.upper()} - Filtered columns: {['dataframe_source'] + columns_to_keep}")
    print(filtered_invoices_df[df_name].head())

# Save filtered dataframes to CSV
for df_name, df in filtered_invoices_df.items():
    output_filename = f"06_{df_name}_filtered_ids.csv"
    output_path = merchant_folder_dir / output_filename
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved {df_name} filtered dataframe to: {output_path}")
    print(f"Shape: {df.shape}")

# Filter merchant dataframe with 'id' columns + specific columns
merchant_id_columns = [col for col in merchant_df.columns if 'id' in col.lower()]

# Start with the required columns in order
merchant_columns_to_keep = ['ATS Number', 'Account Name']

# Add the id columns
merchant_columns_to_keep.extend(merchant_id_columns)

# Create filtered merchant dataframe
filtered_merchant_df = merchant_df[merchant_columns_to_keep].copy()

# Add dataframe name as first column
filtered_merchant_df.insert(0, 'dataframe_source', 'merchant')

print(f"\nMERCHANT - Filtered columns: {['dataframe_source'] + merchant_columns_to_keep}")
print(filtered_merchant_df.head())

# Save filtered merchant dataframe to CSV
merchant_output_filename = "06_merchant_filtered_ids.csv"
merchant_output_path = merchant_folder_dir / merchant_output_filename

filtered_merchant_df.to_csv(merchant_output_path, index=False)
print(f"\nSaved merchant filtered dataframe to: {merchant_output_path}")
print(f"Shape: {filtered_merchant_df.shape}")

# Then at the end, after the merchant filtering section, add:

# Filter master_dataset with 'id' columns + 'account_name'
master_id_columns = [col for col in master_dataset_df.columns if 'id' in col.lower()]

# Start with account_name as the required column
master_columns_to_keep = ['account_name']

# Add the id columns
master_columns_to_keep.extend(master_id_columns)

# Create filtered master_dataset dataframe
filtered_master_dataset_df = master_dataset_df[master_columns_to_keep].copy()

# Add dataframe name as first column
filtered_master_dataset_df.insert(0, 'dataframe_source', 'master_dataset')

print(f"\nMASTER_DATASET - Filtered columns: {['dataframe_source'] + master_columns_to_keep}")
print(filtered_master_dataset_df.head())

# Save filtered master_dataset dataframe to CSV
master_dataset_output_filename = "06_master_dataset_filtered_ids.csv"
master_dataset_output_path = merchant_folder_dir / master_dataset_output_filename

filtered_master_dataset_df.to_csv(master_dataset_output_path, index=False)
print(f"\nSaved master_dataset filtered dataframe to: {master_dataset_output_path}")
print(f"Shape: {filtered_master_dataset_df.shape}")