'''
Docstring for 06.1_unique_ids

this script is filtering for only unique ids from the 06_ script

inputs:
- 06_ats_line_item_filtered_ids.csv
- 06_invoice_line_item_filtered_ids.csv
- 06_ats_invoice_filtered_ids.csv
- 06_invoice_invoice_filtered_ids.csv

outputs:
- 06.1_shared_ids_analysis.csv
- 06.1_shared_ids_by_column.csv

'''

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
merchant_folder_dir = base_dir / "merchant"
output_path = merchant_folder_dir

# Read the final CSVs from your original code
ats_line_item_ids_df = pd.read_csv(merchant_folder_dir / '06_ats_line_item_filtered_ids.csv')
invoice_line_item_ids_df = pd.read_csv(merchant_folder_dir / '06_invoice_line_item_filtered_ids.csv')
ats_invoice_ids_df = pd.read_csv(merchant_folder_dir / '06_ats_invoice_filtered_ids.csv')
invoice_invoice_ids_df = pd.read_csv(merchant_folder_dir / '06_invoice_invoice_filtered_ids.csv')
master_dataset_ids_df = pd.read_csv(merchant_folder_dir / '06_master_dataset_filtered_ids.csv')

all_id_dfs = {
    'ats_line_item': ats_line_item_ids_df,
    'invoice_line_item': invoice_line_item_ids_df,
    'ats_invoice': ats_invoice_ids_df,
    'invoice_invoice': invoice_invoice_ids_df,
    'master_dataset': master_dataset_ids_df  # Add this
}

# Columns to remove
columns_to_remove = ['id', 'invoice_id', 'file_id', 'amtx_file_id', 'location_identifier']

# Remove specified columns from each dataframe
cleaned_dfs = {}
for df_name, df in all_id_dfs.items():
    # Get list of columns to drop that actually exist in this dataframe
    cols_to_drop = [col for col in columns_to_remove if col in df.columns]
    
    # Drop the columns
    cleaned_dfs[df_name] = df.drop(columns=cols_to_drop)
    
    print(f"\n{df_name.upper()}")
    print(f"Dropped columns: {cols_to_drop}")
    print(f"Remaining columns: {list(cleaned_dfs[df_name].columns)}")
    print(f"Shape: {cleaned_dfs[df_name].shape}")
    print(cleaned_dfs[df_name].head())

# Save all cleaned dataframes to CSV
for df_name, df in cleaned_dfs.items():
    output_filename = f"06.1_{df_name}_cleaned.csv"
    output_path = merchant_folder_dir / output_filename
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved {df_name} to: {output_path}")
    print(f"Shape: {df.shape}")

# Columns to exclude
columns_to_exclude = ['Unnamed: 0', 'dataframe_source']

print("=" * 80)
print("FINDING SHARED VALUES ACROSS DATAFRAMES")
print("=" * 80)

# For each column name, track which dataframes contain which values
column_value_tracker = defaultdict(lambda: defaultdict(set))

# Iterate through each dataframe
for df_name, df in all_id_dfs.items():
    print(f"\nProcessing {df_name}...")
    
    # Get columns to check (excluding unwanted columns)
    cols_to_check = [col for col in df.columns if col not in columns_to_exclude]
    
    print(f"  Checking columns: {cols_to_check}")
    
    # For each column, get unique values and track which dataframe they're from
    for col in cols_to_check:
        unique_values = df[col].dropna().unique()
        for value in unique_values:
            column_value_tracker[col][value].add(df_name)
        print(f"    {col}: {len(unique_values)} unique values")

# Now find values that appear in more than one dataframe
print("\n" + "=" * 80)
print("SHARED VALUES ANALYSIS")
print("=" * 80)

shared_results = []

for column_name, value_dict in column_value_tracker.items():
    print(f"\n{column_name.upper()}")
    print("-" * 80)
    
    for value, dataframes in value_dict.items():
        if len(dataframes) > 1:  # Value appears in more than one dataframe
            shared_results.append({
                'column_name': column_name,
                'value': value,
                'num_dataframes': len(dataframes),
                'dataframes': ', '.join(sorted(dataframes))
            })
            print(f"  Value: {value}")
            print(f"    Found in {len(dataframes)} dataframes: {', '.join(sorted(dataframes))}")

# Create summary dataframe
if shared_results:
    shared_df = pd.DataFrame(shared_results)
    shared_df = shared_df.sort_values(['column_name', 'num_dataframes'], ascending=[True, False])
    
    print("\n" + "=" * 80)
    print("SUMMARY OF SHARED VALUES")
    print("=" * 80)
    print(shared_df)
    
    # Save to CSV
    output_file = merchant_folder_dir / '06.1_shared_ids_analysis.csv'
    shared_df.to_csv(output_file, index=False)
    print(f"\nSaved shared values analysis to: {output_file}")
    print(f"Total shared values found: {len(shared_df)}")
    
    # Create summary by column
    column_summary = shared_df.groupby('column_name').agg({
        'value': 'count',
        'num_dataframes': 'max'
    }).rename(columns={'value': 'num_shared_values', 'num_dataframes': 'max_dataframes_per_value'})
    
    print("\n" + "=" * 80)
    print("SUMMARY BY COLUMN")
    print("=" * 80)
    print(column_summary)
    
    # Save column summary
    column_summary_file = merchant_folder_dir / '06.1_shared_ids_by_column.csv'
    column_summary.to_csv(column_summary_file)
    print(f"\nSaved column summary to: {column_summary_file}")
else:
    print("\nNo shared values found across dataframes!")

# Additional analysis: Show which dataframe pairs share the most values
print("\n" + "=" * 80)
print("DATAFRAME PAIR OVERLAP ANALYSIS")
print("=" * 80)

df_names = list(all_id_dfs.keys())
for i, df1_name in enumerate(df_names):
    for df2_name in df_names[i+1:]:
        overlap_count = sum(1 for result in shared_results 
                          if df1_name in result['dataframes'] and df2_name in result['dataframes'])
        if overlap_count > 0:
            print(f"{df1_name} & {df2_name}: {overlap_count} shared values")