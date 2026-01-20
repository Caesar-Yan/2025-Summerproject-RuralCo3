'''
01_read_datasets.py
this script is just reading in the datasets from ruralco, and use pickle to store in python dataframe form for future use

inputs:
ats_invoice.csv
ats_invoice_line_item.csv
invoice.csv
invoice_line_item.csv
Failed Accounts.xlsx
Merchant Discount Detail.xlsx

outputs:
output_dir / 'all_data.pkl'

'''

from pathlib import Path
import pandas as pd
import pickle

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo")
invoices_dir = base_dir / "Data provided by RuralCo 20251202/invoices_export/20251121"
data_dir = base_dir / "Data provided by RuralCo 20251202"

# Output directory for processed data
output_dir = base_dir / "Data provided by RuralCo 20251202\RuralCo3"
output_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist

# Invoice export files (CSV)
invoice_files = {
    'ats_invoice': invoices_dir / "ats_invoice.csv",
    'ats_invoice_line_item': invoices_dir / "ats_invoice_line_item.csv",
    'invoice': invoices_dir / "invoice.csv",
    'invoice_line_item': invoices_dir / "invoice_line_item.csv"
}

# Data directory files (Excel)
data_files = {
    'failed_accounts': data_dir / "Failed Accounts.xlsx",
    'merchant_discount': data_dir / "Merchant Discount Detail.xlsx"
}

# Load all data
def load_data():
    data = {}
    
    # Load CSV invoice files with low_memory=False to handle mixed types
    for name, filepath in invoice_files.items():
        if filepath.exists():
            print(f"Loading {name}...")
            data[name] = pd.read_csv(filepath, low_memory=False)
            print(f"  Loaded {name}: {len(data[name])} rows, {len(data[name].columns)} columns")
        else:
            print(f"Warning: {filepath} not found")
    
    # Load Excel data files
    for name, filepath in data_files.items():
        if filepath.exists():
            print(f"Loading {name}...")
            data[name] = pd.read_excel(filepath)
            print(f"  Loaded {name}: {len(data[name])} rows, {len(data[name].columns)} columns")
        else:
            print(f"Warning: {filepath} not found")
    
    return data

# Load all data
print("="*60)
print("Loading data...")
print("="*60)
all_data = load_data()

# Access individual dataframes
ats_invoice_df = all_data.get('ats_invoice')
ats_invoice_line_item_df = all_data.get('ats_invoice_line_item')
invoice_df = all_data.get('invoice')
invoice_line_item_df = all_data.get('invoice_line_item')
failed_accounts_df = all_data.get('failed_accounts')
merchant_discount_df = all_data.get('merchant_discount')

# Save the dictionary of dataframes to T: drive
output_file = output_dir / 'all_data.pkl'

print("\n" + "="*60)
print(f"Saving data to: {output_file}")
print("="*60)

try:
    # Try to delete the file first if it exists
    if output_file.exists():
        print("Removing existing file...")
        output_file.unlink()
    
    # Save the pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ Successfully saved data to {output_file}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
except Exception as e:
    print(f"✗ Error saving pickle file: {e}")
    print("\nData is loaded in memory but not saved to disk.")

print("\n" + "="*60)
print("Data loading complete!")
print("="*60)
print(f"Available dataframes: {list(all_data.keys())}")
print(f"\nDataframes can be accessed via:")
print("  - all_data dictionary: all_data['invoice']")
print("  - Individual variables: invoice_df, ats_invoice_df, etc.")
print(f"\nProcessed data directory: {output_dir}")
