# t:\projects\2025\RuralCo\Data provided by RuralCo 20251202\invoices_export\20251121
# ats_invoice
# ats_invoice_line_item
# invoice
# invoice_line_item

# t:\projects\2025\RuralCo\Data provided by RuralCo 20251202
# Failed Accounts
# Merchant Discount Detail

from pathlib import Path
import pandas as pd
import pickle

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo")
invoices_dir = base_dir / "Data provided by RuralCo 20251202/invoices_export/20251121"
data_dir = base_dir / "Data provided by RuralCo 20251202"

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
    
    # Load CSV invoice files
    for name, filepath in invoice_files.items():
        if filepath.exists():
            data[name] = pd.read_csv(filepath)
            print(f"Loaded {name}: {len(data[name])} rows, {len(data[name].columns)} columns")
        else:
            print(f"Warning: {filepath} not found")
    
    # Load Excel data files
    for name, filepath in data_files.items():
        if filepath.exists():
            data[name] = pd.read_excel(filepath)
            print(f"Loaded {name}: {len(data[name])} rows, {len(data[name].columns)} columns")
        else:
            print(f"Warning: {filepath} not found")
    
    return data

# Load all data
all_data = load_data()

# Access individual dataframes
ats_invoice_df = all_data.get('ats_invoice')
ats_invoice_line_item_df = all_data.get('ats_invoice_line_item')
invoice_df = all_data.get('invoice')
invoice_line_item_df = all_data.get('invoice_line_item')
failed_accounts_df = all_data.get('failed_accounts')
merchant_discount_df = all_data.get('merchant_discount')

# Save the dictionary of dataframes
with open('all_data.pkl', 'wb') as f:
    pickle.dump(all_data, f)