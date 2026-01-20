import pandas as pd
import numpy as np

# ================================================================
# Load the parquet file
# ================================================================
file_path = r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo2\decoded_v3.parquet"
file_path_master = r"t:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo2\Clean Code\matser_data_complete.parquet"

df = pd.read_parquet(file_path)
df2 = pd.read_parquet(file_path)

# ================================================================
# Basic information
# ================================================================
print(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ================================================================
# Column information
# ================================================================
print("\n" + "="*70)
print("COLUMNS AND DATA TYPES")
print("="*70)
print(df.dtypes)

# ================================================================
# Save column names to transactions_metadata.csv
# ================================================================
metadata = pd.DataFrame({
    'column_name': df.columns
})
metadata.to_csv('transactions_metadata.csv', index=False)
print("\nSaved column names to: transactions_metadata.csv")

# ================================================================
# Save first 100 rows to transactions_sample.csv
# ================================================================
df.head(100).to_csv('transactions_sample.csv', index=False)
print("Saved first 100 rows to: transactions_sample.csv")

