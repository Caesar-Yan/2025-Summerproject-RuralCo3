from pathlib import Path
import pandas as pd

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
output_dir = base_dir / "data_cleaning"

# Load no_flags_df
no_flags_df = pd.read_csv(output_dir / 'no_flags_df.csv')

print("="*80)
print("DISCOUNT_TYPE COLUMN ANALYSIS")
print("="*80)

# Check unique values
print("\nUnique values in discount_type:")
print(no_flags_df['discount_type'].unique())

# Value counts
print("\n" + "="*80)
print("Value counts:")
print(no_flags_df['discount_type'].value_counts(dropna=False))

# Check for numeric values
print("\n" + "="*80)
print("Can be converted to numeric:")
numeric_convertible = pd.to_numeric(no_flags_df['discount_type'], errors='coerce')
print(f"Rows with numeric discount_type: {numeric_convertible.notna().sum()}")

# Show examples of numeric values if any exist
if numeric_convertible.notna().any():
    print("\nExamples of numeric discount_type values:")
    numeric_examples = no_flags_df[numeric_convertible.notna()][['discount_type', 'line_gst_amt_received', 'line_net_amt_derived']].head(20)
    print(numeric_examples.to_string())