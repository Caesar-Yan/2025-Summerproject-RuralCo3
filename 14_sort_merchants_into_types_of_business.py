'''
Docstring for 14_sort_merchants_into_types_of_business

this script is for grouping merchants and finding an average discount rate for reverse engineering the undiscounted amounts

inputs:

outputs:



'''

'''
Docstring for 14_sort_merchants_into_types_of_business

This script groups merchants by business type and calculates average discount rates
for reverse engineering undiscounted transaction amounts in the RuralCo analysis.

inputs:
- merchant_discount_detail.csv: Contains merchant names and their discount rates

outputs:
- merchant_categories.csv: Merchants grouped by business type with average rates
- business_type_summary.csv: Summary statistics by business category

'''

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Set up paths
data_dir = Path('t:\projects\2025\RuralCo\Data provided by RuralCo 20251202')
output_dir = Path('t:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3\merchant')

# Load merchant discount data
print("Loading merchant discount data...")
merchant_df = pd.read_csv(data_dir / 'merchant_discount_detail.csv')

print(f"Loaded {len(merchant_df)} merchant records")
print(f"Columns: {merchant_df.columns.tolist()}")
print("\nFirst few rows:")
print(merchant_df.head())

