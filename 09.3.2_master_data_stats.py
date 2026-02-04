'''
Docstring for 09.3.2_master_data_stats

im just getting some stats for the report here
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

INPUT_FILE = profile_dir / "master_dataset_complete.csv"

# ================================================================
# Load and Analyze Data
# ================================================================

# Load the dataset
df = pd.read_csv(INPUT_FILE)

# Analyze ytd_interest > 0
total_rows = len(df)
rows_with_ytd_interest_gte_0 = (df['ytd_interest'] != 0).sum()
percentage = (rows_with_ytd_interest_gte_0 / total_rows) * 100

print(f"Total rows: {total_rows}")
print(f"Rows with ytd_interest != 0: {rows_with_ytd_interest_gte_0}")
print(f"Percentage with ytd_interest >= 0: {percentage:.2f}%")


