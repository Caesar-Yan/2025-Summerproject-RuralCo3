'''
Docstring for 09.3.1_Examine_total_late_rate

this script is just for examining the percentage of accounts that have a non-zero ytd_interest as shown in master_dataset

inputs:
- master_dataset_complete.csv

outputs:
- 09.3.1_summary.csv
- 09.3.1_comparison_plot.png


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os

# ================================================================
# Configuration
# ================================================================
# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
data_cleaning_dir = base_dir / "data_cleaning"

INPUT_FILE = profile_dir / "master_dataset_complete.csv"
OUTPUT_DIR = profile_dir

# Load the master dataset
print("="*80)
print("LOADING MASTER DATASET")
print("="*80)
df = pd.read_csv(INPUT_FILE)
print(f"Loaded master dataset: {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")

# ================================================================
# Filter out "Closed" industry accounts
# ================================================================
print("\n" + "="*80)
print("FILTERING OUT 'CLOSED' INDUSTRY ACCOUNTS")
print("="*80)

print(f"\nBefore filtering: {len(df):,} rows")
closed_count = (df['industry'] == 'Closed').sum()
print(f"Rows with industry = 'Closed': {closed_count:,}")

df = df[df['industry'] != 'Closed'].copy()
print(f"After filtering: {len(df):,} rows")
print(f"Removed: {closed_count:,} rows")

print("\nRemaining industry distribution:")
print(df['industry'].value_counts())
print("="*80)

# Initialize results dictionary
results = {}

# ================================================================
# Analysis 1: Percentage of accounts with non-zero ytd_interest
# ================================================================
print("\n" + "="*80)
print("ANALYSIS 1: NON-ZERO YTD_INTEREST (ALL VALUES)")
print("="*80)

total_accounts = len(df)
non_zero_interest = (df['ytd_interest'] != 0).sum()
zero_interest = (df['ytd_interest'] == 0).sum()
percentage_non_zero = (non_zero_interest / total_accounts) * 100

print(f"\nTotal accounts: {total_accounts:,}")
print(f"Accounts with non-zero ytd_interest: {non_zero_interest:,}")
print(f"Accounts with zero ytd_interest: {zero_interest:,}")
print(f"Percentage with non-zero ytd_interest: {percentage_non_zero:.2f}%")

results['analysis_1'] = {
    'description': 'Non-zero ytd_interest (all values)',
    'total_accounts': total_accounts,
    'non_zero_count': non_zero_interest,
    'zero_count': zero_interest,
    'percentage_non_zero': percentage_non_zero
}

# Show distribution
print(f"\nYTD Interest distribution:")
print(df['ytd_interest'].describe())

# ================================================================
# Analysis 2: Percentage of accounts with non-zero ytd_interest (excluding negatives)
# ================================================================
print("\n" + "="*80)
print("ANALYSIS 2: NON-ZERO YTD_INTEREST (EXCLUDING NEGATIVE VALUES)")
print("="*80)

# Filter out negative values
df_positive = df[df['ytd_interest'] >= 0].copy()
total_accounts_positive = len(df_positive)
non_zero_interest_positive = (df_positive['ytd_interest'] > 0).sum()
zero_interest_positive = (df_positive['ytd_interest'] == 0).sum()
percentage_non_zero_positive = (non_zero_interest_positive / total_accounts_positive) * 100

negative_count = (df['ytd_interest'] < 0).sum()

print(f"\nTotal accounts (after excluding negatives): {total_accounts_positive:,}")
print(f"Negative ytd_interest accounts excluded: {negative_count:,}")
print(f"Accounts with non-zero ytd_interest: {non_zero_interest_positive:,}")
print(f"Accounts with zero ytd_interest: {zero_interest_positive:,}")
print(f"Percentage with non-zero ytd_interest: {percentage_non_zero_positive:.2f}%")

results['analysis_2'] = {
    'description': 'Non-zero ytd_interest (excluding negatives)',
    'total_accounts': total_accounts_positive,
    'negative_excluded': negative_count,
    'non_zero_count': non_zero_interest_positive,
    'zero_count': zero_interest_positive,
    'percentage_non_zero': percentage_non_zero_positive
}

print(f"\nYTD Interest distribution (positive values only):")
print(df_positive['ytd_interest'].describe())

# ================================================================
# Analysis 3: Percentage with non-zero ytd_interest (absolute value, treating >0.05 as 0)
# ================================================================
print("\n" + "="*80)
print("ANALYSIS 3: NON-ZERO YTD_INTEREST (ABS VALUE, THRESHOLD = 0.05)")
print("="*80)

# Take absolute value and apply threshold
df_threshold = df.copy()
df_threshold['ytd_interest_abs'] = df_threshold['ytd_interest'].abs()
df_threshold['ytd_interest_thresholded'] = df_threshold['ytd_interest_abs'].apply(
    lambda x: 0 if x <= 0.05 else x
)

total_accounts_threshold = len(df_threshold)
non_zero_threshold = (df_threshold['ytd_interest_thresholded'] > 0).sum()
zero_threshold = (df_threshold['ytd_interest_thresholded'] == 0).sum()
percentage_non_zero_threshold = (non_zero_threshold / total_accounts_threshold) * 100

# Count how many were converted to zero by the threshold
converted_to_zero = ((df_threshold['ytd_interest_abs'] > 0) & 
                     (df_threshold['ytd_interest_abs'] <= 0.05)).sum()

print(f"\nTotal accounts: {total_accounts_threshold:,}")
print(f"Accounts with abs(ytd_interest) <= 0.05 (treated as zero): {converted_to_zero:,}")
print(f"Accounts with non-zero ytd_interest (after threshold): {non_zero_threshold:,}")
print(f"Accounts with zero ytd_interest (after threshold): {zero_threshold:,}")
print(f"Percentage with non-zero ytd_interest: {percentage_non_zero_threshold:.2f}%")

results['analysis_3'] = {
    'description': 'Non-zero ytd_interest (abs value, threshold=0.05)',
    'total_accounts': total_accounts_threshold,
    'converted_to_zero_by_threshold': converted_to_zero,
    'non_zero_count': non_zero_threshold,
    'zero_count': zero_threshold,
    'percentage_non_zero': percentage_non_zero_threshold
}

print(f"\nYTD Interest distribution (after abs and threshold):")
print(df_threshold['ytd_interest_thresholded'].describe())

# ================================================================
# Save Results Summary
# ================================================================
print("\n" + "="*80)
print("SAVING RESULTS SUMMARY")
print("="*80)

# Convert results to DataFrame
summary_data = []
for key, values in results.items():
    row = {'analysis': key}
    row.update(values)
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)

# Save to CSV
output_file = OUTPUT_DIR / "09.3.1_summary.csv"
summary_df.to_csv(output_file, index=False)

print(f"\nSummary saved to: {output_file}")
print("\nSummary DataFrame:")
print(summary_df.to_string(index=False))

# ================================================================
# Additional Visualization
# ================================================================
print("\n" + "="*80)
print("CREATING COMPARISON VISUALIZATION")
print("="*80)

# Create bar chart comparing the three analyses
fig, ax = plt.subplots(figsize=(10, 6))

analyses = ['All Values', 'Excl. Negatives', 'Abs + Threshold']
percentages = [
    results['analysis_1']['percentage_non_zero'],
    results['analysis_2']['percentage_non_zero'],
    results['analysis_3']['percentage_non_zero']
]

bars = ax.bar(analyses, percentages, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xlabel('Analysis Method', fontsize=12)
ax.set_title('Percentage of Accounts with Non-Zero YTD Interest', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)

# Add value labels on bars
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plot_file = OUTPUT_DIR / "09.3.1_comparison_plot.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: {plot_file}")
plt.close()

print("\n" + "="*80)
print("ALL PROCESSING COMPLETE!")
print("="*80)





