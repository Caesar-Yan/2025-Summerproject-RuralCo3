import pandas as pd
import numpy as np

# ================================================================
# Load the parquet file
# ================================================================
file_path_master = r"t:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo2\Clean Code\master_dataset_complete.parquet"
df2 = pd.read_parquet(file_path_master)

# ================================================================
# Select specific columns
# ================================================================
columns_to_select = [
    'account_id',
    'account_name',
    'has_due_date',
    'is_delinquent',
    'is_seriously_delinquent',
    'is_revolver',
    'delinquency_severity',
    'utilization_category',
    'revolver_intensity',
    'is_frequent_payer',
    'is_full_payer',
    'is_underpayer'
]

df_filtered = df2[columns_to_select]

# ================================================================
# Columns to analyze for value counts
# ================================================================
analysis_columns = [
    'has_due_date',
    'is_delinquent',
    'is_seriously_delinquent',
    'is_revolver',
    'delinquency_severity',
    'utilization_category',
    'revolver_intensity',
    'is_frequent_payer',
    'is_full_payer',
    'is_underpayer'
]

# ================================================================
# Display value counts for each column
# ================================================================
print("\n" + "="*70)
print("VALUE COUNTS FOR EACH COLUMN")
print("="*70)

for col in analysis_columns:
    print(f"\n{col}:")
    print("-" * 50)
    counts = df_filtered[col].value_counts(dropna=False)
    print(counts)
    print(f"Total: {counts.sum():,}")
    
    # Calculate percentages
    percentages = df_filtered[col].value_counts(normalize=True, dropna=False) * 100
    print("\nPercentages:")
    for val, pct in percentages.items():
        print(f"  {val}: {pct:.2f}%")

# ================================================================
# Create summary dataframe
# ================================================================
summary_data = []

for col in analysis_columns:
    counts = df_filtered[col].value_counts(dropna=False)
    for value, count in counts.items():
        pct = (count / len(df_filtered)) * 100
        summary_data.append({
            'column': col,
            'value': value,
            'count': count,
            'percentage': round(pct, 2)
        })

summary_df = pd.DataFrame(summary_data)

# Save to CSV
summary_df.to_csv('value_counts_summary.csv', index=False)
print("\n" + "="*70)
print("Saved value counts summary to: value_counts_summary.csv")


# ================================================================
# Sample and save dataframes
# ================================================================
# Sample size
SAMPLE_SIZE = 1000

# Sample with all columns
df_all_columns_sample = df2.sample(n=min(SAMPLE_SIZE, len(df2)), random_state=42)
df_all_columns_sample.to_csv('sample_all_columns.csv', index=False)
print(f"Saved all columns sample ({len(df_all_columns_sample):,} rows) to: sample_all_columns.csv")
print("="*70)

# ================================================================
# Average transaction size analysis by customer behavior categories
# ================================================================
print("\n" + "="*70)
print("AVERAGE TRANSACTION SIZE BY CUSTOMER BEHAVIOR CATEGORIES")
print("="*70)

# Columns to analyze
behavior_columns = [
    'is_delinquent',
    'is_seriously_delinquent',
    'is_revolver',
    'delinquency_severity',
    'utilization_category',
    'revolver_intensity',
    'is_frequent_payer',
    'is_full_payer',
    'is_underpayer'
]

# Store results
category_stats = []

for col in behavior_columns:
    print(f"\n{col}:")
    print("-" * 50)
    
    # Group by category and calculate statistics
    stats = df2.groupby(col)['avg_payment_per_txn'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).round(2)
    
    print(stats)
    
    # Add to results list
    for category_value in stats.index:
        category_stats.append({
            'behavior_column': col,
            'category_value': category_value,
            'count': stats.loc[category_value, 'count'],
            'mean': stats.loc[category_value, 'mean'],
            'median': stats.loc[category_value, 'median'],
            'std': stats.loc[category_value, 'std'],
            'min': stats.loc[category_value, 'min'],
            'max': stats.loc[category_value, 'max'],
            'q25': stats.loc[category_value, 'q25'],
            'q75': stats.loc[category_value, 'q75']
        })

# Create DataFrame and save
category_stats_df = pd.DataFrame(category_stats)
category_stats_df.to_csv('avg_transaction_by_behavior.csv', index=False)

print("\n" + "="*70)
print("Saved transaction size analysis to: avg_transaction_by_behavior.csv")
print("="*70)

# Print summary of key findings
print("\nKEY FINDINGS:")
print("-" * 50)
for col in behavior_columns:
    col_data = category_stats_df[category_stats_df['behavior_column'] == col]
    if len(col_data) > 0:
        highest = col_data.loc[col_data['mean'].idxmax()]
        print(f"\n{col}:")
        print(f"  Highest avg transaction: {highest['category_value']} (${highest['mean']:,.2f})")

import pandas as pd
import numpy as np

# ================================================================
# Load the parquet file
# ================================================================
file_path_master = r"t:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo2\Clean Code\master_dataset_complete.parquet"
df2 = pd.read_parquet(file_path_master)

# ================================================================
# Select specific columns
# ================================================================
columns_to_select = [
    'account_id',
    'account_name',
    'has_due_date',
    'is_delinquent',
    'is_seriously_delinquent',
    'is_revolver',
    'delinquency_severity',
    'utilization_category',
    'revolver_intensity',
    'is_frequent_payer',
    'is_full_payer',
    'is_underpayer'
]

df_filtered = df2[columns_to_select]

# ================================================================
# Columns to analyze for value counts
# ================================================================
analysis_columns = [
    'has_due_date',
    'is_delinquent',
    'is_seriously_delinquent',
    'is_revolver',
    'delinquency_severity',
    'utilization_category',
    'revolver_intensity',
    'is_frequent_payer',
    'is_full_payer',
    'is_underpayer'
]

# ================================================================
# Display value counts for each column
# ================================================================
print("\n" + "="*70)
print("VALUE COUNTS FOR EACH COLUMN")
print("="*70)

for col in analysis_columns:
    print(f"\n{col}:")
    print("-" * 50)
    counts = df_filtered[col].value_counts(dropna=False)
    print(counts)
    print(f"Total: {counts.sum():,}")
    
    # Calculate percentages
    percentages = df_filtered[col].value_counts(normalize=True, dropna=False) * 100
    print("\nPercentages:")
    for val, pct in percentages.items():
        print(f"  {val}: {pct:.2f}%")

# ================================================================
# Create summary dataframe
# ================================================================
summary_data = []

for col in analysis_columns:
    counts = df_filtered[col].value_counts(dropna=False)
    for value, count in counts.items():
        pct = (count / len(df_filtered)) * 100
        summary_data.append({
            'column': col,
            'value': value,
            'count': count,
            'percentage': round(pct, 2)
        })

summary_df = pd.DataFrame(summary_data)

# Save to CSV
summary_df.to_csv('value_counts_summary.csv', index=False)
print("\n" + "="*70)
print("Saved value counts summary to: value_counts_summary.csv")


# ================================================================
# Sample and save dataframes
# ================================================================
# Sample size
SAMPLE_SIZE = 1000

# Sample with all columns
df_all_columns_sample = df2.sample(n=min(SAMPLE_SIZE, len(df2)), random_state=42)
df_all_columns_sample.to_csv('sample_all_columns.csv', index=False)
print(f"Saved all columns sample ({len(df_all_columns_sample):,} rows) to: sample_all_columns.csv")
print("="*70)

# ================================================================
# Average transaction size analysis by customer behavior categories
# ================================================================
print("\n" + "="*70)
print("AVERAGE TRANSACTION SIZE BY CUSTOMER BEHAVIOR CATEGORIES")
print("="*70)

# Columns to analyze
behavior_columns = [
    'is_delinquent',
    'is_seriously_delinquent',
    'is_revolver',
    'delinquency_severity',
    'utilization_category',
    'revolver_intensity',
    'is_frequent_payer',
    'is_full_payer',
    'is_underpayer'
]

# Store results
category_stats = []

for col in behavior_columns:
    print(f"\n{col}:")
    print("-" * 50)
    
    # Group by category and calculate statistics
    stats = df2.groupby(col)['avg_payment_per_txn'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).round(2)
    
    print(stats)
    
    # Add to results list
    for category_value in stats.index:
        category_stats.append({
            'behavior_column': col,
            'category_value': category_value,
            'count': stats.loc[category_value, 'count'],
            'mean': stats.loc[category_value, 'mean'],
            'median': stats.loc[category_value, 'median'],
            'std': stats.loc[category_value, 'std'],
            'min': stats.loc[category_value, 'min'],
            'max': stats.loc[category_value, 'max'],
            'q25': stats.loc[category_value, 'q25'],
            'q75': stats.loc[category_value, 'q75']
        })

# Create DataFrame and save
category_stats_df = pd.DataFrame(category_stats)
category_stats_df.to_csv('avg_transaction_by_behavior.csv', index=False)

print("\n" + "="*70)
print("Saved transaction size analysis to: avg_transaction_by_behavior.csv")
print("="*70)

# Print summary of key findings
print("\nKEY FINDINGS:")
print("-" * 50)
for col in behavior_columns:
    col_data = category_stats_df[category_stats_df['behavior_column'] == col]
    if len(col_data) > 0:
        highest = col_data.loc[col_data['mean'].idxmax()]
        print(f"\n{col}:")
        print(f"  Highest avg transaction: {highest['category_value']} (${highest['mean']:,.2f})")

# ================================================================
# Summary statistics for payment_days column
# ================================================================
print("\n" + "="*70)
print("PAYMENT_DAYS SUMMARY STATISTICS")
print("="*70)

# Basic statistics
print("\nOverall Statistics:")
print("-" * 50)
payment_days_stats = df2['payment_days'].describe()
print(payment_days_stats)

# Additional statistics
print(f"\nAdditional Metrics:")
print(f"  Total non-null values: {df2['payment_days'].notna().sum():,}")
print(f"  Missing values: {df2['payment_days'].isna().sum():,}")
print(f"  Unique values: {df2['payment_days'].nunique():,}")
print(f"  Mode: {df2['payment_days'].mode().values[0] if len(df2['payment_days'].mode()) > 0 else 'N/A'}")
print(f"  Skewness: {df2['payment_days'].skew():.2f}")
print(f"  Kurtosis: {df2['payment_days'].kurtosis():.2f}")

# Percentiles
print(f"\nPercentile Distribution:")
print("-" * 50)
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = df2['payment_days'].quantile(p/100)
    print(f"  {p}th percentile: {value:.2f} days")

# ================================================================
# Payment days by customer behavior categories
# ================================================================
print("\n" + "="*70)
print("PAYMENT_DAYS BY CUSTOMER BEHAVIOR CATEGORIES")
print("="*70)

payment_days_stats = []

for col in behavior_columns:
    print(f"\n{col}:")
    print("-" * 50)
    
    # Group by category and calculate statistics
    stats = df2.groupby(col)['payment_days'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).round(2)
    
    print(stats)
    
    # Add to results list
    for category_value in stats.index:
        payment_days_stats.append({
            'behavior_column': col,
            'category_value': category_value,
            'count': stats.loc[category_value, 'count'],
            'mean': stats.loc[category_value, 'mean'],
            'median': stats.loc[category_value, 'median'],
            'std': stats.loc[category_value, 'std'],
            'min': stats.loc[category_value, 'min'],
            'max': stats.loc[category_value, 'max'],
            'q25': stats.loc[category_value, 'q25'],
            'q75': stats.loc[category_value, 'q75']
        })

# Create DataFrame and save
payment_days_stats_df = pd.DataFrame(payment_days_stats)
payment_days_stats_df.to_csv('payment_days_by_behavior.csv', index=False)

print("\n" + "="*70)
print("Saved payment_days analysis to: payment_days_by_behavior.csv")
print("="*70)

# Print summary of key findings
print("\nKEY FINDINGS - PAYMENT DAYS:")
print("-" * 50)
for col in behavior_columns:
    col_data = payment_days_stats_df[payment_days_stats_df['behavior_column'] == col]
    if len(col_data) > 0:
        fastest = col_data.loc[col_data['mean'].idxmin()]
        slowest = col_data.loc[col_data['mean'].idxmax()]
        print(f"\n{col}:")
        print(f"  Fastest payers: {fastest['category_value']} ({fastest['mean']:.1f} days avg)")
        print(f"  Slowest payers: {slowest['category_value']} ({slowest['mean']:.1f} days avg)")