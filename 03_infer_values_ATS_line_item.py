import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

imputed_ats_invoice_line_item_df = ats_invoice_line_item_df

def get_non_null_percentage(df, column_name):
    """
    Calculate the percentage of non-null values in a column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    column_name : str
        Name of the column to check
        
    Returns:
    --------
    str
        Formatted string showing non-null count and percentage
    """
    total_rows = len(df)
    non_null_rows = df[column_name].notna().sum()
    non_null_percentage = (non_null_rows / total_rows) * 100
    
    return f"Non-null {column_name} values: {non_null_rows:,} ({non_null_percentage:.2f}%)"

def check_diff(df, col1, col2, diff_col_name=None):
    """
    Calculate the difference between two columns and print summary statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the columns
    col1 : str
        Name of the first column
    col2 : str
        Name of the second column
    diff_col_name : str, optional
        Name for the difference column. If None, defaults to 'diff_{col1}_{col2}'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with the new difference column added
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Set default column name if not provided
    if diff_col_name is None:
        diff_col_name = f'diff_{col1}_{col2}'
    
    # Calculate the difference
    df_copy[diff_col_name] = df_copy[col1] - df_copy[col2]
    
    # Calculate summary statistics
    mean_val = df_copy[diff_col_name].mean()
    std_val = df_copy[diff_col_name].std()
    null_count = df_copy[diff_col_name].isna().sum()
    max_val = df_copy[diff_col_name].max()
    min_val = df_copy[diff_col_name].min()
    
    # Print summary statistics
    print(f"\nSummary statistics for {diff_col_name} ({col1} - {col2}):")
    print(f"Mean: {mean_val:.4f}")
    print(f"Standard Deviation: {std_val:.4f}")
    print(f"Number of null values: {null_count:,}")
    print(f"Highest value: {max_val:.4f}")
    print(f"Lowest value: {min_val:.4f}")
    
    return df_copy

def merge_updates_to_main_df(main_df, update_df, columns_to_update, id_column='Unnamed: 0'):
    """
    Merge updates from a filtered dataframe back into the main dataframe.
    
    Parameters:
    -----------
    main_df : pandas.DataFrame
        The main dataframe to update
    update_df : pandas.DataFrame
        The dataframe containing updated values
    columns_to_update : list
        List of column names to update
    id_column : str
        The identifier column to merge on (default: 'Unnamed: 0')
    
    Returns:
    --------
    None (updates main_df in place)
    """
    for col in columns_to_update:
        temp_merge = main_df[[id_column, col]].merge(
            update_df[[id_column, col]].rename(columns={col: f'{col}_new'}),
            on=id_column,
            how='left'
        )
        main_df[col] = temp_merge[col].fillna(temp_merge[f'{col}_new'])
    
    # Print completion percentages
    for col in columns_to_update:
        print(get_non_null_percentage(main_df, col))


# ============================================================================================================
# UNIT GROSS AMOUNT IMPUTATION
# ============================================================================================================

print(imputed_ats_invoice_line_item_df.columns)

# Imputing unit_gross_amt_derived
# Want to check that unit_gross_amt_derived == line_gross_amt_received / quantity
# Get rows where unit_gross_amt_derived is null
mask_null = imputed_ats_invoice_line_item_df['unit_gross_amt_derived'].isnull()

# Calculate new column imp_unit_gross_amt_derived for ALL rows
imputed_ats_invoice_line_item_df['imp_unit_gross_amt_derived'] = (
    imputed_ats_invoice_line_item_df['line_gross_amt_received'] / 
    imputed_ats_invoice_line_item_df['quantity']
).round(2)

# Check if imp_unit_gross_amt_derived matches unit_gross_amt_derived where not null
comparison = imputed_ats_invoice_line_item_df.loc[~mask_null, 'imp_unit_gross_amt_derived'] == \
             imputed_ats_invoice_line_item_df.loc[~mask_null, 'unit_gross_amt_derived']

print(f"Match rate: {comparison.sum() / len(comparison) * 100:.2f}%")
print(f"Matches: {comparison.sum()} out of {len(comparison)}")

# ============================================================================================================
# IDENTIFY AND EXPORT ANOMALIES
# ============================================================================================================

# Get rows where values don't match (anomalies)
# Create a mask for non-null rows that don't match
anomaly_mask = ~mask_null & (
    imputed_ats_invoice_line_item_df['imp_unit_gross_amt_derived'] != 
    imputed_ats_invoice_line_item_df['unit_gross_amt_derived']
)

# Save anomalies to CSV ofr inspection
anomalies_df = imputed_ats_invoice_line_item_df[anomaly_mask]
anomalies_df.to_csv('testing.csv', index=False, mode='w')

print(f"Number of anomalies: {anomaly_mask.sum()}")
# unit_gross_amt_derived cannot be calculated because of missing line_gross_amt_received, or imp_ value is negligibly different
# because of rounding error. can just keep original value in this case

# ============================================================================================================
# PERFORM IMPUTATION ON NULL VALUES
# ============================================================================================================

# All testing imputations match
# Can safely impute unit_gross_amt_derived
null_unit_gross_amt_derived = imputed_ats_invoice_line_item_df['unit_gross_amt_derived'].isnull()

# Where unit_gross_amt_derived is null, set it to line_gross_amt_received / quantity
imputed_ats_invoice_line_item_df.loc[null_unit_gross_amt_derived, 'unit_gross_amt_derived'] = (
    imputed_ats_invoice_line_item_df.loc[null_unit_gross_amt_derived, 'line_gross_amt_received'] / 
    imputed_ats_invoice_line_item_df.loc[null_unit_gross_amt_derived, 'quantity']
).round(2)

# Check that there are no null values remaining
remaining_nulls = imputed_ats_invoice_line_item_df['unit_gross_amt_derived'].isnull().sum()
print(f"Remaining null values in unit_gross_amt_derived: {remaining_nulls}")

if remaining_nulls == 0:
    print("✓ All null values successfully imputed!")
else:
    print(f"⚠ Warning: {remaining_nulls} null values still remain")

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

# ============================================================================================================
# EXTRACT NON-EMPTY EXTRAS ROWS
# ============================================================================================================

# Parse actual rates from extras column
# Select all rows that have extras not null and not '{}'
extras_mask = (
    imputed_ats_invoice_line_item_df['extras'].notnull() & 
    (imputed_ats_invoice_line_item_df['extras'] != '{}')
)

extras_df = imputed_ats_invoice_line_item_df[extras_mask]

# Delete the file if it exists and create new
Path('testing.csv').unlink(missing_ok=True)
extras_df.to_csv('testing.csv', index=False)

print(f"Number of rows with non-empty extras: {extras_mask.sum()}")
# 28094 rows total with non-null extras column

# ============================================================================================================
# EXTRACT AND VALIDATE ORIGINAL_PRICE FROM EXTRAS
# ============================================================================================================

# pulling out original price from extras column
# Filter to only rows where extras is not null and not '{}'
# Create mask for rows where extras contains 'original_price'
original_price_mask = (
    extras_df['extras'].notnull() & 
    (extras_df['extras'] != '{}') &
    extras_df['extras'].str.contains('original_price', na=False)
)

# Apply the mask
original_price_df = extras_df[original_price_mask]

# Add flag column to indicate original_price was parsed
original_price_df['flag'] = 'original_price'

# Save to CSV
Path('testing.csv').unlink(missing_ok=True)
original_price_df.to_csv('testing.csv', index=False)

print(f"Number of rows with 'original_price' in extras: {original_price_mask.sum()}")
# 4487/28094 non-empty extras rows

# Extract original_price using regex directly
original_price_df.loc[:, 'imp_original_price'] = original_price_df['extras'].str.extract(
    r"'original_price'\s*:\s*(\d+\.?\d*)"
).astype(float)

# Add discounted_price column
original_price_df['discounted_price'] = original_price_df['imp_original_price'] - original_price_df['discount_offered']

original_price_df = check_diff(original_price_df, 'discounted_price', 'line_net_amt_received', 'discounted_price_check')
# all values equal

Path('testing.csv').unlink(missing_ok=True)
original_price_df.to_csv('testing.csv', index=False)

# Create a boolean column to check if the equality holds
# .round(2) to account for float issues
original_price_df['price_check'] = (
    original_price_df['imp_original_price'].round(2) == 
    (original_price_df['line_net_amt_received'] + original_price_df['discount_offered']).round(2)
)

# See how many rows match
print(f"Number of rows where imp_original_price = line_net_amt_received + discount_offered: {original_price_df['price_check'].sum()}")
print(f"Total rows: {len(original_price_df)}")
print(f"Percentage matching: {original_price_df['price_check'].sum() / len(original_price_df) * 100:.2f}%")

# Filter to only rows where price_check is False
mismatches_df = original_price_df[original_price_df['price_check'] == False]

# Save to CSV
Path('testing.csv').unlink(missing_ok=True)
mismatches_df.to_csv('testing.csv', index=False, mode='w')

print(f"Saved {len(mismatches_df)} rows with price_check = False to testing.csv")
# Have ensured fidelity of data in regards to discounts received by cardholder from gas stations

# ============================================================================================================
# ADD UNDISCOUNTED_PRICE COLUMN TO MAIN DATAFRAME
# ============================================================================================================

# Initialize the new columns with NaN/None
imputed_ats_invoice_line_item_df['undiscounted_price'] = np.nan
imputed_ats_invoice_line_item_df['discounted_price'] = np.nan
imputed_ats_invoice_line_item_df['flag'] = None

# *** CHANGED: Using merge on 'Unnamed: 0' instead of index-based mapping ***
# Map undiscounted_price using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'undiscounted_price']].merge(
    original_price_df[['Unnamed: 0', 'imp_original_price']].rename(columns={'imp_original_price': 'undiscounted_price_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['undiscounted_price'] = temp_merge['undiscounted_price'].fillna(temp_merge['undiscounted_price_new'])

# Map discounted_price using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'discounted_price']].merge(
    original_price_df[['Unnamed: 0', 'discounted_price']].rename(columns={'discounted_price': 'discounted_price_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['discounted_price'] = temp_merge['discounted_price'].fillna(temp_merge['discounted_price_new'])

# Map flag using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'flag']].merge(
    original_price_df[['Unnamed: 0', 'flag']].rename(columns={'flag': 'flag_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['flag'] = temp_merge['flag'].fillna(temp_merge['flag_new'])

print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'undiscounted_price'))
print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'discounted_price'))
print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'flag'))
# 4,487 rows or 1.18%

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

# ============================================================================================================
# EXTRACT AND VALIDATE BULK_RATE AND NEW_UNIT_PRICE FROM EXTRAS
# ============================================================================================================

# pulling out 'new_unit_price' and 'bulk_rate' from extras column
# Filter to only rows where extras is not null and not '{}'
# Create mask for rows where extras contains 'new_unit_price' and 'bulk_rate'
bulk_rate_mask = (
    extras_df['extras'].notnull() & 
    (extras_df['extras'] != '{}') &
    extras_df['extras'].str.contains('bulk_rate', na=False)
)

# Apply the mask and create a copy
bulk_rate_df = extras_df[bulk_rate_mask].copy()
# Save as CSV
bulk_rate_df.to_csv('temp_csvs/bulk_rate_data.csv', index=False)
# Read back
bulk_rate_df = pd.read_csv('temp_csvs/bulk_rate_data.csv')

# Check if every row in bulk_rate_df['extras'] contains 'new_unit_price'
contains_new_unit_price = bulk_rate_df['extras'].str.contains('new_unit_price', na=False)

# Check if ALL rows contain it
all_contain = contains_new_unit_price.all()
all_contain
# TRUE

# Save to CSV
Path('testing.csv').unlink(missing_ok=True)
bulk_rate_df.to_csv('testing.csv', index=False)

print(f"Number of rows with 'bulk_rate' in extras: {len(bulk_rate_df)}")
# 12283/28094 rows with non-empty extras column
# 4487 rows from original_price makes 16770/28094 parsed so far

# Extract 'bulk_rate' and 'new_unit_price' using regex directly
bulk_rate_df.loc[:, 'imp_bulk_rate'] = bulk_rate_df['extras'].str.extract(
    r"'bulk_rate'\s*:\s*(\d+\.?\d*)"
).astype(float).round(4)

bulk_rate_df.loc[:, 'imp_new_unit_price'] = bulk_rate_df['extras'].str.extract(
    r"'new_unit_price'\s*:\s*(\d+\.?\d*)"
).astype(float).round(4)

# Check for nulls in the newly created columns
print(f"Nulls in imp_bulk_rate: {bulk_rate_df['imp_bulk_rate'].isna().sum()}")
print(f"Nulls in imp_new_unit_price: {bulk_rate_df['imp_new_unit_price'].isna().sum()}")
# 959 nulls in imp_bulk_rate
# This is because the actual is recorded as None, so record as 0

# Replace nulls with zero in imp_bulk_rate
bulk_rate_df['imp_bulk_rate'] = bulk_rate_df['imp_bulk_rate'].fillna(0)
print(f"Nulls in imp_bulk_rate: {bulk_rate_df['imp_bulk_rate'].isna().sum()}")

Path('testing.csv').unlink(missing_ok=True)
bulk_rate_df.to_csv('testing.csv', index=False)

# Check if values are equal within tolerance
bulk_rate_df['price_check'] = np.isclose(
    bulk_rate_df['unit_gross_amt_received'],
    bulk_rate_df['imp_new_unit_price'] - bulk_rate_df['imp_bulk_rate'],
    rtol=0,
    atol=1e-4  # tolerance of 0.0001 (4 decimal places)
)

Path('testing.csv').unlink(missing_ok=True)
bulk_rate_df.to_csv('testing.csv', index=False)

# See how many rows match
print(f"Number of rows where imp_original_price = line_net_amt_received + discount_offered: {bulk_rate_df['price_check'].sum()}")
print(f"Total rows: {len(bulk_rate_df)}")
print(f"Percentage matching: {bulk_rate_df['price_check'].sum() / len(bulk_rate_df) * 100:.2f}%")

# Add undiscounted_price column
bulk_rate_df['undiscounted_price'] = (bulk_rate_df['imp_new_unit_price'] * bulk_rate_df['quantity'] * 1.15).round(2)

# Add diff_undiscounted_price_line_net_amt_received column
bulk_rate_df['diff_undiscounted_price_line_net_amt_received'] = (
    bulk_rate_df['undiscounted_price'] - bulk_rate_df['line_net_amt_received']
)

# Generate summary statistics
print(f"\nSummary statistics for diff_undiscounted_price_line_net_amt_received:")
print(f"Mean: {bulk_rate_df['diff_undiscounted_price_line_net_amt_received'].mean():.4f}")
print(f"Standard Deviation: {bulk_rate_df['diff_undiscounted_price_line_net_amt_received'].std():.4f}")
print(get_non_null_percentage(bulk_rate_df, 'diff_undiscounted_price_line_net_amt_received'))
# This is about the right calculation, but i think the differences are due to rounding error

# Will just use line_net_amt_received as the undiscounted price
print(get_non_null_percentage(bulk_rate_df, 'line_net_amt_received'))
bulk_rate_df['undiscounted_price'] = bulk_rate_df['line_net_amt_received'] 

# Add a discounted price column. this will use quantity * (new_price - bulk rate), which is same as unit_gross_amt_received
bulk_rate_df['discounted_price'] = ((bulk_rate_df['quantity'] * bulk_rate_df['unit_gross_amt_received']) * 1.15).round(2)

# Add flag column
bulk_rate_df['flag'] = 'bulk_rate'

Path('testing.csv').unlink(missing_ok=True)
bulk_rate_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD BULK_RATE COLUMN TO MAIN DATAFRAME
# ============================================================================================================

# *** CHANGED: Using merge on 'Unnamed: 0' instead of index-based mapping ***
# Map undiscounted_price using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'undiscounted_price']].merge(
    bulk_rate_df[['Unnamed: 0', 'undiscounted_price']].rename(columns={'undiscounted_price': 'undiscounted_price_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['undiscounted_price'] = temp_merge['undiscounted_price'].fillna(temp_merge['undiscounted_price_new'])

# Map discounted_price using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'discounted_price']].merge(
    bulk_rate_df[['Unnamed: 0', 'discounted_price']].rename(columns={'discounted_price': 'discounted_price_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['discounted_price'] = temp_merge['discounted_price'].fillna(temp_merge['discounted_price_new'])

# Map flag using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'flag']].merge(
    bulk_rate_df[['Unnamed: 0', 'flag']].rename(columns={'flag': 'flag_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['flag'] = temp_merge['flag'].fillna(temp_merge['flag_new'])

print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'undiscounted_price'))
print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'discounted_price'))
print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'flag'))
# 16,770 rows or 4.41% done so far

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

# ============================================================================================================
# IDENTIFY LEFTOVER EXTRAS ROWS
# ============================================================================================================

# Create a mask that excludes rows matching either original_price_mask or bulk_rate_mask
leftover_extras_mask = (
    extras_df['extras'].notnull() & 
    (extras_df['extras'] != '{}') &
    ~extras_df['extras'].str.contains('original_price', na=False) &
    ~extras_df['extras'].str.contains('bulk_rate', na=False)
)

# Apply the mask and create a copy
leftovers_df = extras_df[leftover_extras_mask].copy()
# Save as CSV
leftovers_df.to_csv('temp_csvs/leftover_extras_data.csv', index=False)
# Read back
leftovers_df = pd.read_csv('temp_csvs/leftover_extras_data.csv')

# ============================================================================================================
# EXTRACT AND VALIDATE YOURRATE FROM EXTRAS
# ============================================================================================================

# Create a mask to filter out yourRate from remaining extras rows
yourRate_mask = (
    extras_df['extras'].notnull() & 
    (extras_df['extras'] != '{}') &
    ~extras_df['extras'].str.contains('original_price', na=False) &
    ~extras_df['extras'].str.contains('bulk_rate', na=False) &
    extras_df['extras'].str.contains('yourRate')
)

# Apply the mask and create a copy
yourRate_df = extras_df[yourRate_mask].copy()
# Save as CSV
yourRate_df.to_csv('temp_csvs/yourRate_data.csv', index=False)
# Read back
yourRate_df = pd.read_csv('temp_csvs/yourRate_data.csv')

print(f"Number of rows with 'yourRate' in extras: {len(yourRate_df)}")
# 12283/28094 rows with bulk_rate and new_unit_price
# 4487 rows from original_price makes 16770/28094 parsed so far
# 11311 rows from yourRate 
# total 28081/28094. remaining rows have only address information

# Extract 'bulk_rate' and 'new_unit_price' using regex directly
yourRate_df.loc[:, 'imp_yourRate'] = yourRate_df['extras'].str.extract(
    r"'yourRate'\s*:\s*'(\d+\.?\d*)'"
).astype(float)

Path('testing.csv').unlink(missing_ok=True)
yourRate_df.to_csv('testing.csv', index=False)

yourRate_df['imp_line_net_amt_received'] = yourRate_df['imp_yourRate'] * yourRate_df['quantity']
yourRate_df['imp_line_net_amt_received'] = yourRate_df['imp_line_net_amt_received'].round(2)

# Check relationship is as expected
yourRate_df['is_equal'] = yourRate_df['line_net_amt_received'] == yourRate_df['imp_line_net_amt_received']

# See results
print(yourRate_df['is_equal'].value_counts())
# 1 FALSE, because of negative value for quantity, yourRate, etc.

Path('testing.csv').unlink(missing_ok=True)
yourRate_df.to_csv('testing.csv', index=False)

# Add discounted_price column
yourRate_df['discounted_price'] = (yourRate_df['quantity'] * yourRate_df['imp_yourRate']).round(2)

# Add undiscounted_price column
yourRate_df['undiscounted_price'] = (yourRate_df['quantity'] * yourRate_df['unit_gross_amt_received']).round(2)
# Replace zeros in undiscounted_price with discounted_price values
yourRate_df.loc[yourRate_df['undiscounted_price'] == 0, 'undiscounted_price'] = (
    yourRate_df.loc[yourRate_df['undiscounted_price'] == 0, 'discounted_price']
)

# Add flag column
yourRate_df['flag'] = 'your_rate'

print(len(yourRate_df))

Path('testing.csv').unlink(missing_ok=True)
yourRate_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD YOUR_RATE COLUMN TO MAIN DATAFRAME
# ============================================================================================================

# *** CHANGED: Using merge on 'Unnamed: 0' instead of index-based mapping ***
# Map undiscounted_price using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'undiscounted_price']].merge(
    yourRate_df[['Unnamed: 0', 'undiscounted_price']].rename(columns={'undiscounted_price': 'undiscounted_price_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['undiscounted_price'] = temp_merge['undiscounted_price'].fillna(temp_merge['undiscounted_price_new'])

# Map discounted_price using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'discounted_price']].merge(
    yourRate_df[['Unnamed: 0', 'discounted_price']].rename(columns={'discounted_price': 'discounted_price_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['discounted_price'] = temp_merge['discounted_price'].fillna(temp_merge['discounted_price_new'])

# Map flag using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'flag']].merge(
    yourRate_df[['Unnamed: 0', 'flag']].rename(columns={'flag': 'flag_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['flag'] = temp_merge['flag'].fillna(temp_merge['flag_new'])

print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'undiscounted_price'))
print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'discounted_price'))
print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'flag'))
# 28,081 rows or 7.39% done so far

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

# ============================================================================================================
# FIND ALL DISCOUNT_OFFERED = 0
# ============================================================================================================

discount_zero_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull() & 
    (imputed_ats_invoice_line_item_df['discount_offered'] == 0)  # Added parentheses
)

# Apply the mask and create a copy
discount_zero_df = imputed_ats_invoice_line_item_df[discount_zero_mask].copy()
print(len(discount_zero_df))

# Save to CSV
discount_zero_df.to_csv('testing.csv', index=False)

# Set the main delivery columns
discount_zero_df['undiscounted_price'] = discount_zero_df['line_net_amt_received']
discount_zero_df['discounted_price'] = discount_zero_df['undiscounted_price']
discount_zero_df['flag'] = 'discount_zero'

print(get_non_null_percentage(discount_zero_df, 'undiscounted_price'))
print((discount_zero_df['undiscounted_price'] == 0).sum())

discount_zero_df.loc[
    discount_zero_df['undiscounted_price'] == 0, 
    'flag'
] = 'no_price'

print((discount_zero_df['flag'] == 'no_price').sum())

# Save to CSV
discount_zero_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD DISCOUNT_ZERO COLUMNS TO MAIN DATAFRAME
# ============================================================================================================

# Map undiscounted_price using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'undiscounted_price']].merge(
    discount_zero_df[['Unnamed: 0', 'undiscounted_price']].rename(columns={'undiscounted_price': 'undiscounted_price_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['undiscounted_price'] = temp_merge['undiscounted_price'].fillna(temp_merge['undiscounted_price_new'])

# Map discounted_price using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'discounted_price']].merge(
    discount_zero_df[['Unnamed: 0', 'discounted_price']].rename(columns={'discounted_price': 'discounted_price_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['discounted_price'] = temp_merge['discounted_price'].fillna(temp_merge['discounted_price_new'])

# Map flag using identifier column
temp_merge = imputed_ats_invoice_line_item_df[['Unnamed: 0', 'flag']].merge(
    discount_zero_df[['Unnamed: 0', 'flag']].rename(columns={'flag': 'flag_new'}),
    on='Unnamed: 0',
    how='left'
)
imputed_ats_invoice_line_item_df['flag'] = temp_merge['flag'].fillna(temp_merge['flag_new'])

print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'undiscounted_price'))
print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'discounted_price'))
print(get_non_null_percentage(imputed_ats_invoice_line_item_df, 'flag'))
# 43,531 rows or 11.45% done so far

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

# ============================================================================================================
# FIND ALL DISCOUNT_OFFERED = LINE_GROSS_AMT_RECEIVED I.E. FREE
# ============================================================================================================

free_mask = (
    (imputed_ats_invoice_line_item_df['discount_offered'] != 0) &
    (imputed_ats_invoice_line_item_df['discount_offered'].notnull()) &
    (imputed_ats_invoice_line_item_df['line_gross_amt_received'] == 
     imputed_ats_invoice_line_item_df['discount_offered'])
)

# Apply the mask and create a copy
free_df = imputed_ats_invoice_line_item_df[free_mask].copy()
print(len(free_df))

# Check that these have no other flags
print(free_df['flag'].isnull().all())

# Save to CSV
free_df.to_csv('testing.csv', index=False)

# Set the main delivery columns
free_df['undiscounted_price'] = free_df['line_gross_amt_received']
free_df['discounted_price'] = 0
free_df['flag'] = 'free_gift' 

# Save to CSV
free_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD FREE_GIFT COLUMNS TO MAIN DATAFRAME
# ============================================================================================================

merge_updates_to_main_df(
    imputed_ats_invoice_line_item_df, 
    free_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 43,649 rows or 11.48% done so far

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_ats_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# FIND ALL UNIT_GROSS_AMT_RECEIVED = 0 I.E. price_0
# ============================================================================================================

price_zero_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull() &
    (imputed_ats_invoice_line_item_df['unit_gross_amt_received'] == 0)
)

price_zero_df = imputed_ats_invoice_line_item_df[price_zero_mask].copy()
# Save to CSV
price_zero_df.to_csv('testing.csv', index=False)

# Set the main delivery columns
price_zero_df['undiscounted_price'] = 0
price_zero_df['discounted_price'] = 0
price_zero_df['flag'] = 'price_zero' 

# Save to CSV
price_zero_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD FREE_GIFT COLUMNS TO MAIN DATAFRAME
# ============================================================================================================

merge_updates_to_main_df(
    imputed_ats_invoice_line_item_df, 
    price_zero_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 48,267 rows or 12.69% done so far

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_ats_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# FIND ALL DISCOUNT_OFFERED = NULL I.E. DISCOUNT_ASSUMED_ZERO
# ============================================================================================================

discount_null_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull() & 
    (imputed_ats_invoice_line_item_df['discount_offered'].isnull())  # Added parentheses
)

discount_null_df = imputed_ats_invoice_line_item_df[discount_null_mask].copy()
# Save to CSV
discount_null_df.to_csv('testing.csv', index=False)

# Inspect dataframe
results = analyze_dataframe(
    discount_null_df, 
    df_name='discount_null', 
    output_filename='discount_null_stats.csv'
)

# Set the main delivery columns
discount_null_df['undiscounted_price'] = discount_null_df['line_net_amt_received']
discount_null_df['discounted_price'] = discount_null_df['undiscounted_price']
discount_null_df['flag'] = 'discount_assumed_zero' 

# Save to CSV
discount_null_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD DISCOUNT_ASSUMED_ZERO COLUMNS TO MAIN DATAFRAME
# ============================================================================================================

merge_updates_to_main_df(
    imputed_ats_invoice_line_item_df, 
    discount_null_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 71,300 rows or 18.75% done so far

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_ats_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FIND ALL LINE_GROSS_AMT_RECEIVED = QUANTITY * UNIT_GROSS_AMT_RECEIVED + DISCOUNT 
# NAMED AS DISCOUNT_REGULAR
# ============================================================================================================

# Recreate the mask with current dataframe state
discount_regular_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull() & 
    (imputed_ats_invoice_line_item_df['line_gross_amt_received'] == 
     (imputed_ats_invoice_line_item_df['quantity'] * 
      imputed_ats_invoice_line_item_df['unit_gross_amt_received'] + 
      imputed_ats_invoice_line_item_df['discount_offered']))
)

# Apply the mask
discount_regular_df = imputed_ats_invoice_line_item_df[discount_regular_mask].copy()

# Verify all flags are null
print(f"All flags null: {discount_regular_df['flag'].isnull().all()}")
print(f"Non-null flag count: {discount_regular_df['flag'].notnull().sum()}")

# Save to CSV
discount_regular_df.to_csv('testing.csv', index=False)

# Set the main delivery columns
discount_regular_df['undiscounted_price'] = discount_regular_df['line_gross_amt_received']
discount_regular_df['discounted_price'] = (discount_regular_df['undiscounted_price'] - discount_regular_df['discount_offered'])
discount_regular_df['flag'] = 'discount_regular' 

# Save to CSV
discount_regular_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD DISCOUNT_REGULAR COLUMNS TO MAIN DATAFRAME
# ============================================================================================================

merge_updates_to_main_df(
    imputed_ats_invoice_line_item_df, 
    discount_regular_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 351,689 rows or 83.03% done so far

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_ats_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FIND ALL LINE_GROSS_AMT_RECEIVED = QUANTITY * UNIT_GROSS_AMT_RECEIVED + DISCOUNT 
# NOT CAUGHT WITH FIRST FILTER FOR ODD REASONS
# NAMED AS DISCOUNT_REGULAR
# ============================================================================================================

discount_regular_again_df = no_flags_df.copy()

discount_regular_again_df['check_sum'] = (
    (discount_regular_again_df['quantity'] * 
      discount_regular_again_df['unit_gross_amt_received'] + 
      discount_regular_again_df['discount_offered']
    ).round(2)
)

# Rounding for numerical stability
discount_regular_again_df['line_gross_amt_received'] = discount_regular_again_df['line_gross_amt_received'].round(2)
discount_regular_again_df['diff'] = (discount_regular_again_df['line_gross_amt_received'] - discount_regular_again_df['check_sum']).round(2)

# Save to CSV
discount_regular_again_df.to_csv('testing.csv', index=False)

# Filter for rows where diff == 0
zero_diff_mask = discount_regular_again_df['diff'] == 0

# Set the main delivery columns only for rows where diff == 0
discount_regular_again_df.loc[zero_diff_mask, 'undiscounted_price'] = discount_regular_again_df.loc[zero_diff_mask, 'line_gross_amt_received']
discount_regular_again_df.loc[zero_diff_mask, 'discounted_price'] = (discount_regular_again_df.loc[zero_diff_mask, 'undiscounted_price'] - 
                                                                       discount_regular_again_df.loc[zero_diff_mask, 'discount_offered'])
discount_regular_again_df.loc[zero_diff_mask, 'flag'] = 'discount_regular'

# Save to CSV
discount_regular_again_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD DISCOUNT_ASSUMED_ZERO COLUMNS TO MAIN DATAFRAME
# ============================================================================================================

merge_updates_to_main_df(
    imputed_ats_invoice_line_item_df, 
    discount_regular_again_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 379,799 rows or 99.98% done so far

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_ats_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FIND ALL LINE_NET_AMT_RECEIVED = LINE_GROSS_AMT_RECEIVED - DISCOUNT_OFFERED 
# NAMED AS DISCOUNTED
# ============================================================================================================

discounted_df = no_flags_df.copy()

discounted_df['check_sum'] = (
    (discounted_df['line_gross_amt_received'] -
      discounted_df['discount_offered']
    ).round(2)
)

discounted_df['check_equal'] = (discounted_df['line_net_amt_received'] == 
                                (discounted_df['line_gross_amt_received'] - 
                                 discounted_df['discount_offered'])
                                 )

# Save to CSV
discounted_df.to_csv('testing.csv', index=False)

# Rounding for numerical stability
discounted_df['line_net_amt_received'] = discounted_df['line_net_amt_received'].round(2)
discounted_df['diff'] = (discounted_df['line_net_amt_received'] - discounted_df['check_sum']).round(2)

# Save to CSV
discounted_df.to_csv('testing.csv', index=False)
# All have zero difference to set flags for all to discounted

# Set the main delivery columns only for rows where diff == 0
discounted_df['undiscounted_price'] = discounted_df['line_gross_amt_received']
discounted_df['discounted_price'] = discounted_df['line_net_amt_received']
discounted_df['flag'] = 'discounted'

# Save to CSV
discounted_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD DISCOUNT_ASSUMED_ZERO COLUMNS TO MAIN DATAFRAME
# ============================================================================================================

merge_updates_to_main_df(
    imputed_ats_invoice_line_item_df, 
    discounted_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 380,213 rows or 100.00%!

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv('imputed_ats_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_ats_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_ats_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)