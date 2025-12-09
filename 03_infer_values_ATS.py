# # list of dataframes for quick reference
# ats_invoice_df = all_data.get('ats_invoice')
# ats_invoice_line_item_df = all_data.get('ats_invoice_line_item')
# invoice_df = all_data.get('invoice')
# invoice_line_item_df = all_data.get('invoice_line_item')
# failed_accounts_df = all_data.get('failed_accounts')
# merchant_discount_df = all_data.get('merchant_discount')

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

imputed_ats_invoice_df = ats_invoice_df
imputed_ats_invoice_line_item_df = ats_invoice_line_item_df

# =========================================== imputing ats_invoice ==================================================================== #
print(imputed_ats_invoice_df.columns)

# Imputing gst_amount_ column 
# Get rows where gst_amount is null
null_gst_amount = imputed_ats_invoice_df['gst_amount'].isnull()

# Check if total_amount == gross_transaction_amount for those rows
check = imputed_ats_invoice_df.loc[null_gst_amount, 'total_amount'] == imputed_ats_invoice_df.loc[null_gst_amount, 'gross_transaction_amount']

# See if all are True
all_equal = check.all()
print(f"All total_amount == gross_transaction_amount where gst_amount is null: {all_equal}")

# Count how many match vs don't match
print(f"Matching: {check.sum()}")
print(f"Not matching: {(~check).sum()}")

# Safely impute 0 for gst_amount
imputed_ats_invoice_df['gst_amount'] = imputed_ats_invoice_df['gst_amount'].fillna(0)
# Check for remaining nulls
print(f"Remaining null values in gst_amount: {imputed_ats_invoice_df['gst_amount'].isnull().sum()}")

# Save to CSV
imputed_ats_invoice_df.to_csv('imputed_ats_invoice.csv', index=False, mode='w')

# Imputing rebates column
# Get rows where rebates is null
null_rebates = imputed_ats_invoice_df['rebates'].isnull()

# Check if total_amount - gross_transaction_amount == 0 for those rows
difference = (imputed_ats_invoice_df.loc[null_rebates, 'total_amount'] 
         - imputed_ats_invoice_df.loc[null_rebates, 'gross_transaction_amount'] )

# Check if difference == 0
check = difference == 0

# Verify your expectation
print(check.value_counts())  # Should show all True if expectation holds
print(f"All differences are 0: {check.all()}")  # Should print True

# Can safely impute zeroes for all null rebates rows
imputed_ats_invoice_df['rebates'] = imputed_ats_invoice_df['rebates'].fillna(0)
# Check for remaining nulls
print(f"Remaining null values in rebates: {imputed_ats_invoice_df['rebates'].isnull().sum()}")

# Save to CSV
imputed_ats_invoice_df.to_csv('imputed_ats_invoice.csv', index=False, mode='w')

#  =============================================== ats_invoice_line_item ============================================================ #
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

# Get rows where values don't match (anomalies)
# Create a mask for non-null rows that don't match
anomaly_mask = ~mask_null & (
    imputed_ats_invoice_line_item_df['imp_unit_gross_amt_derived'] != 
    imputed_ats_invoice_line_item_df['unit_gross_amt_derived']
)

# Save anomalies to CSV
anomalies_df = imputed_ats_invoice_line_item_df[anomaly_mask]
anomalies_df.to_csv('testing.csv', index=False, mode='w')

print(f"Number of anomalies: {anomaly_mask.sum()}")
# unit_gross_amt_derived cannot be calculated because of missing line_gross_amt_received, or imp_ value is negligibly different
# because of rounding error. can just keep original value in this case

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

# Save to CSV
Path('testing.csv').unlink(missing_ok=True)
original_price_df.to_csv('testing.csv', index=False)

print(f"Number of rows with 'original_price' in extras: {original_price_mask.sum()}")
# 4487/28094 non-empty extras rows

# Extract original_price using regex directly
original_price_df.loc[:, 'imp_original_price'] = original_price_df['extras'].str.extract(
    r"'original_price'\s*:\s*(\d+\.?\d*)"
).astype(float)

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




# ================================================= generetate the descriptive statistics ====================================
# Generate descriptive statitstics
imputed_data = {
    'imputed_ats_invoice': imputed_ats_invoice_df,
    'imputed_ats_invoice_line_item' : imputed_ats_invoice_line_item_df,
    # add other imputed dataframes here
}

imputed_results = analyze_dataframes(imputed_data, 'imputed_missing_values_summary.csv')