# # list of dataframes for quick reference
# ats_invoice_df = all_data.get('ats_invoice')
# ats_invoice_line_item_df = all_data.get('ats_invoice_line_item')
# invoice_df = all_data.get('invoice')
# invoice_line_item_df = all_data.get('invoice_line_item')
# failed_accounts_df = all_data.get('failed_accounts')
# merchant_discount_df = all_data.get('merchant_discount')

imputed_ats_invoice_df = ats_invoice_df
imputed_ats_invoice_line_item_df = ats_invoice_line_item_df
imputed_invoice_df = invoice_df
imputed_invoice_line_item_df = invoice_line_item_df

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

testing_df = imputed_ats_invoice_line_item_df[extras_mask]
testing_df.to_csv('testing.csv', index=False, mode='w')

print(f"Number of rows with non-empty extras: {extras_mask.sum()}")








# ================================================= generetate the descriptive statistics ====================================
# Generate descriptive statitstics
imputed_data = {
    'imputed_ats_invoice': imputed_ats_invoice_df,
    'imputed_ats_invoice_line_item' : imputed_ats_invoice_line_item_df,
    # add other imputed dataframes here
}

imputed_results = analyze_dataframes(imputed_data, 'imputed_missing_values_summary.csv')