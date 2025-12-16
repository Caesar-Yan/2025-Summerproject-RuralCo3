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






# ================================================= generetate the descriptive statistics ====================================
# Generate descriptive statitstics
imputed_data = {
    'imputed_ats_invoice': imputed_ats_invoice_df,
    'imputed_ats_invoice_line_item' : imputed_ats_invoice_line_item_df,
    # add other imputed dataframes here
}

imputed_results = analyze_dataframes(imputed_data, 'imputed_missing_values_summary.csv')