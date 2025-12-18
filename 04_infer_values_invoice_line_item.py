

# THIS IS OLD CODE THAT IS NOT IN USE ANYMORE IF YOU HAVE IT, PLEASE DELETE


import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

imputed_invoice_df = invoice_df
imputed_invoice_line_item_df = invoice_line_item_df

def calculate_percentage_true(df, column_name):
    """
    Calculate the percentage of True values in a boolean column.
    
    Parameters:
    df: pandas DataFrame
    column_name: str, name of the boolean column to analyze
    
    Returns:
    float: percentage of True values
    """
    percentage = (df[column_name].sum() / len(df)) * 100
    return percentage

def calculate_percentage_not_null(df, column_name):
    """
    Calculate the percentage of non-null values in a column.
    
    Parameters:
    df: pandas DataFrame
    column_name: str, name of the column to analyze
    
    Returns:
    float: percentage of non-null values
    """
    percentage = (df[column_name].notna().sum() / len(df)) * 100
    return percentage

# ======================================= imputing invoice_line_item ==================================================================== #
print(imputed_invoice_line_item_df.columns)

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

imputed_invoice_line_item_df = imputed_invoice_line_item_df[[
    'Unnamed: 0', 'product_code', 'description', 'quantity', 'unit_gross_amt_received',
    'discount_offered', 'line_gross_amt_received', 'line_gst_amt_received', 'line_net_amt_received',
    'gst_indicator', 'unit_gst_amt_derived', 'unit_gross_amt_derived', 'line_discount_derived', 
    'line_net_amt_derived', 'line_gst_total_derived', 'line_gross_amt_derived', 'merchant_identifier',
    'merchant_branch', 'gst_rate',
    'amount_excluding_gst', 'discount_delta', 'total_discount_delta',
    'line_gross_amt_derived_excl_gst'
]]

# Save to CSV
Path('imputed_invoice_line_item.csv').unlink(missing_ok=True)
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False)

# # Create mask for rows with no nulls in specified columns
# mask = (
#     imputed_invoice_line_item_df['quantity'].notna() & 
#     imputed_invoice_line_item_df['unit_gross_amt_received'].notna() & 
#     imputed_invoice_line_item_df['line_gross_amt_received'].notna() &
#     imputed_invoice_line_item_df['discount_offered'].notna() &
#     (imputed_invoice_line_item_df['discount_offered'] != 0)
# )

# # Apply mask to create filtered copy
# discount_testing_imputed_invoice_line_item_df = imputed_invoice_line_item_df[mask].copy()

# # Print statistics
# print(f"Length of filtered dataframe: {len(discount_testing_imputed_invoice_line_item_df)}")
# print(f"Length of original dataframe: {len(imputed_invoice_line_item_df)}")
# rows_excluded = len(imputed_invoice_line_item_df) - len(discount_testing_imputed_invoice_line_item_df)
# percentage_excluded = (rows_excluded / len(imputed_invoice_line_item_df)) * 100
# print(f"Percentage of rows excluded by mask: {percentage_excluded:.2f}%")

# direct copy of df. comment out if filtering with  mask
discount_testing_imputed_invoice_line_item_df = imputed_invoice_line_item_df.copy()

# Checking discount_offered record type (either as percentage, or rate)
# Create new column check_discount_isPercentage
discount_testing_imputed_invoice_line_item_df['check_discount_isPercentage'] = (
    discount_testing_imputed_invoice_line_item_df['quantity'] * discount_testing_imputed_invoice_line_item_df['unit_gross_amt_received'] 
    * (100 - discount_testing_imputed_invoice_line_item_df['discount_offered']) / 100 
    == discount_testing_imputed_invoice_line_item_df['line_gross_amt_received']
)

# Checking discount_offered record type (either as percentage, or rate)
# Create new column check_discount_isPercentage
discount_testing_imputed_invoice_line_item_df['check_discount_isPercentage'] = (
(discount_testing_imputed_invoice_line_item_df['quantity'] * discount_testing_imputed_invoice_line_item_df['unit_gross_amt_received'] 
    * (100 - discount_testing_imputed_invoice_line_item_df['discount_offered']) / 100).round(2) 
    == discount_testing_imputed_invoice_line_item_df['line_gross_amt_received'].round(2)
)

# Check percentage of True values in check_discount_isPercentage
percentage_true = calculate_percentage_true(discount_testing_imputed_invoice_line_item_df, 'check_discount_isPercentage')
print(f"Percentage of records where discount appears to be a percentage: {percentage_true:.2f}%")
# 49.82% of rows have discount recorded as a percentage off 

# Add new column discount_category
discount_testing_imputed_invoice_line_item_df['discount_category'] = discount_testing_imputed_invoice_line_item_df['check_discount_isPercentage'].apply(
    lambda x: "Percentage_off" if x else None
)
# Save to CSV
discount_testing_imputed_invoice_line_item_df.to_csv('testing.csv', index=False)

# Checking discount_offered record type (either as percentage, or rate)
# Create new column check_discount_isTotalOffatCheckout
discount_testing_imputed_invoice_line_item_df['check_discount_isTotalOffatCheckout'] = (
    discount_testing_imputed_invoice_line_item_df['quantity'] * discount_testing_imputed_invoice_line_item_df['unit_gross_amt_received'] 
    - discount_testing_imputed_invoice_line_item_df['discount_offered']  
    == discount_testing_imputed_invoice_line_item_df['line_gross_amt_received']
)

# Check percentage of True values in check_discount_isTotalOffatCheckout
percentage_true = calculate_percentage_true(discount_testing_imputed_invoice_line_item_df, 'check_discount_isTotalOffatCheckout')
print(f"Percentage of records where discount appears to be a total amount off at checkout: {percentage_true:.2f}%")
# 12.14% of rows have discount recorded as a total off at checkout 

# Update discount_category only where check_discount_isTotalOffatCheckout is True
discount_testing_imputed_invoice_line_item_df.loc[
    discount_testing_imputed_invoice_line_item_df['check_discount_isTotalOffatCheckout'] == True, 
    'discount_category'
] = "TotalOff_atCheckout"

# check for values where more than one condition is true
# one case, but it is a coincidence owing to unit_gross_amt being exactly 100
# other transactions from this retailer are totalOff
both_true_mask = (
    (discount_testing_imputed_invoice_line_item_df['check_discount_isTotalOffatCheckout'] == True) & 
    (discount_testing_imputed_invoice_line_item_df['check_discount_isPercentage'] == True)
)

both_true_count = both_true_mask.sum()
print(f"Number of rows where both checks are True: {both_true_count}")

# Get the Unnamed: 0 values for those rows
if both_true_count > 0:
    unnamed_0_values = discount_testing_imputed_invoice_line_item_df.loc[both_true_mask, 'Unnamed: 0'].tolist()
    print(f"Unnamed: 0 values: {unnamed_0_values}")

# Save to CSV
discount_testing_imputed_invoice_line_item_df.to_csv('testing.csv', index=False)

# Checking discount_offered record type (either as percentage, or rate, etc.)
# Create new column check_discount_isNegativeOff
discount_testing_imputed_invoice_line_item_df['check_discount_isNegativeOff'] = (
    discount_testing_imputed_invoice_line_item_df['quantity'] * discount_testing_imputed_invoice_line_item_df['unit_gross_amt_received'] 
    - discount_testing_imputed_invoice_line_item_df['discount_offered']  
    == discount_testing_imputed_invoice_line_item_df['line_net_amt_received']
)

# Check percentage of True values in check_discount_isTotalOffatCheckout
percentage_true = calculate_percentage_true(discount_testing_imputed_invoice_line_item_df, 'check_discount_isNegativeOff')
print(f"Percentage of records where discount appears to be a negative value off: {percentage_true:.2f}%")
# 2.28% of rows have discount recorded as a negative value off

# Update discount_category only where check_discount_isTotalOffatCheckout is True
discount_testing_imputed_invoice_line_item_df.loc[
    discount_testing_imputed_invoice_line_item_df['check_discount_isNegativeOff'] == True, 
    'discount_category'
] = "isNegativeOff"

# check for values where more than one condition is true
check_columns = ['check_discount_isNegativeOff', 'check_discount_isTotalOffatCheckout', 'check_discount_isPercentage']

# Count how many True values each row has across these columns
true_count_per_row = discount_testing_imputed_invoice_line_item_df[check_columns].sum(axis=1)

# Find rows where more than one condition is True
more_than_one_true_mask = true_count_per_row > 1
more_than_one_true_count = more_than_one_true_mask.sum()

# Get the Unnamed: 0 values for those rows
if more_than_one_true_count > 0:
    unnamed_0_values = discount_testing_imputed_invoice_line_item_df.loc[more_than_one_true_mask, 'Unnamed: 0'].tolist()
    print(f"Unnamed: 0 values: {unnamed_0_values}")
    
    # Optionally show which combinations are True for these rows
    print("\nWhich checks are True for these rows:")
    print(discount_testing_imputed_invoice_line_item_df.loc[more_than_one_true_mask, ['Unnamed: 0'] + check_columns])

# total percentage of rows that have discount category verified
percentage_not_null = calculate_percentage_not_null(discount_testing_imputed_invoice_line_item_df, 'discount_category')
print(f"Percentage of discount_category that is not null: {percentage_not_null:.2f}%")
# 64.21% accounted for so far

# Save to CSV
discount_testing_imputed_invoice_line_item_df.to_csv('testing.csv', index=False)


discount_testing_imputed_invoice_line_item_df['check_discount_isAmtOffGross'] = (
    (discount_testing_imputed_invoice_line_item_df['quantity'] * 
     discount_testing_imputed_invoice_line_item_df['unit_gross_amt_received'] - 
     discount_testing_imputed_invoice_line_item_df['discount_offered']) == 
    np.where(
        discount_testing_imputed_invoice_line_item_df['line_net_amt_received'].isna(),
        discount_testing_imputed_invoice_line_item_df['line_net_amt_derived'],
        discount_testing_imputed_invoice_line_item_df['line_net_amt_received']
    )
)

# Check percentage of True values in check_discount_isTotalOffatCheckout
percentage_true = calculate_percentage_true(discount_testing_imputed_invoice_line_item_df, 'check_discount_isAmtOffGross')
print(f"Percentage of records where discount appears to be an amount taken off gross: {percentage_true:.2f}%")
# 14.44% of rows have discount recorded as a sum taken off gross

# Update discount_category only where check_discount_isAmtOffGross is True
discount_testing_imputed_invoice_line_item_df.loc[
    discount_testing_imputed_invoice_line_item_df['check_discount_isAmtOffGross'] == True, 
    'discount_category'
] = "isAmtOffGross"

# check for values where more than one condition is true
check_columns = ['check_discount_isNegativeOff', 'check_discount_isTotalOffatCheckout', 
                 'check_discount_isPercentage', 'check_discount_isAmtOffGross']

# Count how many True values each row has across these columns
true_count_per_row = discount_testing_imputed_invoice_line_item_df[check_columns].sum(axis=1)

# Find rows where more than one condition is True
more_than_one_true_mask = true_count_per_row > 1
more_than_one_true_count = more_than_one_true_mask.sum()

# Get the Unnamed: 0 values for those rows
if more_than_one_true_count > 0:
    unnamed_0_values = discount_testing_imputed_invoice_line_item_df.loc[more_than_one_true_mask, 'Unnamed: 0'].tolist()
    print(f"Unnamed: 0 values: {unnamed_0_values}")
    
    # Optionally show which combinations are True for these rows
    print("\nWhich checks are True for these rows:")
    print(discount_testing_imputed_invoice_line_item_df.loc[more_than_one_true_mask, ['Unnamed: 0'] + check_columns])

# total percentage of rows that have discount category verified
percentage_not_null = calculate_percentage_not_null(discount_testing_imputed_invoice_line_item_df, 'discount_category')
print(f"Percentage of discount_category that is not null: {percentage_not_null:.2f}%")

# Save to CSV
discount_testing_imputed_invoice_line_item_df.to_csv('testing.csv', index=False)








# ================================================= generetate the descriptive statistics ====================================
# Generate descriptive statitstics
imputed_data = {
    'imputed_invoice_line_item': imputed_invoice_line_item_df    
    # add other imputed dataframes here
}

imputed_results = analyze_dataframes(imputed_data, 'imputed_invoices_missing_values_summary.csv')