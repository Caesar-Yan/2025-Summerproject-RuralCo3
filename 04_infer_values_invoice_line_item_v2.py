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

def add_category(df, mask, category_name):
    """Add category label to discount_category column"""
    # Append to existing categories
    df.loc[mask & df['discount_category'].notna(), 'discount_category'] = (
        df.loc[mask & df['discount_category'].notna(), 'discount_category'] + ', ' + category_name
    )
    # Set for rows with no existing category
    df.loc[mask & df['discount_category'].isna(), 'discount_category'] = category_name

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

# direct copy of df. comment out if filtering with  mask
discount_testing_imputed_invoice_line_item_df = imputed_invoice_line_item_df.copy()

# Save to CSV
Path('testing.csv').unlink(missing_ok=True)
discount_testing_imputed_invoice_line_item_df.to_csv('testing.csv', index=False)




# Create new column 'discount_category' based on conditions (vectorized)
discount_testing_imputed_invoice_line_item_df['discount_category'] = None

# Zero condition: all zeroes (APPLY FIRST - this takes priority)
mask0 = (
    (
        (discount_testing_imputed_invoice_line_item_df['quantity'] == 0) &
        (discount_testing_imputed_invoice_line_item_df['unit_gross_amt_received'] == 0) &
        (discount_testing_imputed_invoice_line_item_df['discount_offered'] == 0) &
        (discount_testing_imputed_invoice_line_item_df['line_gross_amt_received'] == 0) &
        (discount_testing_imputed_invoice_line_item_df['line_gst_amt_received'] == 0)
    )
    |
    (
        (discount_testing_imputed_invoice_line_item_df['unit_gst_amt_derived'] == 0) &
        (discount_testing_imputed_invoice_line_item_df['line_discount_derived'] == 0) &
        (discount_testing_imputed_invoice_line_item_df['line_net_amt_derived'] == 0) &
        (discount_testing_imputed_invoice_line_item_df['line_gst_total_derived'] == 0)
    )
)
discount_testing_imputed_invoice_line_item_df.loc[mask0, 'discount_category'] = 'No_info'

# First condition: Negative_sum (exclude mask0 rows)
mask1 = (
    (discount_testing_imputed_invoice_line_item_df['discount_offered'] < 0) &
    (discount_testing_imputed_invoice_line_item_df['line_net_amt_received'] == 
     (discount_testing_imputed_invoice_line_item_df['line_gross_amt_received'] - 
      discount_testing_imputed_invoice_line_item_df['discount_offered'])) &
    (~mask0)  # Exclude rows already marked as No_info
)
discount_testing_imputed_invoice_line_item_df.loc[mask1, 'discount_category'] = "Negative_sum"

# Second condition: sum_off_per_unit (exact match OR rounded to 2 decimals, exclude mask0 rows)
mask2 = (
    (
        (discount_testing_imputed_invoice_line_item_df['line_net_amt_received'] == 
         (discount_testing_imputed_invoice_line_item_df['quantity'] * 
          (discount_testing_imputed_invoice_line_item_df['unit_gross_amt_received'] - 
           discount_testing_imputed_invoice_line_item_df['discount_offered'])))
        |
        (discount_testing_imputed_invoice_line_item_df['line_net_amt_received'].round(2) == 
         (discount_testing_imputed_invoice_line_item_df['quantity'] * 
          (discount_testing_imputed_invoice_line_item_df['unit_gross_amt_received'] - 
           discount_testing_imputed_invoice_line_item_df['discount_offered'])).round(2))
    ) &
    (~mask0)  # Exclude rows already marked as No_info
)

# Third condition: percentage off total
mask3 = (
    (discount_testing_imputed_invoice_line_item_df['line_discount_derived'] == 
     discount_testing_imputed_invoice_line_item_df['line_gross_amt_derived'] * 
     discount_testing_imputed_invoice_line_item_df['discount_offered'] / 100)
    &
    (~mask0)  # Exclude rows already marked as No_info
)
 

# Handle mask2
add_category(discount_testing_imputed_invoice_line_item_df, mask2, 'sum_off_per_unit')

# Handle mask3
add_category(discount_testing_imputed_invoice_line_item_df, mask3, 'percentage_off')





print(f"\nDiscount category value counts:")
print(discount_testing_imputed_invoice_line_item_df['discount_category'].value_counts(dropna=False))

# Sort by discount_category before saving
discount_testing_imputed_invoice_line_item_df = discount_testing_imputed_invoice_line_item_df.sort_values(
    by='discount_category', 
    na_position='last'
)

# Save to CSV
Path('testing.csv').unlink(missing_ok=True)
discount_testing_imputed_invoice_line_item_df.to_csv('testing.csv', index=False)