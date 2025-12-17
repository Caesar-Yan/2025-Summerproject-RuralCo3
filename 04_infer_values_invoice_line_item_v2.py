import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Load the data
with open('all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

# Load all functions
with open('invoice_functions.pkl', 'rb') as f:
    funcs = pickle.load(f)

invoice_df = all_data.get('invoice')
invoice_line_item_df = all_data.get('invoice_line_item')

imputed_invoice_df = invoice_df.copy()
imputed_invoice_line_item_df = invoice_line_item_df.copy()

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

def save_and_summarize(df, mask, filename='testing.csv', label='Filtered'):
    """
    Apply mask to dataframe, save to CSV, and print summary statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to filter
    mask : pd.Series (boolean)
        Boolean mask to apply
    filename : str
        Output CSV filename (default: 'testing.csv')
    label : str
        Label for the print output (default: 'Filtered')
    
    Returns:
    --------
    pd.DataFrame
        The filtered dataframe
    """
    # Apply mask and save
    filtered_df = df[mask]
    filtered_df.to_csv(filename, index=False, mode='w')
    
    # Calculate statistics
    filtered_count = len(filtered_df)
    total_count = len(df)
    percentage = (filtered_count / total_count * 100) if total_count > 0 else 0
    
    # Print summary
    print(f"\n{label} Results:")
    print(f"  Rows: {filtered_count:,} out of {total_count:,}")
    print(f"  Percentage: {percentage:.2f}%")
    print(f"  Saved to: {filename}")
    
    return filtered_df

# ======================================= imputing invoice_line_item ==================================================================== #

# Initialize the new columns with NaN/None
imputed_invoice_line_item_df['undiscounted_price'] = np.nan
imputed_invoice_line_item_df['discounted_price'] = np.nan
imputed_invoice_line_item_df['flag'] = None

# Save to CSV
Path('testing.csv').unlink(missing_ok=True)
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False)

# ============================================================================================================
# FILTER OUT DISCOUNT_ZERO
# ============================================================================================================

discount_zero_mask = (
    imputed_invoice_line_item_df['discount_offered'] == 0
)

discount_zero_df = save_and_summarize(
    imputed_invoice_line_item_df, 
    discount_zero_mask, 
    'testing.csv',
    'discount_zero'
)

# want to use only line_gross_amt_received here
discount_zero_df['check_relationship'] = (
    (discount_zero_df['line_gross_amt_received'] == (
        discount_zero_df['quantity'] * 
        discount_zero_df['unit_gross_amt_received'] - 
        discount_zero_df['discount_offered']
        )
    )
    |
    ((discount_zero_df['line_gross_amt_received']).round(2) == (
        discount_zero_df['quantity'] * 
        discount_zero_df['unit_gross_amt_received'] - 
        discount_zero_df['discount_offered']
        ).round(2)
    )
    |
    ((discount_zero_df['line_gross_amt_received']) == 
     (discount_zero_df['line_net_amt_received'])
    )
    |
    ((discount_zero_df['line_gross_amt_received']) == 
     (discount_zero_df['line_net_amt_derived'])
    )
)

discount_zero_df.to_csv('testing.csv', index=False)

percentage_true = calculate_percentage_true(discount_zero_df, 'check_relationship')
print(f"Percentage where relationship holds true: {percentage_true:.2f}%")
# 85.68%

# Set the main delivery columns
discount_zero_df['undiscounted_price'] = discount_zero_df['line_gross_amt_received']
discount_zero_df['discounted_price'] = discount_zero_df['undiscounted_price']
discount_zero_df['flag'] = 'discount_zero' 

# Save to CSV
discount_zero_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD discount_zero FLAG TO MAIN DATAFRAME
# ============================================================================================================

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    discount_zero_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 71,625 rows, or 12.31% flagged

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER OUT PERCENTAGE_OFF FROM imputed_invoice_line_item_df
# ============================================================================================================

# Filter for lines where discount_offered is recorded as a percentage to take off
percentage_off_mask = (
        (no_flags_df['quantity'] * 
        no_flags_df['unit_gross_amt_received'] * 
        (100 - no_flags_df['discount_offered']) / 100 == 
        no_flags_df['line_gross_amt_received']
        )
        |
        (
            (no_flags_df['quantity'] * 
             no_flags_df['unit_gross_amt_received'] * 
             (100 - no_flags_df['discount_offered']) / 100
             ).round(2) == 
             (no_flags_df['line_gross_amt_received']).round(2)
)
)

percentage_off_df = save_and_summarize(
    no_flags_df, 
    percentage_off_mask, 
    'testing.csv',
    'Percentage_off'
)

# Check the math to calculate undiscounted price. line_gross_derived is assumed to be undiscounted price
# use isclose() to account for rounding error
percentage_off_df['check_relationship'] = (
    np.isclose(
        percentage_off_df['line_gross_amt_derived'] - percentage_off_df['line_discount_derived'],
        percentage_off_df['line_net_amt_derived'],
        atol=0.01
    )
    |
    # accounting for rows with discount recorded as negative
    np.isclose(
        percentage_off_df['line_gross_amt_derived'] + percentage_off_df['line_discount_derived'],
        percentage_off_df['line_net_amt_derived'],
        atol=0.01
    )
    |
    np.isclose(
        (percentage_off_df['line_gross_amt_derived'] - percentage_off_df['line_discount_derived']).round(2),
        (percentage_off_df['line_net_amt_derived']).round(2),
        atol=0.01
    )
    |
    # accounting for rows with discount recorded as negative
    np.isclose(
        (percentage_off_df['line_gross_amt_derived'] + percentage_off_df['line_discount_derived']).round(2),
        (percentage_off_df['line_net_amt_derived']).round(2),
        atol=0.01
    )
)

percentage_off_df.to_csv('testing.csv', index=False)

percentage_true = calculate_percentage_true(percentage_off_df, 'check_relationship')
print(f"Percentage where relationship holds true: {percentage_true:.2f}%")
# 99.41%

# Filter for lines where check_relationship is TRUE
percentage_off_check_true_mask = (
    percentage_off_df['check_relationship'] == True
)

percentage_off_df = save_and_summarize(
    percentage_off_df, 
    percentage_off_check_true_mask, 
    'testing.csv',
    'Percentage_off_check_true'
)

# Set the main delivery columns
percentage_off_df['undiscounted_price'] = percentage_off_df['line_gross_amt_received']
percentage_off_df['discounted_price'] = percentage_off_df['line_net_amt_derived']
percentage_off_df['flag'] = 'percentage_off' 

# Save to CSV
percentage_off_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD PERCENTAGE_OFF FLAG TO MAIN DATAFRAME
# ============================================================================================================

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    percentage_off_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 94,162 rows, or 16.19% flagged

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER OUT discount_negative_sum FROM imputed_invoice_line_item_df
# ============================================================================================================

# Filter for lines where discount_offered is recorded as a percentage to take off
negative_sum_mask = (
        no_flags_df['discount_offered'] < 0
)

negative_sum_df = save_and_summarize(
    no_flags_df, 
    negative_sum_mask, 
    'testing.csv',
    'negative_sum'
)

# Check the math to calculate undiscounted price. line_gross_derived is assumed to be undiscounted price
# use isclose() to account for rounding error
negative_sum_df['check_relationship'] = (
    negative_sum_df['line_net_amt_received'] +
    negative_sum_df['discount_offered'] ==
    negative_sum_df['']
)

percentage_off_df.to_csv('testing.csv', index=False)

percentage_true = calculate_percentage_true(percentage_off_df, 'check_relationship')
print(f"Percentage where relationship holds true: {percentage_true:.2f}%")
# 99.41%

# Filter for lines where check_relationship is TRUE
percentage_off_check_true_mask = (
    percentage_off_df['check_relationship'] == True
)

percentage_off_df = save_and_summarize(
    percentage_off_df, 
    percentage_off_check_true_mask, 
    'testing.csv',
    'Percentage_off_check_true'
)

# Set the main delivery columns
percentage_off_df['undiscounted_price'] = percentage_off_df['line_gross_amt_received']
percentage_off_df['discounted_price'] = percentage_off_df['line_net_amt_derived']
percentage_off_df['flag'] = 'percentage_off' 

# Save to CSV
percentage_off_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD PERCENTAGE_OFF FLAG TO MAIN DATAFRAME
# ============================================================================================================

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    percentage_off_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 94,162 rows, or 16.19% flagged

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)









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
 
