import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import dill

# Load the data
with open('all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

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

def analyze_dataframes(data_dict, output_filename='missing_values_summary.csv'):
    """
    Analyze multiple dataframes for missing values and basic statistics.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are table names and values are DataFrames
    output_filename : str
        Name of the output CSV file (default: 'missing_values_summary.csv')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the analysis results
    """
    results = []
    
    for name, df in data_dict.items():
        total_rows = len(df)  # Total number of rows in the dataframe
        
        for column in df.columns:
            # Basic missing percentage
            missing_percentage = (df[column].isnull().sum() / len(df)) * 100
            
            # Count non-empty rows (not null/NaN)
            non_empty_count = df[column].notna().sum()
            
            row_data = {
                'table_name': name,
                'column_name': column,
                'percetage_present': round(100 - missing_percentage, 2),
                'total_rows': total_rows,
                'non_empty_count': non_empty_count
            }
            
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                # Calculate statistics (excluding NaN values)
                row_data['mean'] = round(df[column].mean(), 2)
                row_data['median'] = round(df[column].median(), 2)
                row_data['mode'] = df[column].mode()[0] if not df[column].mode().empty else None
                row_data['std_dev'] = round(df[column].std(), 2)
            else:
                # For non-numeric columns, set these as None/NaN
                row_data['mean'] = None
                row_data['median'] = None
                row_data['mode'] = None
                row_data['std_dev'] = None
            
            results.append(row_data)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False, mode='w')
    
    print(f"Saved to {output_filename}")
    
    return results_df

def analyze_dataframe(df, df_name='dataframe', output_filename='missing_values_summary.csv'):
    """
    Analyze a single dataframe for missing values and basic statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    df_name : str
        Name to use for the dataframe in the output (default: 'dataframe')
    output_filename : str
        Name of the output CSV file (default: 'missing_values_summary.csv')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the analysis results
    """
    results = []
    total_rows = len(df)
    
    for column in df.columns:
        # Basic missing percentage
        missing_percentage = (df[column].isnull().sum() / len(df)) * 100
        
        # Count non-empty rows (not null/NaN)
        non_empty_count = df[column].notna().sum()
        
        row_data = {
            'table_name': df_name,
            'column_name': column,
            'percetage_present': round(100 - missing_percentage, 2),
            'total_rows': total_rows,
            'non_empty_count': non_empty_count
        }
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Calculate statistics (excluding NaN values)
            row_data['mean'] = round(df[column].mean(), 2)
            row_data['median'] = round(df[column].median(), 2)
            row_data['mode'] = df[column].mode()[0] if not df[column].mode().empty else None
            row_data['std_dev'] = round(df[column].std(), 2)
        else:
            # For non-numeric columns, set these as None/NaN
            row_data['mean'] = None
            row_data['median'] = None
            row_data['mode'] = None
            row_data['std_dev'] = None
        
        results.append(row_data)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False, mode='w')
    
    print(f"Saved to {output_filename}")
    
    return results_df

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

# Check the math to calculate undiscounted price. line_gross_amt_derived is assumed to be undiscounted price
negative_sum_df['check_relationship'] = np.isclose(
    negative_sum_df['line_net_amt_received'] + negative_sum_df['discount_offered'],
    negative_sum_df['line_gross_amt_derived'],
    atol=0.01
)

negative_sum_df.to_csv('testing.csv', index=False)

percentage_true = calculate_percentage_true(negative_sum_df, 'check_relationship')
print(f"Percentage where relationship holds true: {percentage_true:.2f}%")
# 15.16%

# Filter for lines where check_relationship is TRUE
negative_sum_check_true_mask = (
    negative_sum_df['check_relationship'] == True
)

negative_sum_check_true_df = save_and_summarize(
    negative_sum_df, 
    negative_sum_check_true_mask, 
    'testing.csv',
    'negative_sum_check_true'
)

# Set the main delivery columns
negative_sum_check_true_df['undiscounted_price'] = negative_sum_check_true_df['line_gross_amt_received']
negative_sum_check_true_df['discounted_price'] = negative_sum_check_true_df['line_net_amt_derived']
negative_sum_check_true_df['flag'] = 'negative_sum_off' 

# Save to CSV
negative_sum_check_true_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD negative_sum_off FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    negative_sum_check_true_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 95,061 rows, or 16.34% flagged

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)


# ============================================================================================================
# FILTER OUT x_for_the_price_of_1 FROM imputed_invoice_line_item_df
# ============================================================================================================

# Filter for lines where discount_offered is recorded as a percentage to take off
x_for_1_mask = (
        no_flags_df['discount_offered'] < 0
)

x_for_1_df = save_and_summarize(
    no_flags_df, 
    x_for_1_mask, 
    'testing.csv',
    'x_for_1'
)

# Check the math to calculate undiscounted price. line_gross_amt_derived is assumed to be undiscounted price
x_for_1_df['check_relationship'] = np.isclose(
    x_for_1_df['unit_gross_amt_received'] * (x_for_1_df['discount_offered'] / 100 - 1) * -1,
    x_for_1_df['line_net_amt_received'],
    atol=0.01
)

# Check if two columns are the same
x_for_1_df['check_same'] = (
    x_for_1_df['unit_gross_amt_received'].round(2) == x_for_1_df['line_gross_amt_received'].round(2)
)

calculate_percentage_true(x_for_1_df, 'check_same')
# They are same

x_for_1_df.to_csv('testing.csv', index=False)

percentage_true = calculate_percentage_true(x_for_1_df, 'check_relationship')
print(f"Percentage where relationship holds true: {percentage_true:.2f}%")
# 99.38%

# Filter for lines where check_relationship is TRUE
x_for_1_check_true_mask = (
    x_for_1_df['check_relationship'] == True
)

x_for_1_check_true_df = save_and_summarize(
    x_for_1_df, 
    x_for_1_check_true_mask, 
    'testing.csv',
    'x_for_1_check_true'
)

# Set the main delivery columns
x_for_1_check_true_df['undiscounted_price'] = x_for_1_check_true_df['line_net_amt_received']
x_for_1_check_true_df['discounted_price'] = x_for_1_check_true_df['line_gross_amt_received']
x_for_1_check_true_df['flag'] = 'x_for_1' 

# Save to CSV
x_for_1_check_true_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD x_for_1 FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    x_for_1_check_true_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 100,061 rows, or 17.20% flagged

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER OUT amt_off_total FROM imputed_invoice_line_item_df
# ============================================================================================================

# Filter for lines where discount_offered is recorded as a percentage to take off
amt_off_total_mask = (
    np.isclose(
        no_flags_df['line_gross_amt_derived'] - no_flags_df['line_discount_derived'],
        no_flags_df['line_net_amt_derived'],
        atol=0.01
        ) &
        no_flags_df['discount_offered'].notnull()
)

amt_off_total_df = save_and_summarize(
    no_flags_df, 
    amt_off_total_mask, 
    'testing.csv',
    'amt_off_total'
)

# no need to check math this time

# Set the main delivery columns
amt_off_total_df['undiscounted_price'] = amt_off_total_df['line_gross_amt_derived']
amt_off_total_df['discounted_price'] = amt_off_total_df['line_net_amt_derived']
amt_off_total_df['flag'] = 'amt_off_total' 

# Save to CSV
amt_off_total_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD amt_off_total FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    amt_off_total_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 237,814 rows or 40.88% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for discount_offered = null, found line_discount_derived notnull instead
# ============================================================================================================

amt_off_total_derived_mask = (
    no_flags_df['discount_offered'].isna()
)

amt_off_total_derived_df = save_and_summarize(
    no_flags_df, 
    amt_off_total_derived_mask, 
    'testing.csv',
    'discount_null'
)

# Check the math to calculate undiscounted price. line_gross_amt_derived is assumed to be undiscounted price
amt_off_total_derived_df['check_relationship'] = np.isclose(
    amt_off_total_derived_df['line_net_amt_derived'] + amt_off_total_derived_df['line_discount_derived'],
    amt_off_total_derived_df['line_gross_amt_derived'],
    atol=0.01
)

percentage_true = calculate_percentage_true(amt_off_total_derived_df, 'check_relationship')
print(f"Percentage where relationship holds true: {percentage_true:.2f}%")
# 30.00%

# Save to CSV
amt_off_total_derived_df.to_csv('testing.csv', index=False)

# Filter for lines where check_relationship is TRUE
amt_off_total_derived_mask = (
    amt_off_total_derived_df['check_relationship'] == True
)

amt_off_total_derived_df = save_and_summarize(
    amt_off_total_derived_df, 
    amt_off_total_derived_mask, 
    'testing.csv',
    'x_for_1_check_true'
)

# Set the main delivery columns
amt_off_total_derived_df['undiscounted_price'] = amt_off_total_derived_df['line_gross_amt_derived']
amt_off_total_derived_df['discounted_price'] = amt_off_total_derived_df['line_net_amt_derived']
amt_off_total_derived_df['flag'] = 'amt_off_total' 

# Save to CSV
amt_off_total_derived_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD amt_off_total FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    amt_off_total_derived_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 339,358 rows or 58.34% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for discount_offered.isnull() AND line_discount_derived.isnull()
# ============================================================================================================

discount_null_assumed_zero_mask = (
    no_flags_df['discount_offered'].isna() &
    no_flags_df['line_discount_derived'].isna()
)

discount_null_assumed_zero_df = save_and_summarize(
    no_flags_df, 
    discount_null_assumed_zero_mask, 
    'testing.csv',
    'discounts_null'
)

# seems like a pattern where a bunch of rows have only quantity and line_gross_amt_received
# therefore just use line_gross_amt_received as the undiscounted price
fuel_no_discount_mask = (
    discount_null_assumed_zero_df['line_gross_amt_derived'].isna() &
    discount_null_assumed_zero_df['line_net_amt_derived'].isna() &
    discount_null_assumed_zero_df['unit_gross_amt_derived'].isna() &
    discount_null_assumed_zero_df['line_net_amt_received'].isna() &
    discount_null_assumed_zero_df['quantity'].notnull() &
    discount_null_assumed_zero_df['line_gross_amt_received'].notnull()
)

fuel_no_discount_df = save_and_summarize(
    discount_null_assumed_zero_df, 
    fuel_no_discount_mask, 
    'testing.csv',
    'fuel_no_discount'
)

# Inspect dataframe
results = analyze_dataframe(
    fuel_no_discount_df, 
    df_name='stats', 
    output_filename='stats.csv'
)
# no others values columns have any information

# Set the main delivery columns
fuel_no_discount_df['undiscounted_price'] = fuel_no_discount_df['line_gross_amt_received']
fuel_no_discount_df['discounted_price'] = fuel_no_discount_df['line_gross_amt_received']
fuel_no_discount_df['flag'] = 'discount_assumed_zero' 

# Save to CSV
fuel_no_discount_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD amt_off_total FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    fuel_no_discount_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 538,990 rows or 92.66% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for free_gift
# ============================================================================================================

free_gift_mask = (
    (no_flags_df['quantity'] * no_flags_df['line_gross_amt_received'] + 
     no_flags_df['discount_offered']== 0) &
     (no_flags_df['line_net_amt_derived'] == 0)
)

free_gift_df = save_and_summarize(
    no_flags_df, 
    free_gift_mask, 
    'testing.csv',
    'free_gift'
)

# Set the main delivery columns
free_gift_df['undiscounted_price'] = free_gift_df['line_gross_amt_received']
free_gift_df['discounted_price'] = free_gift_df['line_net_amt_derived']
free_gift_df['flag'] = 'free_gift' 

# Save to CSV
free_gift_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD amt_off_total FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    free_gift_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 539,002 rows or 92.66% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for amt_off_total
# ============================================================================================================

amt_off_total_maks2 = (
    ((no_flags_df['line_gross_amt_received'] - no_flags_df['discount_offered']) ==
    no_flags_df['line_net_amt_received'])
)

amt_off_total_df2 = save_and_summarize(
    no_flags_df, 
    amt_off_total_maks2, 
    'testing.csv',
    'ame_off_total'
)

# Set the main delivery columns
amt_off_total_df2['undiscounted_price'] = amt_off_total_df2['line_gross_amt_received']
amt_off_total_df2['discounted_price'] = amt_off_total_df2['line_net_amt_received']
amt_off_total_df2['flag'] = 'amt_off_total' 

# Save to CSV
amt_off_total_df2.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD amt_off_total FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    amt_off_total_df2, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 539,465 rows or 92.74% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for discount_assumed_zero2
# ============================================================================================================

discount_assumed_zero_mask2 = (
    no_flags_df['discount_offered'].isna() &
    no_flags_df['line_discount_derived'].isna() &
    (no_flags_df['line_gross_amt_received'] + no_flags_df['line_gst_amt_received'] ==
     no_flags_df['line_net_amt_derived'])
)

discount_assumed_zero_df2 = save_and_summarize(
    no_flags_df, 
    discount_assumed_zero_mask2, 
    'testing.csv',
    'discount_assumed_zero'
)

# check line_net and line_gross (derived) are equal for all rows
check_equal = discount_assumed_zero_df2['line_gross_amt_derived'] == discount_assumed_zero_df2['line_net_amt_derived']
print(f"Equal: {check_equal.sum():,} / {len(check_equal):,} ({check_equal.mean()*100:.2f}%)")
print(f"All equal? {check_equal.all()}")
# all are equal so use either for prices

# Set the main delivery columns
discount_assumed_zero_df2['undiscounted_price'] = discount_assumed_zero_df2['line_net_amt_derived']
discount_assumed_zero_df2['discounted_price'] = discount_assumed_zero_df2['discounted_price']
discount_assumed_zero_df2['flag'] = 'discount_assumed_zero' 

# Save to CSV
discount_assumed_zero_df2.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD discount_assumed_zero2 FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    discount_assumed_zero_df2, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 541,940 rows or 93.17% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for percentage_off2
# ============================================================================================================

percentage_off_mask2 = (
    np.isclose(
        no_flags_df['line_net_amt_received'],
        (no_flags_df['line_gross_amt_received'] * (1 - no_flags_df['discount_offered']/100)),
        atol=0.01
    ) & 
    (no_flags_df['line_net_amt_received'] != 0)
)

percetange_off_df2 = save_and_summarize(
    no_flags_df, 
    percentage_off_mask2, 
    'testing.csv',
    'percetange_off_df2'
)

# Set the main delivery columns
percetange_off_df2['undiscounted_price'] = percetange_off_df2['line_gross_amt_received']
percetange_off_df2['discounted_price'] = percetange_off_df2['line_net_amt_received']
percetange_off_df2['flag'] = 'percentage_off' 

# Save to CSV
percetange_off_df2.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD discount_assumed_zero2 FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    percetange_off_df2, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 546,091 rows or 93.88% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for amt_off_total3
# ============================================================================================================

amt_off_total3_mask = (
    np.isclose(
        no_flags_df['line_net_amt_received'],
        (no_flags_df['line_gross_amt_received'] - no_flags_df['discount_offered']),
        atol=0.01
    )
)

amt_off_total3_df = save_and_summarize(
    no_flags_df, 
    amt_off_total3_mask, 
    'testing.csv',
    'amt_off_total3_df'
)

# Set the main delivery columns
amt_off_total3_df['undiscounted_price'] = amt_off_total3_df['line_gross_amt_received']
amt_off_total3_df['discounted_price'] = amt_off_total3_df['line_net_amt_received']
amt_off_total3_df['flag'] = 'amt_off_total' 

# Save to CSV
amt_off_total3_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD amt_off_total3 FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    amt_off_total3_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 546,434 rows or 93.94% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for negative_sum_off
# ============================================================================================================

negative_sum_off_mask = (
    np.isclose(
        no_flags_df['line_gross_amt_derived'],
        (no_flags_df['line_net_amt_derived'] - no_flags_df['discount_offered']),
        atol=0.01
    ) & 
    (no_flags_df['line_gross_amt_derived'] < 0)

)

negative_sum_off_df = save_and_summarize(
    no_flags_df, 
    negative_sum_off_mask, 
    'testing.csv',
    'negative_sum_off_df'
)

# Set the main delivery columns
negative_sum_off_df['undiscounted_price'] = negative_sum_off_df['line_gross_amt_derived']
negative_sum_off_df['discounted_price'] = negative_sum_off_df['line_net_amt_derived']
negative_sum_off_df['flag'] = 'negative_sum_off' 

# Save to CSV
negative_sum_off_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD negative_sum_off FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    negative_sum_off_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 546,772 rows or 94.00% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for percentage_off
# ============================================================================================================

percentage_off_mask3 = (
    np.isclose(
        no_flags_df['line_gross_amt_received'],
        no_flags_df['quantity'] * no_flags_df['unit_gross_amt_received'] * (1 - no_flags_df['discount_offered'] / 100),
        atol=0.01
    ) 
)

percentage_off_df3 = save_and_summarize(
    no_flags_df, 
    percentage_off_mask3, 
    'testing.csv',
    'percentage_off_df3'
)

# Set the main delivery columns
percentage_off_df3['undiscounted_price'] = percentage_off_df3['line_gross_amt_received']
percentage_off_df3['discounted_price'] = percentage_off_df3['quantity'] * percentage_off_df3['unit_gross_amt_received']
percentage_off_df3['flag'] = 'percentage_off' 

# Save to CSV
percentage_off_df3.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD percentage_off FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    percentage_off_df3, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 546,910 rows or 94.02% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for negative_price_no_discount
# ============================================================================================================

negative_price_no_discount_mask = (
    np.isclose(
        no_flags_df['line_gross_amt_derived'],
        no_flags_df['line_gross_amt_received'] - no_flags_df['line_gst_amt_received'],
        atol=0.01
    ) &
    (no_flags_df['line_gross_amt_derived'] < 0)
)

negative_price_no_discount_df = save_and_summarize(
    no_flags_df, 
    negative_price_no_discount_mask, 
    'testing.csv',
    'negative_price_no_discount_df'
)

# Set the main delivery columns
negative_price_no_discount_df['undiscounted_price'] = negative_price_no_discount_df['line_gross_amt_derived']
negative_price_no_discount_df['discounted_price'] = negative_price_no_discount_df['undiscounted_price']
negative_price_no_discount_df['flag'] = 'negative_price_no_discount' 

# Save to CSV
negative_price_no_discount_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD negative_price_no_discount FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    negative_price_no_discount_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 546,994 rows or 94.03% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for discount_zero
# ============================================================================================================

discount_zero_mask3 = (
    no_flags_df['line_gross_amt_received'].notnull() &
    no_flags_df['quantity'].notnull() &
    no_flags_df['unit_gross_amt_received'].notnull() &
    no_flags_df['line_gst_amt_received'].notnull() &
    no_flags_df['line_gross_amt_derived'].notnull() &
    no_flags_df['discount_offered'].isna() &
    (no_flags_df['description'].str.lower() != 'charges')
)

discount_zero_df3 = save_and_summarize(
    no_flags_df, 
    discount_zero_mask3, 
    'testing.csv',
    'discount_zero_df3'
)

# check relationship
discount_zero_df3['check_relationship'] = np.isclose(
    discount_zero_df3['line_gross_amt_received'] + discount_zero_df3['line_gst_amt_received'],
    discount_zero_df3['line_gross_amt_derived'],
    atol = 0.01
)
# Check if all values are True
all_true = discount_zero_df3['check_relationship'].all()
print(f"All relationships correct? {all_true}")

# Save to CSV
discount_zero_df3.to_csv('testing.csv', index=False)

# Set the main delivery columns
discount_zero_df3['undiscounted_price'] = discount_zero_df3['line_gross_amt_derived']
discount_zero_df3['discounted_price'] = discount_zero_df3['undiscounted_price']
discount_zero_df3['flag'] = 'discount_zero' 

# Save to CSV
discount_zero_df3.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD discount_zero FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    discount_zero_df3, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 547,426 rows or 94.11% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for free_gift2
# ============================================================================================================

free_gift_mask2 = (
    pd.to_numeric(no_flags_df['discount_type'], errors='coerce').notna() &
    no_flags_df['line_gst_amt_received'].notna()
)

free_gift_df2 = save_and_summarize(
    no_flags_df, 
    free_gift_mask2, 
    'testing.csv',
    'free_gift_df2'
)

# check relationship
free_gift_df2['check_relationship'] = np.isclose(
    free_gift_df2['discount_type'],
    free_gift_df2['line_gst_amt_received'],
    atol = 0.01
)

# Check if all values are True
all_true = free_gift_df2['check_relationship'].all()
print(f"All relationships correct? {all_true}")

# check price zero for all
free_gift_df2['check_zero'] = (free_gift_df2['line_net_amt_derived'] == 0)

# Check if all values are True
all_true = free_gift_df2['check_zero'].all()
print(f"All values true? {all_true}")

# Save to CSV
free_gift_df2.to_csv('testing.csv', index=False)

# Set the main delivery columns
free_gift_df2['undiscounted_price'] = discount_zero_df3['discount_type']
free_gift_df2['discounted_price'] = discount_zero_df3['line_net_amt_derived']
free_gift_df2['flag'] = 'free_gift' 

# Save to CSV
free_gift_df2.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD free_gift2 FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    free_gift_df2, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 548,154 rows or 94.23% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

# ============================================================================================================
# FILTER for discount_zero4
# ============================================================================================================

discount_zero_mask4 = (
    no_flags_df['line_net_amt_derived'].notnull() &
    no_flags_df['line_gross_amt_derived'].notnull() &
    no_flags_df['discount_offered'].isna()
)

discount_zero_df4 = save_and_summarize(
    no_flags_df, 
    discount_zero_mask4, 
    'testing.csv',
    'discount_zero_df4'
)

# check price zero for all
discount_zero_df4['check_same'] = (discount_zero_df4['line_gross_amt_derived'] == discount_zero_df4['line_net_amt_derived'])

# Check if all values are True
all_true = discount_zero_df4['check_same'].all()
print(f"All values true? {all_true}")

# Save to CSV
discount_zero_df4.to_csv('testing.csv', index=False)

# Set the main delivery columns
discount_zero_df4['undiscounted_price'] = discount_zero_df4['line_gross_amt_derived']
discount_zero_df4['discounted_price'] = discount_zero_df4['undiscounted_price']
discount_zero_df4['flag'] = 'discount_zero' 

# Save to CSV
discount_zero_df4.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD free_gift2 FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    discount_zero_df4, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 548,618 rows or 94.31% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)


# ============================================================================================================
# FILTER for complimentary
# ============================================================================================================

complimentary_mask = (
    (no_flags_df['line_net_amt_received'] == 0) &
    no_flags_df['line_gross_amt_received'].notnull() &
    (no_flags_df['discount_offered'] == 100)
)

complimentary_df = save_and_summarize(
    no_flags_df, 
    complimentary_mask, 
    'testing.csv',
    'complimentary_df'
)

# Set the main delivery columns
complimentary_df['undiscounted_price'] = complimentary_df['line_gross_amt_received']
complimentary_df['discounted_price'] = complimentary_df['line_net_amt_received']
complimentary_df['flag'] = 'complimentary' 

# Save to CSV
complimentary_df.to_csv('testing.csv', index=False)

# ============================================================================================================
# ADD free_gift2 FLAG TO MAIN DATAFRAME

merge_updates_to_main_df(
    imputed_invoice_line_item_df, 
    complimentary_df, 
    ['undiscounted_price', 'discounted_price', 'flag']
)
# 548,775 rows or 94.34% done

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

no_flags_mask = (
    imputed_invoice_line_item_df['flag'].isnull()
)

no_flags_df = imputed_invoice_line_item_df[no_flags_mask].copy()
# Save to CSV
no_flags_df.to_csv('no_flags_df.csv', index=False)

