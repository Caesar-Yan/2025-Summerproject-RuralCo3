'''
Docstring for 13_L3_matching
this script is continuing on from the work done in the 12_ scripts.
L1 and L2 matching was done based on merchant id numbers and parsing text fields in invoice data
L3 matching method is yet to be determined

Inputs:
- 12_invoice_line_items_undiscounted_matched_merchant_L1L2.csv
- Merchant Discount Detail.xlsx

Outputs:
- 13_matching_progress.csv
- 13_invoice_line_items_still_unmatched.csv
- 13_filtered_mask_WIP.csv
- 13_non_empty_merchant_branch.csv
- 13_mask.csv
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os

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
    # Apply mask and create a COPY
    filtered_df = df[mask].copy()  # Added .copy() here
    filtered_df.to_csv(output_dir / filename, index=False, mode='w')
    
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
    results_df.to_csv(output_dir / output_filename, index=False, mode='w')
    
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
    results_df.to_csv(output_dir / output_filename, index=False, mode='w')
    
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

def filter_non_empty_column(df, column_name, output_filename='filtered.csv', label='Filtered'):
    """
    Filter dataframe for rows with non-empty values in specified column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to filter
    column_name : str
        Name of the column to filter on
    output_filename : str
        Output CSV filename
    label : str
        Label for the print output
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe with non-empty values in specified column
    """
    mask = df[column_name].notna() & (df[column_name] != '')
    
    filtered_df = save_and_summarize(
        df, 
        mask, 
        output_filename,
        label
    )
    
    return filtered_df

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
merchant_folder_dir = base_dir / "merchant"
data_cleaning_dir = base_dir / "data_cleaning"
output_dir = merchant_folder_dir

# =========================================================
# LOAD FULL PARENT FILE
# =========================================================
full_df = pd.read_csv(data_cleaning_dir / '12_invoice_line_items_undiscounted_matched_merchant_L1L2.csv')

# Add identifier if needed
if 'Unnamed: 0' not in full_df.columns:
    full_df.insert(0, 'Unnamed: 0', range(len(full_df)))

print(f"\nLoaded full dataset: {len(full_df):,} rows")
print(f"\nInitial match_layer value counts:")
print(full_df['match_layer'].value_counts())

# =========================================================
# FILTER FOR UNMATCHED ROWS TO ANALYZE
# =========================================================
invoice_line_items_unmatched_df = full_df[full_df['match_layer'] == 'unmatched'].copy()

print(f"\nUnmatched rows: {len(invoice_line_items_unmatched_df):,}")

# =========================================================
# ALLIED MERCHANT ANALYSIS
# =========================================================
# Filter for rows with "Allied" in merchant_branch column
mask_allied = invoice_line_items_unmatched_df['merchant_branch'].str.contains('Allied', case=False, na=False)

allied_df = save_and_summarize(
    invoice_line_items_unmatched_df, 
    mask_allied, 
    '13_filtered_mask_WIP.csv',
    'Allied merchants'
)

# Check for "Allied Concrete" or "Allied Irwin" in merchant_branch
mask_allied_concrete = allied_df['merchant_branch'].str.contains('Allied Concrete', case=False, na=False)
mask_allied_irwin = allied_df['merchant_branch'].str.contains('Allied Irwin', case=False, na=False)

# Print counts
print(f"\nAllied Concrete matches: {mask_allied_concrete.sum():,}")
print(f"Allied Irwin matches: {mask_allied_irwin.sum():,}")

# Optionally, see the unique values that contain "Allied"
print("\nUnique merchant_branch values containing 'Allied':")
print(allied_df['merchant_branch'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH ALLIED FLAGS
# =========================================================
# Update the Allied rows in the FULL dataframe
mask_allied_full = full_df['merchant_branch'].str.contains('Allied', case=False, na=False) & (full_df['match_layer'] == 'unmatched')
full_df.loc[mask_allied_full, 'match_layer'] = 'L3_no_discount_offered'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_allied_full.sum():,} Allied rows to 'no_discount_offered'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize(
    full_df, 
    unmatched_mask, 
    '13_invoice_line_items_still_unmatched.csv',
    'Still unmatched after Allied update'
)

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Allied rows flagged as 'no_discount_found': {mask_allied_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

# Filter for non-empty merchant_branch and save
mask_non_empty_branch = invoice_line_items_unmatched_df['merchant_branch'].notna() & (remaining_unmatched_df['merchant_branch'] != '')

non_empty_branch_df = save_and_summarize(
    invoice_line_items_unmatched_df, 
    mask_non_empty_branch, 
    '13_mask.csv',
    'Non-empty merchant_branch'
)

# =========================================================
# FARMSIDE MERCHANT ANALYSIS
# =========================================================
# Filter for rows with "Farmside" in merchant_branch column
mask_farmside = non_empty_branch_df['merchant_branch'].str.contains('Farmside', case=False, na=False)

farmside_df = save_and_summarize(
    non_empty_branch_df, 
    mask_farmside, 
    '13_filtered_mask_WIP.csv',
    'Farmside merchants'
)

# Print unique values
print("\nUnique merchant_branch values containing 'Farmside':")
print(farmside_df['merchant_branch'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH FARMSIDE FLAGS
# =========================================================
# Update the Farmside rows in the FULL dataframe
mask_farmside_full = full_df['merchant_branch'].str.contains('Farmside', case=False, na=False) & (full_df['match_layer'] == 'unmatched')
full_df.loc[mask_farmside_full, 'match_layer'] = 'L3_no_discount_offered'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_farmside_full.sum():,} Farmside rows to 'L3_no_discount_offered'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize(
    full_df, 
    unmatched_mask, 
    '13_invoice_line_items_still_unmatched.csv',
    'Still unmatched after Farmside update'
)

print(f"\n{'='*70}")
print(f"FINAL SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Allied rows flagged: {mask_allied_full.sum():,}")
print(f"Farmside rows flagged: {mask_farmside_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")

non_empty_branch_df = filter_non_empty_column(
    remaining_unmatched_df,
    'merchant_branch',
    '13_non_empty_merchant_branch.csv',
    'Non-empty merchant_branch'
)

# =========================================================
# BLACKWOODS MERCHANT MATCHING
# =========================================================
# Load merchant reference data
MERCHANT_PATH = base_dir.parent / "Merchant Discount Detail.xlsx"
merchant_df = pd.read_excel(MERCHANT_PATH)

# Filter for Blackwoods ATS Number 100882
blackwoods_merchant = merchant_df[merchant_df['ATS Number'] == 100882].iloc[0]

# Filter for rows with "Blackwoods Protector" in merchant_identifier
mask_blackwoods = non_empty_branch_df['merchant_identifier'].str.contains('Blackwoods Protector', case=False, na=False)

blackwoods_df = save_and_summarize(
    non_empty_branch_df, 
    mask_blackwoods, 
    '13_filtered_mask_WIP.csv',
    'Blackwoods merchants'
)

print("\nUnique merchant_identifier values containing 'Blackwoods Protector':")
print(blackwoods_df['merchant_identifier'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH BLACKWOODS METADATA
# =========================================================
# Update the Blackwoods rows in the FULL dataframe
mask_blackwoods_full = (
    full_df['merchant_identifier'].str.contains('Blackwoods Protector', case=False, na=False) & 
    (full_df['match_layer'] == 'unmatched')
)

# Append merchant metadata
full_df.loc[mask_blackwoods_full, 'matched_ats_number'] = blackwoods_merchant['ATS Number']
full_df.loc[mask_blackwoods_full, 'matched_account_name'] = blackwoods_merchant['Account Name']
full_df.loc[mask_blackwoods_full, 'matched_discount_offered'] = blackwoods_merchant['Discount Offered']
full_df.loc[mask_blackwoods_full, 'matched_discount_detail'] = blackwoods_merchant['Discount Offered 2']
full_df.loc[mask_blackwoods_full, 'match_layer'] = 'L3_blackwoods'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_blackwoods_full.sum():,} Blackwoods rows")
print(f"Merchant: {blackwoods_merchant['Account Name']}")
print(f"ATS Number: {blackwoods_merchant['ATS Number']}")
print(f"Discount Offered: {blackwoods_merchant['Discount Offered']}")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize(
    full_df, 
    unmatched_mask, 
    '13_invoice_line_items_still_unmatched.csv',
    'Still unmatched after Blackwoods update'
)

# =========================================================
# METHVEN MOTORS MERCHANT MATCHING
# =========================================================
# Filter for Methven Motors ATS Number 101041
methven_merchant = merchant_df[merchant_df['ATS Number'] == 101041].iloc[0]

# Filter for rows with "Methven Motors" in merchant_branch
mask_methven = non_empty_branch_df['merchant_branch'].str.contains('Methven Motors', case=False, na=False)

methven_df = save_and_summarize(
    non_empty_branch_df, 
    mask_methven, 
    '13_filtered_mask_WIP.csv',
    'Methven Motors merchants'
)

print("\nUnique merchant_branch values containing 'Methven Motors':")
print(methven_df['merchant_branch'].unique())

# =========================================================
# UPDATE FULL DATAFRAME WITH METHVEN MOTORS METADATA
# =========================================================
# Update the Methven Motors rows in the FULL dataframe
mask_methven_full = (
    full_df['merchant_branch'].str.contains('Methven Motors', case=False, na=False) & 
    (full_df['match_layer'] == 'unmatched')
)

# Append merchant metadata
full_df.loc[mask_methven_full, 'matched_ats_number'] = methven_merchant['ATS Number']
full_df.loc[mask_methven_full, 'matched_account_name'] = methven_merchant['Account Name']
full_df.loc[mask_methven_full, 'matched_discount_offered'] = methven_merchant['Discount Offered']
full_df.loc[mask_methven_full, 'matched_discount_detail'] = methven_merchant['Discount Offered 2']
full_df.loc[mask_methven_full, 'match_layer'] = 'L3_methven_motors'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_methven_full.sum():,} Methven Motors rows")
print(f"Merchant: {methven_merchant['Account Name']}")
print(f"ATS Number: {methven_merchant['ATS Number']}")
print(f"Discount Offered: {methven_merchant['Discount Offered']}")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize(
    full_df, 
    unmatched_mask, 
    '13_invoice_line_items_still_unmatched.csv',
    'Still unmatched after Methven Motors update'
)

# =========================================================
# MOBIL MERCHANT ANALYSIS
# =========================================================
# Filter for rows with "Mobil" in merchant_branch column
mask_mobil = non_empty_branch_df['merchant_branch'].str.contains('Mobil', case=False, na=False)

mobil_df = save_and_summarize(
    non_empty_branch_df, 
    mask_mobil, 
    '13_filtered_mask_WIP.csv',
    'Mobil merchants'
)

# Print unique values
print("\nUnique merchant_branch values containing 'Mobil':")
print(mobil_df['merchant_branch'].unique())

# none of these are petrol that is discounted using ruralco mobil card

# =========================================================
# LABEL REMAINING NON-EMPTY MERCHANT_BRANCH AS L3_NO_DISCOUNT_OFFERED
# =========================================================
# Update rows with non-empty merchant_branch that are still unmatched
mask_non_empty_branch_full = (
    full_df['merchant_branch'].notna() & 
    (full_df['merchant_branch'] != '') & 
    (full_df['match_layer'] == 'unmatched')
)

# Update match_layer
full_df.loc[mask_non_empty_branch_full, 'match_layer'] = 'L3_no_discount_offered'

# Print update summary
print(f"\n{'='*70}")
print(f"Updated {mask_non_empty_branch_full.sum():,} rows with non-empty merchant_branch to 'L3_no_discount_offered'")
print(f"\nUpdated match_layer value counts:")
print(full_df['match_layer'].value_counts())
print(f"{'='*70}")

# =========================================================
# SAVE UPDATED FULL FILE
# =========================================================
full_df.to_csv(merchant_folder_dir / '13_matching_progress.csv', index=False)
print(f"\nSaved updated file to: 13_matching_progress.csv")

# =========================================================
# FILTER AND SAVE REMAINING UNMATCHED
# =========================================================
unmatched_mask = full_df['match_layer'] == 'unmatched'

remaining_unmatched_df = save_and_summarize(
    full_df, 
    unmatched_mask, 
    '13_invoice_line_items_still_unmatched.csv',
    'Still unmatched after L3_no_discount_offered update'
)

print(f"\n{'='*70}")
print(f"FINAL L3 MATCHING SUMMARY")
print(f"{'='*70}")
print(f"Total rows in dataset: {len(full_df):,}")
print(f"Rows labeled 'L3_no_discount_offered': {mask_non_empty_branch_full.sum():,}")
print(f"Remaining unmatched rows: {len(remaining_unmatched_df):,}")
print(f"{'='*70}")