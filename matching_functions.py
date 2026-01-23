'''
Docstring for 000_functions

this is where the functions for the matching and data cleaning are stored


usage is by pasting in this:

# Import your custom functions
from matching_functions import (
    calculate_percentage_true,
    calculate_percentage_not_null,
    add_category,
    save_and_summarize,
    analyze_dataframes,
    analyze_dataframe,
    get_non_null_percentage,
    check_diff,
    merge_updates_to_main_df,
    filter_non_empty_column
)

'''



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

def save_and_summarize2(df, mask, filename='testing.csv', label='Filtered', output_dir=None):
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
    output_dir : Path or str
        Directory to save the file (default: None, saves to current directory)
    
    Returns:
    --------
    pd.DataFrame
        The filtered dataframe
    """
    # Apply mask and create a COPY
    filtered_df = df[mask].copy()
    
    # Handle output directory
    if output_dir is not None:
        from pathlib import Path
        output_path = Path(output_dir) / filename
    else:
        output_path = filename
    
    filtered_df.to_csv(output_path, index=False, mode='w')
    
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