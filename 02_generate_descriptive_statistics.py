'''
Docstring for 02_generate_descriptive_statistics

this script is just for generating some data overview stats
'''

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Define base directories (using same format as other scripts)
base_dir = Path("T:/projects/2025/RuralCo")
invoices_dir = base_dir / "Data provided by RuralCo 20251202/invoices_export/20251121"
ruralco2_dir = base_dir / "Data provided by RuralCo 20251202/RuralCo2/Clean Code"
profile_dir = base_dir / "Data provided by RuralCo 20251202/RuralCo3/payment_profile"
data_cleaning_dir = base_dir / "Data provided by RuralCo 20251202/RuralCo3/data_cleaning"

# Create output directory if it doesn't exist
data_cleaning_dir.mkdir(parents=True, exist_ok=True)

# Define file paths
file_paths = {
    'invoice_line_item': invoices_dir / "invoice_line_item.csv",
    'invoice': invoices_dir / "invoice.csv", 
    'ats_invoice_line_item': invoices_dir / "ats_invoice_line_item.csv",
    'ats_invoice': invoices_dir / "ats_invoice.csv",
    'master_data_complete': profile_dir / "master_dataset_complete.csv"
}

# Load the invoice files
def load_invoice_files():
    """Load all required invoice files"""
    data = {}
    
    for name, filepath in file_paths.items():
        if filepath.exists():
            print(f"Loading {name}...")
            if name == 'master_data_complete':
                data[name] = pd.read_csv(filepath)
            else:
                data[name] = pd.read_csv(filepath, low_memory=False)
            print(f"  Loaded {name}: {len(data[name])} rows, {len(data[name].columns)} columns")
        else:
            print(f"Warning: {filepath} not found")
            data[name] = None
    
    return data

print("="*60)
print("Loading invoice files...")
print("="*60)
all_data = load_invoice_files()

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

def generate_aesthetic_table(data_dict):
    """
    Generate an aesthetic table with dataset overview statistics.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are dataset names and values are DataFrames
        
    Returns:
    --------
    pd.DataFrame
        Formatted table with number of rows, columns, and mean column sparsity
    """
    results = []
    
    for name, df in data_dict.items():
        if df is not None:
            # Calculate number of rows and columns
            num_rows = len(df)
            num_columns = len(df.columns)
            
            # Calculate mean column sparsity (percentage of missing values across all columns)
            sparsity_per_column = df.isnull().sum() / num_rows * 100
            mean_sparsity = sparsity_per_column.mean()
            
            results.append({
                'Dataset': name,
                'Number of Rows': f"{num_rows:,}",
                'Number of Columns': num_columns,
                'Mean Column Sparsity (%)': f"{mean_sparsity:.2f}%"
            })
        else:
            results.append({
                'Dataset': name,
                'Number of Rows': 'File not found',
                'Number of Columns': 'File not found',
                'Mean Column Sparsity (%)': 'File not found'
            })
    
    results_df = pd.DataFrame(results)
    
    # Create a more aesthetic display
    print("\n" + "="*80)
    print("DATASET OVERVIEW STATISTICS")
    print("="*80)
    print()
    
    # Display the table with proper formatting
    for _, row in results_df.iterrows():
        print(f"📊 {row['Dataset'].upper().replace('_', ' ')}")
        print(f"   Rows: {row['Number of Rows']}")
        print(f"   Columns: {row['Number of Columns']}")
        print(f"   Mean Sparsity: {row['Mean Column Sparsity (%)']}")
        print("-" * 50)
    
    return results_df

# Generate the aesthetic overview table
overview_table = generate_aesthetic_table(all_data)

# Save the overview table to CSV in data_cleaning directory
overview_output_path = data_cleaning_dir / '02_dataset_overview_statistics.csv'
overview_table.to_csv(overview_output_path, index=False)
print(f"\n✓ Overview table saved to: {overview_output_path}")

# Generate detailed descriptive statistics (keeping original functionality)
detailed_output_path = data_cleaning_dir / 'original_missing_values_summary.csv'
original_results = analyze_dataframes(all_data, detailed_output_path)
print(f"✓ Detailed analysis saved to: {detailed_output_path}")

# Save the function to data_cleaning directory
function_output_path = data_cleaning_dir / 'analyze_dataframe.pkl'
with open(function_output_path, 'wb') as f:
    pickle.dump(analyze_dataframe, f)
print(f"✓ Function saved to: {function_output_path}")

