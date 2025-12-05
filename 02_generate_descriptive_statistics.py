import pandas as pd
import numpy as np

results = []

for name, df in all_data.items():
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
results_df.to_csv('missing_values_summary.csv', index=False, mode='w')

print("Saved to missing_values_summary.csv")