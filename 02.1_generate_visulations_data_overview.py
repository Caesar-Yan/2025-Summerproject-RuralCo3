import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# Load your data
with open('all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

# Option 1: If you want to analyze your imputed data
# Comment out the above and use this instead:
# with open('imputed_all_data.pkl', 'rb') as f:  # or whatever you named it
#     all_data = pickle.load(f)

def create_sparsity_visualizations(data_dict, output_prefix='sparsity', exclude_tables=None):
    """
    Create sparsity visualizations for multiple dataframes.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary where keys are table names and values are DataFrames
    output_prefix : str
        Prefix for output filenames (default: 'sparsity')
    exclude_tables : list
        List of table names to exclude from analysis (default: None)
    """
    
    if exclude_tables is None:
        exclude_tables = []
    
    # Calculate sparsity for each column in each dataframe
    results = []
    
    for table_name, df in data_dict.items():
        # Skip excluded tables
        if table_name in exclude_tables:
            print(f"Skipping table: {table_name}")
            continue
            
        total_rows = len(df)
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100
            
            results.append({
                'table_name': table_name,
                'column_name': column,
                'sparsity': missing_percentage,
                'present': 100 - missing_percentage
            })
    
    sparsity_df = pd.DataFrame(results)
    
    # VISUALIZATION: Summary bar chart
    print("Creating summary comparison...")
    tables = sorted(sparsity_df['table_name'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    summary_data = []
    for table in tables:
        table_data = sparsity_df[sparsity_df['table_name'] == table]
        summary_data.append({
            'Table': table,
            'Complete (0%)': len(table_data[table_data['sparsity'] == 0]),
            'Low (<10%)': len(table_data[(table_data['sparsity'] > 0) & (table_data['sparsity'] < 10)]),
            'Medium (10-50%)': len(table_data[(table_data['sparsity'] >= 10) & (table_data['sparsity'] < 50)]),
            'High (50-90%)': len(table_data[(table_data['sparsity'] >= 50) & (table_data['sparsity'] < 90)]),
            'Very High (≥90%)': len(table_data[table_data['sparsity'] >= 90])
        })
    
    summary_df = pd.DataFrame(summary_data).set_index('Table')
    
    summary_df.plot(kind='bar', stacked=True, ax=ax, 
                    color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b'],
                    edgecolor='black', linewidth=0.5)
    
    ax.set_title('Distribution of Column Sparsity Levels by Table', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Table Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Columns', fontsize=12, fontweight='bold')
    ax.legend(title='Sparsity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_prefix}_summary.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SPARSITY SUMMARY STATISTICS")
    print("="*70)
    
    for table in tables:
        table_data = sparsity_df[sparsity_df['table_name'] == table]
        print(f"\n{table}:")
        print(f"  Total columns: {len(table_data)}")
        print(f"  Complete columns (0% sparsity): {len(table_data[table_data['sparsity'] == 0])}")
        print(f"  Empty columns (100% sparsity): {len(table_data[table_data['sparsity'] == 100])}")
        print(f"  Average sparsity: {table_data['sparsity'].mean():.2f}%")
        print(f"  Median sparsity: {table_data['sparsity'].median():.2f}%")
        
        if table_data['sparsity'].max() > 0:
            max_sparse = table_data[table_data['sparsity'] == table_data['sparsity'].max()]
            print(f"  Highest sparsity: {table_data['sparsity'].max():.2f}% ({max_sparse['column_name'].values[0]})")
    
    print("\n" + "="*70)
    print("✓ Visualization created successfully!")
    print("="*70)
    
    return sparsity_df


# Run the analysis
if __name__ == "__main__":
    print("Analyzing sparsity patterns in your datasets...\n")
    
    # Exclude failed_accounts and merchant_discount tables
    exclude_list = ['failed_accounts', 'merchant_discount']
    
    sparsity_results = create_sparsity_visualizations(
        all_data, 
        output_prefix='sparsity',
        exclude_tables=exclude_list
    )
    
    # Optionally save the sparsity data
    sparsity_results.to_csv('sparsity_analysis.csv', index=False)
    print(f"\n✓ Sparsity data saved to sparsity_analysis.csv")
