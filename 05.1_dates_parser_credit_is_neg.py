import pandas as pd
import numpy as np
from pathlib import Path

# Define base directory
base_dir = Path("p:/Desktop/2025/data605/2025-Summerproject-RuralCo3")

# Read the final CSVs from your original code
ats_df = pd.read_csv('datetime_parsed_ats_invoice_line_item_df.csv')
invoice_df = pd.read_csv('datetime_parsed_invoice_line_item_df.csv')

print("Original data loaded successfully!")
print(f"ATS rows: {len(ats_df)}")
print(f"Invoice rows: {len(invoice_df)}")

def transform_prices_by_indicator(df, df_name):
    """
    Transform undiscounted_price and discounted_price based on debit_credit_indicator:
    - If indicator is 'D' (Debit), ensure prices are positive
    - If indicator is 'C' (Credit), ensure prices are negative
    """
    df_transformed = df.copy()
    
    # Check if required columns exist
    required_cols = ['debit_credit_indicator', 'undiscounted_price', 'discounted_price']
    missing_cols = [col for col in required_cols if col not in df_transformed.columns]
    
    if missing_cols:
        print(f"Warning: {df_name} is missing columns: {missing_cols}")
        return df_transformed
    
    print(f"\n{df_name} - Before transformation:")
    print(f"Debit_credit_indicator value counts:")
    print(df_transformed['debit_credit_indicator'].value_counts())
    
    # Count rows that will be affected
    debit_mask = df_transformed['debit_credit_indicator'] == 'D'
    credit_mask = df_transformed['debit_credit_indicator'] == 'C'
    
    debit_negative_undiscounted = (debit_mask & (df_transformed['undiscounted_price'] < 0)).sum()
    debit_negative_discounted = (debit_mask & (df_transformed['discounted_price'] < 0)).sum()
    credit_positive_undiscounted = (credit_mask & (df_transformed['undiscounted_price'] > 0)).sum()
    credit_positive_discounted = (credit_mask & (df_transformed['discounted_price'] > 0)).sum()
    
    print(f"\nRows requiring transformation:")
    print(f"  Debit (D) with negative undiscounted_price: {debit_negative_undiscounted}")
    print(f"  Debit (D) with negative discounted_price: {debit_negative_discounted}")
    print(f"  Credit (C) with positive undiscounted_price: {credit_positive_undiscounted}")
    print(f"  Credit (C) with positive discounted_price: {credit_positive_discounted}")
    
    # Apply transformations
    # For Debit (D): ensure positive values
    df_transformed.loc[debit_mask, 'undiscounted_price'] = df_transformed.loc[debit_mask, 'undiscounted_price'].abs()
    df_transformed.loc[debit_mask, 'discounted_price'] = df_transformed.loc[debit_mask, 'discounted_price'].abs()
    
    # For Credit (C): ensure negative values
    df_transformed.loc[credit_mask, 'undiscounted_price'] = -df_transformed.loc[credit_mask, 'undiscounted_price'].abs()
    df_transformed.loc[credit_mask, 'discounted_price'] = -df_transformed.loc[credit_mask, 'discounted_price'].abs()
    
    print(f"\n{df_name} - After transformation:")
    print(f"  Debit (D) rows with negative undiscounted_price: {(debit_mask & (df_transformed['undiscounted_price'] < 0)).sum()}")
    print(f"  Debit (D) rows with negative discounted_price: {(debit_mask & (df_transformed['discounted_price'] < 0)).sum()}")
    print(f"  Credit (C) rows with positive undiscounted_price: {(credit_mask & (df_transformed['undiscounted_price'] > 0)).sum()}")
    print(f"  Credit (C) rows with positive discounted_price: {(credit_mask & (df_transformed['discounted_price'] > 0)).sum()}")
    
    return df_transformed

# Transform both dataframes
print("\n" + "="*80)
print("TRANSFORMING ATS DATAFRAME")
print("="*80)
ats_df_transformed = transform_prices_by_indicator(ats_df, "ATS")

print("\n" + "="*80)
print("TRANSFORMING INVOICE DATAFRAME")
print("="*80)
invoice_df_transformed = transform_prices_by_indicator(invoice_df, "Invoice")

# Save the transformed dataframes
ats_output_file = 'datetime_parsed_ats_invoice_line_item_df_transformed.csv'
invoice_output_file = 'datetime_parsed_invoice_line_item_df_transformed.csv'

ats_df_transformed.to_csv(ats_output_file, index=False)
invoice_df_transformed.to_csv(invoice_output_file, index=False)

print("\n" + "="*80)
print("TRANSFORMATION COMPLETE!")
print("="*80)
print(f"Transformed ATS data saved to: {ats_output_file}")
print(f"Transformed Invoice data saved to: {invoice_output_file}")

# Show sample of transformed data
print("\n" + "="*80)
print("SAMPLE OF TRANSFORMED DATA")
print("="*80)
print("\nATS Sample (first 10 rows):")
print(ats_df_transformed[['debit_credit_indicator', 'undiscounted_price', 'discounted_price']].head(10))

print("\nInvoice Sample (first 10 rows):")
print(invoice_df_transformed[['debit_credit_indicator', 'undiscounted_price', 'discounted_price']].head(10))