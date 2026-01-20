'''
Docstring for 05.1_dates_parser_credit_is_neg

this script is simply standardising credit and negative values. 
according to the column variable debit_credit_indicator, debits are to be positive, and credits are to be negative
it also clears periods with incomplete invoice data, as signalled by invoice count <20,000

inputs:
- datetime_parsed_ats_invoice_line_item_df.csv
- datetime_parsed_invoice_line_item_df.csv

outputs:
- datetime_parsed_ats_invoice_line_item_df_transformed.csv
- datetime_parsed_invoice_line_item_df_transformed.csv
- invoice_period_monthly_counts-transformed.csv
- date_from_invoice_monthly_counts-transformed.csv
'''

import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
output_dir = base_dir / "data_cleaning"
output_dir.mkdir(exist_ok=True)

# Read the final CSVs from your original code
ats_df = pd.read_csv(output_dir / 'datetime_parsed_ats_invoice_line_item_df.csv')
invoice_df = pd.read_csv(output_dir / 'datetime_parsed_invoice_line_item_df.csv')

print("Original data loaded successfully!")
print(f"ATS rows: {len(ats_df)}")
print(f"Invoice rows: {len(invoice_df)}")

# ================================================================
# Filter by invoice_period date range to clear out incomplete data
# ================================================================
print("\n" + "="*80)
print("FILTERING BY INVOICE PERIOD DATE RANGE")
print("="*80)

# Define date range
START_DATE = pd.Timestamp("2024-02-01")
END_DATE = pd.Timestamp("2025-12-01")

print(f"Keeping invoices between {START_DATE.strftime('%Y-%m-%d')} and {END_DATE.strftime('%Y-%m-%d')}")

# Convert invoice_period to datetime (if not already)
ats_df['invoice_period'] = pd.to_datetime(ats_df['invoice_period'], errors='coerce')
invoice_df['invoice_period'] = pd.to_datetime(invoice_df['invoice_period'], errors='coerce')

# Filter ATS
print(f"\nATS DataFrame:")
print(f"  Before filtering: {len(ats_df):,} rows")
ats_df = ats_df[
    (ats_df['invoice_period'] >= START_DATE) & 
    (ats_df['invoice_period'] <= END_DATE)
].copy()
print(f"  After filtering: {len(ats_df):,} rows")
print(f"  Removed: {len(ats_df) - len(ats_df):,} rows")

# Filter Invoice
print(f"\nInvoice DataFrame:")
print(f"  Before filtering: {len(invoice_df):,} rows")
invoice_df_before = len(invoice_df)
invoice_df = invoice_df[
    (invoice_df['invoice_period'] >= START_DATE) & 
    (invoice_df['invoice_period'] <= END_DATE)
].copy()
print(f"  After filtering: {len(invoice_df):,} rows")
print(f"  Removed: {invoice_df_before - len(invoice_df):,} rows")

print("="*80)

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

ats_df_transformed.to_csv(output_dir / ats_output_file, index=False)
invoice_df_transformed.to_csv(output_dir / invoice_output_file, index=False)

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

# ============================================================================================================
# CREATE SUMMARY DATAFRAMES FOR INVOICE COUNTS PER MONTH
# ============================================================================================================
# For ATS dataframe
print("\n" + "="*80)
print("INVOICE COUNT SUMMARY - ATS")
print("="*80)

# Convert to datetime for grouping
ats_invoice_period_dt = pd.to_datetime(ats_df_transformed['invoice_period'])
ats_date_from_invoice_dt = pd.to_datetime(ats_df_transformed['date_from_invoice'])
# Count by invoice_period (month)

ats_invoice_period_counts = ats_df_transformed.groupby(
    ats_invoice_period_dt.dt.to_period('M')
).size().reset_index(name='count')
ats_invoice_period_counts.columns = ['month', 'invoice_count']
ats_invoice_period_counts['month'] = ats_invoice_period_counts['month'].astype(str)
# Count by date_from_invoice (month)

ats_date_from_invoice_counts = ats_df_transformed.groupby(
    ats_date_from_invoice_dt.dt.to_period('M')
).size().reset_index(name='count')
ats_date_from_invoice_counts.columns = ['month', 'invoice_count']
ats_date_from_invoice_counts['month'] = ats_date_from_invoice_counts['month'].astype(str)

# Save ATS summary dataframes with -transformed suffix
ats_invoice_period_counts.to_csv(output_dir / 'ats_invoice_period_monthly_counts-transformed.csv', index=False)
ats_date_from_invoice_counts.to_csv(output_dir / 'ats_date_from_invoice_monthly_counts-transformed.csv', index=False)
print("\nATS - Invoices per month (by invoice_period):")
print(ats_invoice_period_counts.to_string(index=False))
print(f"\nTotal: {ats_invoice_period_counts['invoice_count'].sum()}")
print("\nATS - Invoices per month (by date_from_invoice):")
print(ats_date_from_invoice_counts.to_string(index=False))
print(f"\nTotal: {ats_date_from_invoice_counts['invoice_count'].sum()}")

# For Invoice dataframe
print("\n" + "="*80)
print("INVOICE COUNT SUMMARY - INVOICE")
print("="*80)

# Convert to datetime for grouping
invoice_period_dt = pd.to_datetime(invoice_df_transformed['invoice_period'])
date_from_invoice_dt = pd.to_datetime(invoice_df_transformed['date_from_invoice'])

# Count by invoice_period (month)
invoice_period_counts = invoice_df_transformed.groupby(
    invoice_period_dt.dt.to_period('M')
).size().reset_index(name='count')
invoice_period_counts.columns = ['month', 'invoice_count']
invoice_period_counts['month'] = invoice_period_counts['month'].astype(str)

# Count by date_from_invoice (month)
date_from_invoice_counts = invoice_df_transformed.groupby(
    date_from_invoice_dt.dt.to_period('M')
).size().reset_index(name='count')
date_from_invoice_counts.columns = ['month', 'invoice_count']
date_from_invoice_counts['month'] = date_from_invoice_counts['month'].astype(str)

# Save Invoice summary dataframes with -transformed suffix
invoice_period_counts.to_csv(output_dir / 'invoice_period_monthly_counts-transformed.csv', index=False)
date_from_invoice_counts.to_csv(output_dir / 'date_from_invoice_monthly_counts-transformed.csv', index=False)
print("\nINVOICE - Invoices per month (by invoice_period):")
print(invoice_period_counts.to_string(index=False))
print(f"\nTotal: {invoice_period_counts['invoice_count'].sum()}")
print("\nINVOICE - Invoices per month (by date_from_invoice):")
print(date_from_invoice_counts.to_string(index=False))
print(f"\nTotal: {date_from_invoice_counts['invoice_count'].sum()}")