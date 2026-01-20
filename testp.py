import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import dill

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
output_dir = base_dir / "data_cleaning"
output_dir.mkdir(exist_ok=True)

# Load the data
with open(base_dir / 'all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

def get_next_20th(date):
    """
    Given a date, return the next 20th day.
    - If date is on or before the 19th of the month, return 20th of same month
    - If date is on or after the 20th, return 20th of next month
    """
    if pd.isna(date):
        return pd.NaT
    
    date = pd.to_datetime(date)
    
    # Always go to next month's 20th
    if date.month == 12:
        return date.replace(year=date.year + 1, month=1, day=20)
    else:
        return date.replace(month=date.month + 1, day=20)

# Invoice export files (CSV)
imputed_files = {
    'imputed_ats_invoice_line_item': output_dir / "imputed_ats_invoice_line_item.csv",
    'imputed_invoice_line_item': output_dir / "imputed_invoice_line_item.csv"
}

# Read the CSV files into dataframes
imputed_ats_invoice_line_item_df = pd.read_csv(imputed_files['imputed_ats_invoice_line_item'])
imputed_invoice_line_item_df = pd.read_csv(imputed_files['imputed_invoice_line_item'])

datetime_parsed_ats_invoice_line_item_df = imputed_ats_invoice_line_item_df.copy()
datetime_parsed_invoice_line_item_df = imputed_invoice_line_item_df.copy()

# ============================================================================================================
# PARSE OUT JUST DATE INFO FROM CREATED_AT AND UPDATED_AT COLUMNS
# ============================================================================================================

def parse_date(df, columns, format='%d/%m/%Y'):
    """
    Parse datetime columns and create new formatted date columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing datetime columns to parse
    columns : list or str
        Column name(s) to parse - can be a single string or list of strings
    format : str, optional
        Output date format (default: '%d/%m/%Y' for dd/mm/yyyy)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with new parsed_* columns added in specified format
    """
    df_copy = df.copy()
    
    # Handle single column passed as string
    if isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col in df_copy.columns:
            # Parse to datetime
            parsed_datetime = pd.to_datetime(df_copy[col])
            # Create new column name
            newcol_name = f'parsed_{col}'
            # Format as dd/mm/yyyy (or specified format)
            df_copy[newcol_name] = parsed_datetime.dt.strftime(format)
        else:
            print(f"Warning: Column '{col}' not found in dataframe")
    
    return df_copy

# Apply the parsing function
datetime_parsed_ats_invoice_line_item_df = parse_date(datetime_parsed_ats_invoice_line_item_df, ['created_at', 'updated_at'])
datetime_parsed_invoice_line_item_df = parse_date(datetime_parsed_invoice_line_item_df, ['created_at', 'updated_at'])

datetime_parsed_ats_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_ats_invoice_line_item_df.csv', index=False)
datetime_parsed_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_invoice_line_item_df.csv', index=False)

# dataframes being used and referenced
datetime_parsed_ats_invoice_line_item_df = imputed_ats_invoice_line_item_df.copy()
datetime_parsed_invoice_line_item_df = imputed_invoice_line_item_df.copy()
ats_invoice_df = all_data.get('ats_invoice')
invoice_df = all_data.get('invoice')

# the plan is to get a date for every invoice item somehow
# almost every id from the invoice and ats_invoice tables have a date / transaction date
# take date from the invoice file [id] and match it to the line_item file [invoice_id]

# ============================================================================================================
# GETTING DATES FROM INVOICE DF INTO THE LINE_ITEM DF FOR ATS
# ============================================================================================================

# just use ats for now because it's smaller
ats_invoice_dates_df = ats_invoice_df.copy()

ats_invoice_dates_df = ats_invoice_dates_df[['id', 'date']]
ats_invoice_dates_df.to_csv(output_dir / 'ats_invoice_dates_df.csv', index=False)

# Merge dates from ats_invoice_dates_df to datetime_parsed_ats_invoice_line_item_df
# Matching id from ats_invoice_dates_df to invoice_id from datetime_parsed_ats_invoice_line_item_df
datetime_parsed_ats_invoice_line_item_df = datetime_parsed_ats_invoice_line_item_df.merge(
    ats_invoice_dates_df[['id', 'date']], 
    left_on='invoice_id', 
    right_on='id', 
    how='left',
    suffixes=('', '_from_invoice')
)

# Rename the merged date column to 'date_imputed'
datetime_parsed_ats_invoice_line_item_df = datetime_parsed_ats_invoice_line_item_df.rename(columns={'date': 'date_from_invoice'})

# Drop the extra 'id' column from the merge
datetime_parsed_ats_invoice_line_item_df = datetime_parsed_ats_invoice_line_item_df.drop(columns=['id_from_invoice'])
datetime_parsed_ats_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_ats_invoice_line_item_df.csv', index=False)

# ============================================================================================================
# IMPUTE A TRANSACTION DATE BASED ON WHAT REAL INFO WE HAVE, AND AVERAGING BASED ON THAT

# Create new column 'date_diff'
datetime_parsed_ats_invoice_line_item_df['date_diff'] = pd.NA

# Drop the existing date_diff column and start fresh
datetime_parsed_ats_invoice_line_item_df = datetime_parsed_ats_invoice_line_item_df.drop(columns=['date_diff'], errors='ignore')

# Calculate date_diff only where transaction_date is not null
mask = datetime_parsed_ats_invoice_line_item_df['transaction_date'].notnull()

# Create the date_diff column as timedelta type
date_diff_timedelta = pd.to_timedelta(
    pd.to_datetime(datetime_parsed_ats_invoice_line_item_df['date_from_invoice']) - 
    pd.to_datetime(datetime_parsed_ats_invoice_line_item_df['transaction_date'])
).where(mask)

# Format as "XX days" without hours
datetime_parsed_ats_invoice_line_item_df['date_diff'] = date_diff_timedelta.dt.days.apply(
    lambda x: f"{int(x)} days" if pd.notna(x) else None
)

# Calculate average on the timedelta values (before formatting)
average_date_diff = date_diff_timedelta.dropna().mean()
# approx. 15.3 days

# Round to nearest day
average_date_diff_days = round(average_date_diff.days + average_date_diff.seconds / 86400)

print(f"Average date_diff: {average_date_diff}")
print(f"Average in days: {average_date_diff.days if pd.notna(average_date_diff) else 'N/A'}")

datetime_parsed_ats_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_ats_invoice_line_item_df.csv', index=False)

# Create date_imputed column - start with transaction_date
datetime_parsed_ats_invoice_line_item_df['date_imputed'] = pd.to_datetime(
    datetime_parsed_ats_invoice_line_item_df['transaction_date']
)

# Where date_imputed is null, use date_from_invoice minus average_date_diff_days
null_mask = datetime_parsed_ats_invoice_line_item_df['date_imputed'].isnull()
datetime_parsed_ats_invoice_line_item_df.loc[null_mask, 'date_imputed'] = (
    pd.to_datetime(datetime_parsed_ats_invoice_line_item_df.loc[null_mask, 'date_from_invoice']) - 
    pd.Timedelta(days=average_date_diff_days)
)

# date imputation complete
datetime_parsed_ats_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_ats_invoice_line_item_df.csv', index=False)

# ============================================================================================================
# REPEAT PROCESS FOR INVOICE_LINE_ITEM AND INVOICE
# ============================================================================================================
# just use invoice for now
invoice_dates_df = invoice_df.copy()

invoice_dates_df = invoice_dates_df[['id', 'date']]
invoice_dates_df.to_csv(output_dir / 'invoice_dates_df.csv', index=False)

# Merge dates from invoice_dates_df to datetime_parsed_invoice_line_item_df
# Matching id from invoice_dates_df to invoice_id from datetime_parsed_invoice_line_item_df
datetime_parsed_invoice_line_item_df = datetime_parsed_invoice_line_item_df.merge(
    invoice_dates_df[['id', 'date']], 
    left_on='invoice_id', 
    right_on='id', 
    how='left',
    suffixes=('', '_from_invoice')
)

# Rename the merged date column to 'date_from_invoice'
datetime_parsed_invoice_line_item_df = datetime_parsed_invoice_line_item_df.rename(columns={'date': 'date_from_invoice'})

# Drop the extra 'id' column from the merge
datetime_parsed_invoice_line_item_df = datetime_parsed_invoice_line_item_df.drop(columns=['id_from_invoice'])
datetime_parsed_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_invoice_line_item_df.csv', index=False)

# ============================================================================================================
# IMPUTE A TRANSACTION DATE BASED ON WHAT REAL INFO WE HAVE, AND AVERAGING BASED ON THAT

# Create new column 'date_diff'
datetime_parsed_invoice_line_item_df['date_diff'] = pd.NA

# Drop the existing date_diff column and start fresh
datetime_parsed_invoice_line_item_df = datetime_parsed_invoice_line_item_df.drop(columns=['date_diff'], errors='ignore')

# Calculate date_diff only where transaction_date is not null
mask = datetime_parsed_invoice_line_item_df['transaction_date'].notnull()

# Convert dates with error handling for invalid dates
date_from_invoice_converted = pd.to_datetime(
    datetime_parsed_invoice_line_item_df['date_from_invoice'], 
    errors='coerce'  # This will convert invalid dates to NaT
)
transaction_date_converted = pd.to_datetime(
    datetime_parsed_invoice_line_item_df['transaction_date'], 
    errors='coerce'
)

# Create the date_diff column as timedelta type
# Only calculate where BOTH dates are valid AND transaction_date is not null
valid_mask = mask & date_from_invoice_converted.notna() & transaction_date_converted.notna()

date_diff_timedelta = pd.to_timedelta(
    date_from_invoice_converted - transaction_date_converted
).where(valid_mask)

# Format as "XX days" without hours
datetime_parsed_invoice_line_item_df['date_diff'] = date_diff_timedelta.dt.days.apply(
    lambda x: f"{int(x)} days" if pd.notna(x) else None
)

# Calculate average on the timedelta values (before formatting)
average_date_diff = date_diff_timedelta.dropna().mean()

# Round to nearest day
average_date_diff_days = round(average_date_diff.days + average_date_diff.seconds / 86400)

print(f"Average date_diff: {average_date_diff}")
print(f"Average in days (rounded): {average_date_diff_days} days")
# approx. 22 days

# Check for invalid dates
print(f"\nInvalid dates found in date_from_invoice: {date_from_invoice_converted.isna().sum()}")
print(f"Invalid dates found in transaction_date: {transaction_date_converted.isna().sum()}")

datetime_parsed_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_invoice_line_item_df.csv', index=False)

# Create date_imputed column - start with transaction_date (with error handling)
datetime_parsed_invoice_line_item_df['date_imputed'] = pd.to_datetime(
    datetime_parsed_invoice_line_item_df['transaction_date'],
    errors='coerce'  # Convert invalid dates to NaT
)

# Where date_imputed is null, use date_from_invoice minus average_date_diff_days
null_mask = datetime_parsed_invoice_line_item_df['date_imputed'].isnull()

# Also convert date_from_invoice with error handling
date_from_invoice_safe = pd.to_datetime(
    datetime_parsed_invoice_line_item_df.loc[null_mask, 'date_from_invoice'],
    errors='coerce'
)

datetime_parsed_invoice_line_item_df.loc[null_mask, 'date_imputed'] = (
    date_from_invoice_safe - pd.Timedelta(days=average_date_diff_days)
)

# date imputation complete
datetime_parsed_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_invoice_line_item_df.csv', index=False)

# Verify the imputation
print("\nDate imputation summary:")
print(f"Total rows: {len(datetime_parsed_invoice_line_item_df)}")
print(f"Rows with valid transaction_date: {datetime_parsed_invoice_line_item_df['transaction_date'].notna().sum()}")
print(f"Rows imputed from date_from_invoice: {null_mask.sum()}")
print(f"Remaining nulls in date_imputed: {datetime_parsed_invoice_line_item_df['date_imputed'].isna().sum()}")


# For ATS dataframe
datetime_parsed_ats_invoice_line_item_df['invoice_period'] = datetime_parsed_ats_invoice_line_item_df['date_from_invoice'].apply(get_next_20th)

# For Invoice dataframe
datetime_parsed_invoice_line_item_df['invoice_period'] = datetime_parsed_invoice_line_item_df['date_from_invoice'].apply(get_next_20th)

# Save the updated dataframes
datetime_parsed_ats_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_ats_invoice_line_item_df.csv', index=False)
datetime_parsed_invoice_line_item_df.to_csv(output_dir / 'datetime_parsed_invoice_line_item_df.csv', index=False)

print("Invoice period column added successfully!")


# ============================================================================================================
# CREATE SUMMARY DATAFRAMES FOR INVOICE COUNTS PER MONTH
# ============================================================================================================

# For ATS dataframe
print("\n" + "="*80)
print("INVOICE COUNT SUMMARY - ATS")
print("="*80)

# Convert to datetime for grouping
ats_invoice_period_dt = pd.to_datetime(datetime_parsed_ats_invoice_line_item_df['invoice_period'])
ats_date_from_invoice_dt = pd.to_datetime(datetime_parsed_ats_invoice_line_item_df['date_from_invoice'])

# Count by invoice_period (month)
ats_invoice_period_counts = datetime_parsed_ats_invoice_line_item_df.groupby(
    ats_invoice_period_dt.dt.to_period('M')
).size().reset_index(name='count')
ats_invoice_period_counts.columns = ['month', 'invoice_count']
ats_invoice_period_counts['month'] = ats_invoice_period_counts['month'].astype(str)

# Count by date_from_invoice (month)
ats_date_from_invoice_counts = datetime_parsed_ats_invoice_line_item_df.groupby(
    ats_date_from_invoice_dt.dt.to_period('M')
).size().reset_index(name='count')
ats_date_from_invoice_counts.columns = ['month', 'invoice_count']
ats_date_from_invoice_counts['month'] = ats_date_from_invoice_counts['month'].astype(str)

# Save ATS summary dataframes
ats_invoice_period_counts.to_csv(output_dir / 'ats_invoice_period_monthly_counts.csv', index=False)
ats_date_from_invoice_counts.to_csv(output_dir / 'ats_date_from_invoice_monthly_counts.csv', index=False)

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
invoice_period_dt = pd.to_datetime(datetime_parsed_invoice_line_item_df['invoice_period'])
date_from_invoice_dt = pd.to_datetime(datetime_parsed_invoice_line_item_df['date_from_invoice'])

# Count by invoice_period (month)
invoice_period_counts = datetime_parsed_invoice_line_item_df.groupby(
    invoice_period_dt.dt.to_period('M')
).size().reset_index(name='count')
invoice_period_counts.columns = ['month', 'invoice_count']
invoice_period_counts['month'] = invoice_period_counts['month'].astype(str)

# Count by date_from_invoice (month)
date_from_invoice_counts = datetime_parsed_invoice_line_item_df.groupby(
    date_from_invoice_dt.dt.to_period('M')
).size().reset_index(name='count')
date_from_invoice_counts.columns = ['month', 'invoice_count']
date_from_invoice_counts['month'] = date_from_invoice_counts['month'].astype(str)

# Save Invoice summary dataframes
invoice_period_counts.to_csv(output_dir / 'invoice_period_monthly_counts.csv', index=False)
date_from_invoice_counts.to_csv(output_dir / 'date_from_invoice_monthly_counts.csv', index=False)

print("\nINVOICE - Invoices per month (by invoice_period):")
print(invoice_period_counts.to_string(index=False))
print(f"\nTotal: {invoice_period_counts['invoice_count'].sum()}")

print("\nINVOICE - Invoices per month (by date_from_invoice):")
print(date_from_invoice_counts.to_string(index=False))
print(f"\nTotal: {date_from_invoice_counts['invoice_count'].sum()}")

print("\n" + "="*80)
print("Summary CSV files saved:")
print("  - ats_invoice_period_monthly_counts.csv")
print("  - ats_date_from_invoice_monthly_counts.csv")
print("  - invoice_period_monthly_counts.csv")
print("  - date_from_invoice_monthly_counts.csv")
print("="*80)