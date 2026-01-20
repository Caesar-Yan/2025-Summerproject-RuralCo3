'''
Docstring for 08.1_group_by_invoice_transformed

this script groups line_items by invoice_id and keeps only the invoice period and total prices data. 
also calculates a discount_amount on the difference of undiscounted_price and discounted_price
this is essentially just keeping all the neccessary information for estimating revenue

columns included:
- invoice_id
- total_discounted_price
- total undiscounted_price
- discount_amount
- invoice_period

inputs:
- datetime_parsed_ats_invoice_line_item_df_transformed.csv
- datetime_parsed_invoice_line_item_df_transformed.csv

outputs:
- ats_grouped_by_invoice_transformed.csv
- invoice_grouped_by_invoice_transformed.csv

'''



import pandas as pd
import numpy as np
import pickle

# ================================================================
# Load the two imputed line-item datasets
# ================================================================
ats_path = "datetime_parsed_ats_invoice_line_item_df_transformed.csv"
invoice_path = "datetime_parsed_invoice_line_item_df_transformed.csv"

ats = pd.read_csv(ats_path)
invoice = pd.read_csv(invoice_path)

print(f"Loaded ATS dataset: {len(ats):,} line-items")
print(f"Loaded Invoice dataset: {len(invoice):,} line-items")

# ================================================================
# Group by invoice_id and sum prices, also get invoice_period
# ================================================================

# For ATS
ats_grouped = ats.groupby('invoice_id').agg({
    'discounted_price': 'sum',
    'undiscounted_price': 'sum',
    'invoice_period': 'first'  # Take the first invoice_period (should be same for all line items in an invoice)
}).reset_index()

ats_grouped = ats_grouped.rename(columns={
    'discounted_price': 'total_discounted_price',
    'undiscounted_price': 'total_undiscounted_price'
})

print(f"\nATS grouped by invoice_id: {len(ats_grouped):,} unique invoices")

# For Invoice
invoice_grouped = invoice.groupby('invoice_id').agg({
    'discounted_price': 'sum',
    'undiscounted_price': 'sum',
    'invoice_period': 'first'  # Take the first invoice_period (should be same for all line items in an invoice)
}).reset_index()

invoice_grouped = invoice_grouped.rename(columns={
    'discounted_price': 'total_discounted_price',
    'undiscounted_price': 'total_undiscounted_price'
})

print(f"Invoice grouped by invoice_id: {len(invoice_grouped):,} unique invoices")

# Display samples
print("\nATS Sample:")
print(ats_grouped.head())

print("\nInvoice Sample:")
print(invoice_grouped.head())

# Optional: Save the grouped data
ats_grouped.to_csv('ats_grouped_by_invoice_transformed.csv', index=False)
invoice_grouped.to_csv('invoice_grouped_by_invoice_transformed.csv', index=False)
print("\nSaved grouped_transformed datasets to CSV files")