import pandas as pd
import numpy as np

# ================================================================
# Load the grouped invoice datasets (already has invoice_period)
# ================================================================
ats_grouped = pd.read_csv('ats_grouped_by_invoice_transformed.csv')
invoice_grouped = pd.read_csv('invoice_grouped_by_invoice_transformed.csv')

print(f"Loaded ATS grouped data: {len(ats_grouped):,} invoices")
print(f"Loaded Invoice grouped data: {len(invoice_grouped):,} invoices")

print(f"\nATS columns: {ats_grouped.columns.tolist()}")
print(f"Invoice columns: {invoice_grouped.columns.tolist()}")

# ================================================================
# Calculate discount amounts per invoice
# ================================================================

# For ATS
ats_grouped['discount_amount'] = (
    ats_grouped['total_undiscounted_price'] - ats_grouped['total_discounted_price']
).clip(lower=0).round(2)

# For Invoice
invoice_grouped['discount_amount'] = (
    invoice_grouped['total_undiscounted_price'] - invoice_grouped['total_discounted_price']
).clip(lower=0).round(2)

# ================================================================
# Calculate statistics and save to results dataframe
# ================================================================

# ATS statistics
total_discount_ats = ats_grouped['discount_amount'].sum()
avg_discount_ats = ats_grouped['discount_amount'].mean()
median_discount_ats = ats_grouped['discount_amount'].median()
invoices_with_discount_ats = (ats_grouped['discount_amount'] > 0).sum()
pct_with_discount_ats = (invoices_with_discount_ats / len(ats_grouped) * 100)

# Invoice statistics
total_discount_invoice = invoice_grouped['discount_amount'].sum()
avg_discount_invoice = invoice_grouped['discount_amount'].mean()
median_discount_invoice = invoice_grouped['discount_amount'].median()
invoices_with_discount_invoice = (invoice_grouped['discount_amount'] > 0).sum()
pct_with_discount_invoice = (invoices_with_discount_invoice / len(invoice_grouped) * 100)

# Combined statistics
total_discount_combined = total_discount_ats + total_discount_invoice
total_invoices_combined = len(ats_grouped) + len(invoice_grouped)
invoices_with_discount_combined = invoices_with_discount_ats + invoices_with_discount_invoice
pct_with_discount_combined = (invoices_with_discount_combined / total_invoices_combined * 100)

# ================================================================
# Create results dataframe
# ================================================================
results = pd.DataFrame({
    'customer_type': ['ATS', 'Invoice', 'Combined'],
    'total_invoices': [len(ats_grouped), len(invoice_grouped), total_invoices_combined],
    'invoices_with_discount': [invoices_with_discount_ats, invoices_with_discount_invoice, invoices_with_discount_combined],
    'pct_with_discount': [pct_with_discount_ats, pct_with_discount_invoice, pct_with_discount_combined],
    'total_discount_amount': [total_discount_ats, total_discount_invoice, total_discount_combined],
    'avg_discount_per_invoice': [avg_discount_ats, avg_discount_invoice, 
                                  (total_discount_combined / total_invoices_combined)],
    'median_discount': [median_discount_ats, median_discount_invoice, 
                        pd.concat([ats_grouped['discount_amount'], invoice_grouped['discount_amount']]).median()]
})

# Round numeric columns for display
results['total_discount_amount'] = results['total_discount_amount'].round(2)
results['avg_discount_per_invoice'] = results['avg_discount_per_invoice'].round(2)
results['median_discount'] = results['median_discount'].round(2)
results['pct_with_discount'] = results['pct_with_discount'].round(2)

# ================================================================
# Display results
# ================================================================
print("\n" + "="*70)
print("DISCOUNT ANALYSIS RESULTS")
print("="*70)
print(results.to_string(index=False))

# ================================================================
# Save all outputs
# ================================================================
ats_grouped.to_csv('ats_grouped_transformed_with_discounts.csv', index=False)
invoice_grouped.to_csv('invoice_grouped_transformed_with_discounts.csv', index=False)
results.to_csv('discount_analysis_results_transformed.csv', index=False)

print("\n" + "="*70)
print("SAVED FILES:")
print("="*70)
print("- ats_grouped_transformed_with_discounts.csv (includes invoice_period)")
print("- invoice_grouped_transformed_with_discounts.csv (includes invoice_period)")
print("- discount_analysis_results_transformed.csv")

print("\nSample of ATS grouped data with invoice_period:")
print(ats_grouped[['invoice_id', 'invoice_period', 'discount_amount']].head())