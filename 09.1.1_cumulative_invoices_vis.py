"""
Script: 09.1.1_cumulative_invoices_vis.py

Purpose:
    Create visualizations of invoice totals by period using outputs from 09.1 and 14.2.
    
    Two visualizations:
    1. Month by month chart of undiscounted vs discounted totals
    2. Cumulative line chart of undiscounted vs discounted amounts

Inputs:
    - ats_grouped_transformed_with_discounts.csv (from 09.1)
    - invoice_grouped_transformed_with_discounts.csv (from 09.1)
    - 14.2_updated_invoice_line_items_with_discounts.csv (from 14.2)

Outputs:
    - monthly_totals_chart.png
    - cumulative_totals_chart.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
data_cleaning_dir = base_dir / "data_cleaning"
merchant_dir = base_dir / "merchant"
output_dir = base_dir / "visualizations"
output_dir.mkdir(exist_ok=True)

print("="*70)
print("LOADING DATA")
print("="*70)

# Load the grouped invoice datasets from 09.1
ats_grouped = pd.read_csv(data_cleaning_dir / 'ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv(data_cleaning_dir / 'invoice_grouped_transformed_with_discounts.csv')

# Load the updated invoice line items from 14.2
updated_invoice_items = pd.read_csv(merchant_dir / '14.2_updated_invoice_line_items_with_discounts.csv')

print(f"Loaded ATS grouped data: {len(ats_grouped):,} invoices")
print(f"Loaded Invoice grouped data: {len(invoice_grouped):,} invoices")
print(f"Loaded updated invoice line items: {len(updated_invoice_items):,} line items")

# ================================================================
# PROCESS INVOICE_PERIOD AND AGGREGATE BY MONTH
# ================================================================
print("\n" + "="*70)
print("PROCESSING INVOICE PERIODS")
print("="*70)

def process_invoice_period(df, source_name):
    """Process invoice_period column and aggregate by month"""
    df = df.copy()
    
    # Convert invoice_period to datetime
    df['invoice_period'] = pd.to_datetime(df['invoice_period'], errors='coerce')
    
    # Extract year-month for grouping
    df['year_month'] = df['invoice_period'].dt.to_period('M')
    
    # Group by year_month and sum the totals
    monthly_summary = df.groupby('year_month').agg({
        'total_undiscounted_price': 'sum',
        'total_discounted_price': 'sum',
        'discount_amount': 'sum'
    }).reset_index()
    
    monthly_summary['source'] = source_name
    monthly_summary['invoice_period'] = monthly_summary['year_month'].dt.to_timestamp()
    
    print(f"{source_name} date range: {df['invoice_period'].min()} to {df['invoice_period'].max()}")
    print(f"{source_name} monthly periods: {len(monthly_summary)}")
    
    return monthly_summary

# Process both datasets
ats_monthly = process_invoice_period(ats_grouped, 'ATS')
invoice_monthly = process_invoice_period(invoice_grouped, 'Invoice')

# Combine the datasets
combined_monthly = pd.concat([ats_monthly, invoice_monthly], ignore_index=True)

# Aggregate across both sources by month
final_monthly = combined_monthly.groupby('invoice_period').agg({
    'total_undiscounted_price': 'sum',
    'total_discounted_price': 'sum',
    'discount_amount': 'sum'
}).reset_index().sort_values('invoice_period')

print(f"\nCombined monthly periods: {len(final_monthly)}")
print(f"Total date range: {final_monthly['invoice_period'].min()} to {final_monthly['invoice_period'].max()}")

# ================================================================
# CREATE VISUALIZATION 1: MONTHLY TOTALS BAR CHART
# ================================================================
print("\n" + "="*70)
print("CREATING MONTHLY TOTALS VISUALIZATION")
print("="*70)

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 8))

# Prepare data for plotting
x = final_monthly['invoice_period']
width = 12   # Width of individual bars in days
offset = 6  # Small offset to group bars together within each month

# Create bars with grouped spacing (bars close together within each month)
bars1 = ax.bar(x - pd.Timedelta(days=offset), final_monthly['total_undiscounted_price'], 
               width=width, label='Undiscounted Total', alpha=0.8, color='#2E8B57')
bars2 = ax.bar(x + pd.Timedelta(days=offset), final_monthly['total_discounted_price'], 
               width=width, label='Discounted Total', alpha=0.8, color='#4682B4')

# Formatting
ax.set_xlabel('Invoice Period', fontsize=16, fontweight='bold')
ax.set_ylabel('Total Amount ($)', fontsize=16, fontweight='bold')
ax.set_title('Monthly Invoice Totals: Undiscounted vs Discounted', fontsize=21, fontweight='bold', pad=20)
ax.legend(fontsize=14)

# Format y-axis to show values in millions
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
plt.xticks(rotation=45)

# Add grid
ax.grid(True, alpha=0.3)

# Tight layout
plt.tight_layout()

# Save the chart
monthly_chart_path = output_dir / '09.1.1_monthly_totals_chart.png'
plt.savefig(monthly_chart_path, dpi=300, bbox_inches='tight')
print(f"✓ Monthly totals chart saved: {monthly_chart_path}")
plt.close()  # Close the figure to free memory

# ================================================================
# CREATE VISUALIZATION 2: CUMULATIVE LINE CHART
# ================================================================
print("\n" + "="*70)
print("CREATING CUMULATIVE TOTALS VISUALIZATION")
print("="*70)

# Calculate cumulative totals
final_monthly = final_monthly.sort_values('invoice_period')
final_monthly['cumulative_undiscounted'] = final_monthly['total_undiscounted_price'].cumsum()
final_monthly['cumulative_discounted'] = final_monthly['total_discounted_price'].cumsum()
final_monthly['cumulative_discount_amount'] = final_monthly['discount_amount'].cumsum()

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 8))

# Plot cumulative lines
ax.plot(final_monthly['invoice_period'], final_monthly['cumulative_undiscounted'], 
        marker='o', linewidth=3, markersize=6, label='Cumulative Undiscounted', color='#2E8B57')
ax.plot(final_monthly['invoice_period'], final_monthly['cumulative_discounted'], 
        marker='s', linewidth=3, markersize=6, label='Cumulative Discounted', color='#4682B4')

# Fill area between lines to show discount amount
ax.fill_between(final_monthly['invoice_period'], 
                final_monthly['cumulative_discounted'], 
                final_monthly['cumulative_undiscounted'],
                alpha=0.3, color='#FFD700', label='Cumulative Discount Amount')

# Formatting
ax.set_xlabel('Invoice Period', fontsize=16, fontweight='bold')
ax.set_ylabel('Cumulative Amount ($)', fontsize=16, fontweight='bold')
ax.set_title('Cumulative Invoice Totals Over Time', fontsize=21, fontweight='bold', pad=20)
ax.legend(fontsize=14)

# Format y-axis to show values in millions
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
plt.xticks(rotation=45)

# Add grid
ax.grid(True, alpha=0.3)

# Tight layout
plt.tight_layout()

# Save the chart
cumulative_chart_path = output_dir / '09.1.1_cumulative_totals_chart.png'
plt.savefig(cumulative_chart_path, dpi=300, bbox_inches='tight')
print(f"✓ Cumulative totals chart saved: {cumulative_chart_path}")
plt.close()  # Close the figure to free memory

# ================================================================
# CREATE VISUALIZATION 3: CUMULATIVE DISCOUNT AMOUNT CHART
# ================================================================
print("\n" + "="*70)
print("CREATING CUMULATIVE DISCOUNT AMOUNT VISUALIZATION")
print("="*70)

# Create figure and axis for discount chart
fig, ax = plt.subplots(figsize=(15, 8))

# Plot cumulative discount amount
ax.plot(final_monthly['invoice_period'], final_monthly['cumulative_discount_amount'], 
        marker='D', linewidth=4, markersize=8, label='Cumulative Discount Amount', color='#DC143C')

# Fill area under the curve
ax.fill_between(final_monthly['invoice_period'], 
                0,
                final_monthly['cumulative_discount_amount'],
                alpha=0.3, color='#DC143C')

# Formatting
ax.set_xlabel('Invoice Period', fontsize=16, fontweight='bold')
ax.set_ylabel('Cumulative Discount Amount ($)', fontsize=16, fontweight='bold')
ax.set_title('Cumulative Discount Amount Over Time', fontsize=21, fontweight='bold', pad=20)
ax.legend(fontsize=14)

# Format y-axis to show values in millions
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
plt.xticks(rotation=45)

# Add grid
ax.grid(True, alpha=0.3)

# Add text annotation for total discount
total_discount_millions = final_monthly['cumulative_discount_amount'].iloc[-1] / 1e6
ax.annotate(f'Total: ${total_discount_millions:.1f}M', 
            xy=(final_monthly['invoice_period'].iloc[-1], final_monthly['cumulative_discount_amount'].iloc[-1]),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            fontsize=11, fontweight='bold')

# Tight layout
plt.tight_layout()

# Save the chart
discount_chart_path = output_dir / '09.1.1_cumulative_discount_chart.png'
plt.savefig(discount_chart_path, dpi=300, bbox_inches='tight')
print(f"✓ Cumulative discount chart saved: {discount_chart_path}")
plt.close()  # Close the figure to free memory

# ================================================================
# CREATE VISUALIZATION 4: MONTHLY DISCOUNT AMOUNT LINE CHART
# ================================================================
print("\n" + "="*70)
print("CREATING MONTHLY DISCOUNT AMOUNT VISUALIZATION")
print("="*70)

# Create figure and axis for monthly discount chart
fig, ax = plt.subplots(figsize=(15, 8))

# Plot monthly discount amounts as a line chart
ax.plot(final_monthly['invoice_period'], final_monthly['discount_amount'], 
        marker='o', linewidth=3, markersize=8, label='Monthly Discount Amount', color='#FF6B35')

# Fill area under the curve
ax.fill_between(final_monthly['invoice_period'], 
                0,
                final_monthly['discount_amount'],
                alpha=0.3, color='#FF6B35')

# Formatting
ax.set_xlabel('Invoice Period', fontsize=16, fontweight='bold')
ax.set_ylabel('Monthly Discount Amount ($)', fontsize=16, fontweight='bold')
ax.set_title('Monthly Discount Amount Over Time', fontsize=21, fontweight='bold', pad=20)
ax.legend(fontsize=14)

# Format y-axis to show values in thousands or millions as appropriate
max_discount = final_monthly['discount_amount'].max()
if max_discount > 1e6:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
else:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
plt.xticks(rotation=45)

# Add grid
ax.grid(True, alpha=0.3)

# Tight layout
plt.tight_layout()

# Save the chart
monthly_discount_chart_path = output_dir / '09.1.1_monthly_discount_line_chart.png'
plt.savefig(monthly_discount_chart_path, dpi=300, bbox_inches='tight')
print(f"✓ Monthly discount line chart saved: {monthly_discount_chart_path}")
plt.close()  # Close the figure to free memory

# ================================================================
# CREATE VISUALIZATION 5: UNDISCOUNTED/DISCOUNTED RATIO CHART
# ================================================================
print("\n" + "="*70)
print("CREATING UNDISCOUNTED/DISCOUNTED RATIO VISUALIZATION")
print("="*70)

# Calculate the ratio of undiscounted to discounted amounts
final_monthly['price_ratio'] = final_monthly['total_undiscounted_price'] / final_monthly['total_discounted_price']

# Create figure and axis for ratio chart
fig, ax = plt.subplots(figsize=(15, 8))

# Plot ratio as a line chart
ax.plot(final_monthly['invoice_period'], final_monthly['price_ratio'], 
        marker='s', linewidth=3, markersize=8, label='Undiscounted/Discounted Ratio', color='#8A2BE2')

# Fill area between ratio line and 1.0 baseline
ax.fill_between(final_monthly['invoice_period'], 
                1.0,
                final_monthly['price_ratio'],
                alpha=0.3, color='#8A2BE2')

# Add horizontal line at y=1.0 for reference
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No Discount (1.0)')

# Formatting
ax.set_xlabel('Invoice Period', fontsize=16, fontweight='bold')
ax.set_ylabel('Price Ratio (Undiscounted / Discounted)', fontsize=16, fontweight='bold')
ax.set_title('Monthly Price Ratio: Undiscounted vs Discounted', fontsize=21, fontweight='bold', pad=20)
ax.legend(fontsize=14)

# Format y-axis to show ratio values
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}x'))

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Every 3 months
plt.xticks(rotation=45)

# Add grid
ax.grid(True, alpha=0.3)

# Set y-axis to start from a reasonable minimum to show variation
y_min = max(1.0, final_monthly['price_ratio'].min() * 0.95)
ax.set_ylim(bottom=y_min)

# Tight layout
plt.tight_layout()

# Save the chart
ratio_chart_path = output_dir / '09.1.1_price_ratio_chart.png'
plt.savefig(ratio_chart_path, dpi=300, bbox_inches='tight')
print(f"✓ Price ratio chart saved: {ratio_chart_path}")
plt.close()  # Close the figure to free memory

# ================================================================
# SUMMARY STATISTICS
# ================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

total_undiscounted = final_monthly['total_undiscounted_price'].sum()
total_discounted = final_monthly['total_discounted_price'].sum()
total_discount_amount = final_monthly['discount_amount'].sum()
discount_percentage = (total_discount_amount / total_undiscounted) * 100

print(f"📊 OVERALL TOTALS:")
print(f"  • Total Undiscounted Amount: ${total_undiscounted:,.2f}")
print(f"  • Total Discounted Amount: ${total_discounted:,.2f}")
print(f"  • Total Discount Amount: ${total_discount_amount:,.2f}")
print(f"  • Average Discount Rate: {discount_percentage:.2f}%")

print(f"\n📅 PERIOD COVERAGE:")
print(f"  • Start Date: {final_monthly['invoice_period'].min().strftime('%B %Y')}")
print(f"  • End Date: {final_monthly['invoice_period'].max().strftime('%B %Y')}")
print(f"  • Number of Months: {len(final_monthly)}")

print(f"\n📈 MONTHLY AVERAGES:")
print(f"  • Average Monthly Undiscounted: ${final_monthly['total_undiscounted_price'].mean():,.2f}")
print(f"  • Average Monthly Discounted: ${final_monthly['total_discounted_price'].mean():,.2f}")
print(f"  • Average Monthly Discount: ${final_monthly['discount_amount'].mean():,.2f}")

print(f"\n📁 OUTPUT FILES:")
print(f"  • {monthly_chart_path.name}")
print(f"  • {cumulative_chart_path.name}")
print(f"  • {discount_chart_path.name}")
print(f"  • {monthly_discount_chart_path.name}")
print(f"  • {ratio_chart_path.name}")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
