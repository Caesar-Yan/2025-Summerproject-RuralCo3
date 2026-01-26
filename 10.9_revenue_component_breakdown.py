'''
Docstring for 10.9_Revenue_component_breakdown_comparison

This script creates visualizations comparing the revenue breakdown between scenarios:
- NO DISCOUNT: Stacked bar chart showing Interest Revenue + Retained Discounts
- WITH DISCOUNT: Line overlay showing total revenue for comparison

Uses Monte Carlo simulation results from 10.8 to show uncertainty bands.

Inputs:
- 10.8_simulation_results_distribution.csv (Monte Carlo results)
- 10.7_with_discount_details.csv (single run detailed results)
- 10.7_no_discount_details.csv (single run detailed results)

Outputs:
- 10.9_revenue_component_breakdown_monthly.png (monthly breakdown)
- 10.9_revenue_component_breakdown_cumulative.png (cumulative breakdown)
- 10.9_revenue_component_annual_summary.png (annual summary bars)
- 10.9_component_breakdown_data.csv (underlying data)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os

# ================================================================
# CONFIGURATION
# ================================================================
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
visualisations_dir = base_dir / "visualisations"
OUTPUT_DIR = visualisations_dir

FY2025_START = pd.Timestamp("2024-07-01")
FY2025_END = pd.Timestamp("2025-06-30")
TARGET_REVENUE = 1_043_000

print("\n" + "="*70)
print("REVENUE COMPONENT BREAKDOWN COMPARISON")
print("="*70)

# ================================================================
# Load detailed simulation results from 10.7
# ================================================================
print("\n" + "="*70)
print("LOADING DETAILED SIMULATION RESULTS")
print("="*70)

try:
    with_discount_df = pd.read_csv(visualisations_dir / '10.7_with_discount_details.csv')
    no_discount_df = pd.read_csv(visualisations_dir / '10.7_no_discount_details.csv')
    
    print(f"✓ Loaded with_discount: {len(with_discount_df):,} invoices")
    print(f"✓ Loaded no_discount: {len(no_discount_df):,} invoices")
    
except FileNotFoundError as e:
    print(f"ERROR: Could not find detailed results from 10.7")
    print("Please run 10.7_Calculation_with_calibrated_baseline.py first")
    exit(1)

# Parse dates
with_discount_df['payment_date'] = pd.to_datetime(with_discount_df['payment_date'])
no_discount_df['payment_date'] = pd.to_datetime(no_discount_df['payment_date'])

# ================================================================
# Calculate monthly aggregations
# ================================================================
print("\n" + "="*70)
print("CALCULATING MONTHLY AGGREGATIONS")
print("="*70)

# WITH DISCOUNT - monthly
with_discount_monthly = with_discount_df.groupby(
    with_discount_df['payment_date'].dt.to_period('M')
).agg({
    'interest_charged': 'sum',
    'retained_discounts': 'sum',
    'credit_card_revenue': 'sum'
}).reset_index()
with_discount_monthly['payment_date'] = with_discount_monthly['payment_date'].dt.to_timestamp()
with_discount_monthly = with_discount_monthly.sort_values('payment_date')

# NO DISCOUNT - monthly
no_discount_monthly = no_discount_df.groupby(
    no_discount_df['payment_date'].dt.to_period('M')
).agg({
    'interest_charged': 'sum',
    'retained_discounts': 'sum',
    'credit_card_revenue': 'sum'
}).reset_index()
no_discount_monthly['payment_date'] = no_discount_monthly['payment_date'].dt.to_timestamp()
no_discount_monthly = no_discount_monthly.sort_values('payment_date')

print(f"✓ Calculated monthly data")
print(f"  Months covered: {len(with_discount_monthly)}")

# ================================================================
# Calculate cumulative values
# ================================================================
with_discount_monthly['cumulative_interest'] = with_discount_monthly['interest_charged'].cumsum()
with_discount_monthly['cumulative_retained'] = with_discount_monthly['retained_discounts'].cumsum()
with_discount_monthly['cumulative_total'] = with_discount_monthly['credit_card_revenue'].cumsum()

no_discount_monthly['cumulative_interest'] = no_discount_monthly['interest_charged'].cumsum()
no_discount_monthly['cumulative_retained'] = no_discount_monthly['retained_discounts'].cumsum()
no_discount_monthly['cumulative_total'] = no_discount_monthly['credit_card_revenue'].cumsum()

# ================================================================
# Save component breakdown data
# ================================================================
print("\n" + "="*70)
print("SAVING COMPONENT BREAKDOWN DATA")
print("="*70)

breakdown_data = pd.DataFrame({
    'payment_month': no_discount_monthly['payment_date'],
    
    # NO DISCOUNT - Monthly
    'no_discount_interest_monthly': no_discount_monthly['interest_charged'],
    'no_discount_retained_monthly': no_discount_monthly['retained_discounts'],
    'no_discount_total_monthly': no_discount_monthly['credit_card_revenue'],
    
    # NO DISCOUNT - Cumulative
    'no_discount_interest_cumulative': no_discount_monthly['cumulative_interest'],
    'no_discount_retained_cumulative': no_discount_monthly['cumulative_retained'],
    'no_discount_total_cumulative': no_discount_monthly['cumulative_total'],
    
    # WITH DISCOUNT - Monthly
    'with_discount_interest_monthly': with_discount_monthly['interest_charged'].values,
    'with_discount_retained_monthly': with_discount_monthly['retained_discounts'].values,
    'with_discount_total_monthly': with_discount_monthly['credit_card_revenue'].values,
    
    # WITH DISCOUNT - Cumulative
    'with_discount_interest_cumulative': with_discount_monthly['cumulative_interest'].values,
    'with_discount_retained_cumulative': with_discount_monthly['cumulative_retained'].values,
    'with_discount_total_cumulative': with_discount_monthly['cumulative_total'].values,
})

breakdown_output = os.path.join(OUTPUT_DIR, '10.9_component_breakdown_data.csv')
breakdown_data.to_csv(breakdown_output, index=False)
print(f"✓ {breakdown_output}")

# ================================================================
# VISUALIZATION 1: Monthly Revenue Component Breakdown
# ================================================================
print("\n" + "="*70)
print("CREATING MONTHLY BREAKDOWN VISUALIZATION")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 8))

# Create stacked bar chart for NO DISCOUNT components
x = no_discount_monthly['payment_date']
width = 20  # Width in days for bars

# Plot bars
bars1 = ax.bar(x, no_discount_monthly['interest_charged'], 
               width=width, label='No Discount - Interest Revenue',
               color='#4472C4', alpha=0.8, edgecolor='black', linewidth=0.5)

bars2 = ax.bar(x, no_discount_monthly['retained_discounts'], 
               width=width, label='No Discount - Retained Discounts',
               bottom=no_discount_monthly['interest_charged'],
               color='#8FAADC', alpha=0.8, edgecolor='black', linewidth=0.5)

# Overlay WITH DISCOUNT as a line
ax.plot(with_discount_monthly['payment_date'], 
        with_discount_monthly['credit_card_revenue'],
        marker='o', linewidth=3, markersize=8, 
        label='With Discount - Total Revenue',
        color='#70AD47', zorder=5)

# Add target reference line (monthly target)
monthly_target = TARGET_REVENUE / 12
ax.axhline(y=monthly_target, color='red', linestyle='--', linewidth=2, 
           label=f'Monthly Target (${monthly_target:,.0f})', alpha=0.6)

ax.set_title(f'FY2025 Monthly Revenue Component Breakdown\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Payment Month', fontsize=14)
ax.set_ylabel('Monthly Revenue ($)', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
monthly_output = os.path.join(OUTPUT_DIR, '10.9_revenue_component_breakdown_monthly.png')
plt.savefig(monthly_output, dpi=300, bbox_inches='tight')
print(f"✓ {monthly_output}")
plt.close()

# ================================================================
# VISUALIZATION 2: Cumulative Revenue Component Breakdown
# ================================================================
print("\n" + "="*70)
print("CREATING CUMULATIVE BREAKDOWN VISUALIZATION")
print("="*70)

fig, ax = plt.subplots(figsize=(16, 8))

# Create stacked area chart for NO DISCOUNT components
ax.fill_between(no_discount_monthly['payment_date'], 
                0, 
                no_discount_monthly['cumulative_interest'],
                alpha=0.6, color='#4472C4', label='No Discount - Interest Revenue',
                edgecolor='black', linewidth=1)

ax.fill_between(no_discount_monthly['payment_date'], 
                no_discount_monthly['cumulative_interest'],
                no_discount_monthly['cumulative_total'],
                alpha=0.6, color='#8FAADC', label='No Discount - Retained Discounts',
                edgecolor='black', linewidth=1)

# Add line for NO DISCOUNT total
ax.plot(no_discount_monthly['payment_date'], 
        no_discount_monthly['cumulative_total'],
        marker='s', linewidth=2.5, markersize=8, 
        label='No Discount - Total',
        color='#4472C4', zorder=5)

# Overlay WITH DISCOUNT as a line
ax.plot(with_discount_monthly['payment_date'], 
        with_discount_monthly['cumulative_total'],
        marker='o', linewidth=3, markersize=8, 
        label='With Discount - Total Revenue',
        color='#70AD47', zorder=5)

# Add target line
ax.axhline(y=TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target: ${TARGET_REVENUE:,.0f}', alpha=0.8)

ax.set_title(f'FY2025 Cumulative Revenue Component Breakdown\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Payment Month', fontsize=14)
ax.set_ylabel('Cumulative Revenue ($)', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
cumulative_output = os.path.join(OUTPUT_DIR, '10.9_revenue_component_breakdown_cumulative.png')
plt.savefig(cumulative_output, dpi=300, bbox_inches='tight')
print(f"✓ {cumulative_output}")
plt.close()

# ================================================================
# VISUALIZATION 3: Annual Summary Bars (Side-by-side components)
# ================================================================
print("\n" + "="*70)
print("CREATING ANNUAL SUMMARY VISUALIZATION")
print("="*70)

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate annual totals for NO DISCOUNT scenario
no_discount_interest = no_discount_df['interest_charged'].sum()
no_discount_retained = no_discount_df['retained_discounts'].sum()
no_discount_total = no_discount_df['credit_card_revenue'].sum()

with_discount_total = with_discount_df['credit_card_revenue'].sum()

# Set up bar positions
x = np.arange(2)  # Two components: Interest and Retained
width = 0.35  # Increased width to reduce whitespace

# Create side-by-side bars for the two components
bars1 = ax.bar(x[0], no_discount_interest, width, 
               label='Interest Revenue',
               color='#4472C4', alpha=0.8, edgecolor='black', linewidth=1.5)

bars2 = ax.bar(x[1], no_discount_retained, width,
               label='Retained Discounts',
               color='#8FAADC', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
ax.text(x[0], no_discount_interest + 10000, f'${no_discount_interest:,.0f}', 
        ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.text(x[1], no_discount_retained + 10000, f'${no_discount_retained:,.0f}', 
        ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add horizontal line for WITH DISCOUNT total (for comparison)
ax.axhline(y=with_discount_total, color='#70AD47', linestyle='-', linewidth=3, 
           label=f'With Discount Total: ${with_discount_total:,.0f}', alpha=0.8, zorder=5)

ax.set_ylabel('Annual Revenue ($)', fontsize=14)
ax.set_title(f'FY2025 No Discount Revenue Components vs With Discount Total\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(['Interest Revenue', 'Retained Discounts'], fontsize=13)
ax.set_xlim(-0.5, 1.5)  # Reduce whitespace on sides
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add annotation showing the sum
ax.text(0.5, no_discount_total + 20000, 
        f'No Discount Total = ${no_discount_total:,.0f}',
        ha='center', fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
annual_output = os.path.join(OUTPUT_DIR, '10.9_revenue_component_annual_summary.png')
plt.savefig(annual_output, dpi=300, bbox_inches='tight')
print(f"✓ {annual_output}")
plt.close()

# ================================================================
# Print Summary Statistics
# ================================================================
print("\n" + "="*70)
print("REVENUE COMPONENT SUMMARY")
print("="*70)

# Calculate totals
no_discount_interest = no_discount_df['interest_charged'].sum()
no_discount_retained = no_discount_df['retained_discounts'].sum()
no_discount_total = no_discount_df['credit_card_revenue'].sum()

with_discount_interest = with_discount_df['interest_charged'].sum()
with_discount_retained = with_discount_df['retained_discounts'].sum()
with_discount_total = with_discount_df['credit_card_revenue'].sum()

print("\nNO DISCOUNT scenario:")
print(f"  Interest Revenue:     ${no_discount_interest:>15,.2f} ({no_discount_interest/no_discount_total*100:>5.1f}%)")
print(f"  Retained Discounts:   ${no_discount_retained:>15,.2f} ({no_discount_retained/no_discount_total*100:>5.1f}%)")
print(f"  Total Revenue:        ${no_discount_total:>15,.2f}")

print("\nWITH DISCOUNT scenario:")
print(f"  Interest Revenue:     ${with_discount_interest:>15,.2f} ({with_discount_interest/with_discount_total*100:>5.1f}%)")
print(f"  Retained Discounts:   ${with_discount_retained:>15,.2f} ({with_discount_retained/with_discount_total*100:>5.1f}%)")
print(f"  Total Revenue:        ${with_discount_total:>15,.2f}")

print("\nRevenue Difference (No Discount - With Discount):")
diff_total = no_discount_total - with_discount_total
diff_interest = no_discount_interest - with_discount_interest
diff_retained = no_discount_retained - with_discount_retained

print(f"  Interest Revenue:     ${diff_interest:>15,.2f}")
print(f"  Retained Discounts:   ${diff_retained:>15,.2f}")
print(f"  Total Revenue:        ${diff_total:>15,.2f}")

print("\nTarget Comparison:")
print(f"  Target:               ${TARGET_REVENUE:>15,.2f}")
print(f"  No Discount Total:    ${no_discount_total:>15,.2f}")
print(f"  Gap:                  ${no_discount_total - TARGET_REVENUE:>15,.2f}")
print(f"  Gap as %:             {(no_discount_total - TARGET_REVENUE) / TARGET_REVENUE * 100:>15.2f}%")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)