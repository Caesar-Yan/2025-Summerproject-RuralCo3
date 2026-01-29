'''
Docstring for 10.3.1_separate_cumulative_revenue

This script creates individual cumulative revenue over time graphs for each calibration method,
showing WITH DISCOUNT vs NO DISCOUNT scenarios with target line but without annotation boxes.

inputs:
- 10.3_FY2025_{METHOD}_detailed.xlsx (for each method)

outputs:
- 10.3.1_cumulative_revenue_MULTIPLIER.png
- 10.3.1_cumulative_revenue_UNIFORM.png

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os

# ================================================================
# Configuration
# ================================================================
ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
TARGET_REVENUE = 1_043_000  # $1.043M target from calibration

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
visualisations_dir = base_dir / "visualisations"
visualisations_dir.mkdir(exist_ok=True)

# New Zealand FY2025 definition
FY2025_START = "01/07/2024"
FY2025_END = "30/06/2025"

# ================================================================
# Find available method files
# ================================================================
print("="*70)
print("LOADING METHOD DATA")
print("="*70)

available_methods = []
method_files = list(visualisations_dir.glob('10.3_FY2025_*_detailed.xlsx'))

for file_path in method_files:
    # Extract method name from filename
    # Format: 10.3_FY2025_{METHOD}_detailed.xlsx
    filename = file_path.stem
    method = filename.replace('10.3_FY2025_', '').replace('_detailed', '')
    available_methods.append((method, file_path))

if not available_methods:
    print("✗ ERROR: No method detail files found!")
    print("  Expected files like: 10.3_FY2025_MULTIPLIER_detailed.xlsx")
    exit()

print(f"✓ Found {len(available_methods)} methods:")
for method, file_path in available_methods:
    print(f"  - {method}: {file_path.name}")

# ================================================================
# Create individual cumulative revenue charts
# ================================================================
print("\n" + "="*70)
print("CREATING INDIVIDUAL CUMULATIVE REVENUE CHARTS")
print("="*70)

for method, file_path in available_methods:
    print(f"\nProcessing {method} method...")
    
    # Load data
    with_discount_df = pd.read_excel(file_path, sheet_name='With_Discount')
    no_discount_df = pd.read_excel(file_path, sheet_name='No_Discount')
    
    # Parse dates
    with_discount_df['payment_date'] = pd.to_datetime(with_discount_df['payment_date'])
    no_discount_df['payment_date'] = pd.to_datetime(no_discount_df['payment_date'])
    
    # Aggregate by payment month for WITH DISCOUNT
    monthly_with = with_discount_df.groupby(with_discount_df['payment_date'].dt.to_period('M')).agg({
        'credit_card_revenue': 'sum'
    }).reset_index()
    monthly_with['payment_date'] = monthly_with['payment_date'].dt.to_timestamp()
    monthly_with['cumulative'] = monthly_with['credit_card_revenue'].cumsum()
    
    # Aggregate by payment month for NO DISCOUNT
    monthly_no = no_discount_df.groupby(no_discount_df['payment_date'].dt.to_period('M')).agg({
        'interest_charged': 'sum',
        'retained_discounts': 'sum',
        'credit_card_revenue': 'sum'
    }).reset_index()
    monthly_no['payment_date'] = monthly_no['payment_date'].dt.to_timestamp()
    monthly_no['cumulative_interest'] = monthly_no['interest_charged'].cumsum()
    monthly_no['cumulative_retained'] = monthly_no['retained_discounts'].cumsum()
    monthly_no['cumulative_total'] = monthly_no['credit_card_revenue'].cumsum()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Plot WITH DISCOUNT (simple line)
    ax.plot(monthly_with['payment_date'], monthly_with['cumulative'], 
            marker='o', linewidth=3, label='With Discount (Interest Only)', 
            color='#70AD47', markersize=10, markeredgecolor='white', markeredgewidth=2)
    
    # Plot NO DISCOUNT as stacked area to show components
    ax.fill_between(monthly_no['payment_date'], 0, monthly_no['cumulative_interest'],
                     alpha=0.3, color='#4472C4', label='No Discount - Interest')
    ax.fill_between(monthly_no['payment_date'], monthly_no['cumulative_interest'], 
                     monthly_no['cumulative_total'],
                     alpha=0.3, color='#8FAADC', label='No Discount - Retained Discounts')
    
    # Plot NO DISCOUNT total line
    ax.plot(monthly_no['payment_date'], monthly_no['cumulative_total'], 
            marker='s', linewidth=3, label='No Discount (Total)', 
            color='#4472C4', markersize=10, markeredgecolor='white', markeredgewidth=2)
    
    # Add TARGET LINE at $1.043M
    ax.axhline(y=TARGET_REVENUE, color='red', linestyle='--', linewidth=3, 
               label=f'Target Revenue (${TARGET_REVENUE:,.0f})', alpha=0.8)
    
    # Find when NO DISCOUNT scenario crosses target (for vertical line only)
    cross_idx = None
    for i, val in enumerate(monthly_no['cumulative_total']):
        if val >= TARGET_REVENUE:
            cross_idx = i
            break
    
    if cross_idx is not None:
        cross_date = monthly_no['payment_date'].iloc[cross_idx]
        # Add vertical line at crossing point (no annotation box)
        ax.axvline(x=cross_date, color='red', linestyle=':', linewidth=2, alpha=0.5)
    
    # Formatting
    ax.set_title(f'{method} Method - Cumulative Revenue Over Time\n{FY2025_START} - {FY2025_END}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Payment Month', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Revenue ($)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # Add final values as simple text annotations
    final_with = monthly_with['cumulative'].iloc[-1]
    final_no = monthly_no['cumulative_total'].iloc[-1]
    final_interest = monthly_no['cumulative_interest'].iloc[-1]
    final_retained = monthly_no['cumulative_retained'].iloc[-1]
    
    # Annotation for WITH DISCOUNT
    ax.annotate(f'${final_with:,.0f}', 
                xy=(monthly_with['payment_date'].iloc[-1], final_with),
                xytext=(10, -30), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#70AD47',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='#70AD47', linewidth=2))
    
    # Annotation for NO DISCOUNT
    ax.annotate(f'${final_no:,.0f}', 
                xy=(monthly_no['payment_date'].iloc[-1], final_no),
                xytext=(10, 20), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#4472C4',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='#4472C4', linewidth=2))
    
    # Add summary text box with key metrics
    late_rate = (no_discount_df['is_late'].sum() / len(no_discount_df)) * 100
    
    summary_text = (
        f"Late Payment Rate: {late_rate:.1f}%\n"
        f"With Discount Total: ${final_with:,.0f}\n"
        f"No Discount Total: ${final_no:,.0f}\n"
        f"  (Interest: ${final_interest:,.0f})\n"
        f"  (Retained: ${final_retained:,.0f})\n"
        f"Difference: ${final_no - final_with:,.0f}"
    )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                edgecolor='black', linewidth=2)
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    output_file = visualisations_dir / f'10.3.1_cumulative_revenue_{method}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()

# ================================================================
# Summary
# ================================================================
print("\n" + "="*70)
print("SEPARATE CUMULATIVE REVENUE CHARTS COMPLETE")
print("="*70)
print(f"\nFiles created in: {visualisations_dir}")
print(f"\nGenerated {len(available_methods)} cumulative revenue charts:")
for method, _ in available_methods:
    print(f"  - 10.3.1_cumulative_revenue_{method}.png")
print("\nEach chart shows:")
print("  • WITH DISCOUNT: Interest revenue only (green line)")
print("  • NO DISCOUNT: Interest + Retained discounts (blue stacked area)")
print("  • Target revenue line at $1,043,000 (red dashed)")
print("  • Vertical line when target is reached (red dotted)")
print("  • Summary metrics in text box")
print("="*70)