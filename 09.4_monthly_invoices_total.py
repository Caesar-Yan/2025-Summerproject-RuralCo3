'''
Docstring for 09.4_monthly_invoices_total

this script generates monthly totals for different periods, and gives statistics for invoice data


inputs:
- ats_grouped_transformed_with_discounts.csv
- invoice_grouped_transformed_with_discounts.csv

outputs:
- monthly_discounted_Period_1_2023-2024.png
- monthly_discounted_Period_2_2024-2025.png
- monthly_discounted_Period_3_FY2025.png
- monthly_discounted_Period_4_Entire.png
- period_comparison_summary.csv
- monthly_totals_Period_1_2023-2024.csv
- monthly_totals_Period_2_2024-2025.csv
- monthly_totals_Period_3_FY2025.csv
- monthly_totals_Period_4_Entire.csv

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from pathlib import Path

# ================================================================
# Configuration
# ================================================================
# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
data_cleaning_dir = base_dir / "data_cleaning"
visualisations_dir = base_dir / "visualisations"
visualisations_dir.mkdir(exist_ok=True) 

# ================================================================
# Load invoice data
# ================================================================
print("="*70)
print("LOADING INVOICE DATA")
print("="*70)

# Load combined invoice data
ats_grouped = pd.read_csv(data_cleaning_dir / 'ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv(data_cleaning_dir / 'invoice_grouped_transformed_with_discounts.csv')

# Combine datasets
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"Total invoices loaded: {len(combined_df):,}")

# Filter out negative prices
combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()
print(f"After filtering negatives: {len(combined_df):,}")

# Define the four time periods
PERIODS = [
    {
        'name': 'Period_1_2023-2024',
        'start': pd.Timestamp("2023-12-20"),
        'end': pd.Timestamp("2024-12-20"),
        'title': 'Monthly Total Discounted Price\n2023-12-20 to 2024-12-20'
    },
    {
        'name': 'Period_2_2024-2025',
        'start': pd.Timestamp("2024-12-21"),
        'end': pd.Timestamp("2025-12-20"),
        'title': 'Monthly Total Discounted Price\n2024-12-21 to 2025-12-20'
    },
    {
        'name': 'Period_3_FY2025',
        'start': pd.Timestamp("2024-06-30"),
        'end': pd.Timestamp("2025-07-01"),
        'title': 'Monthly Total Discounted Price\nFY2025 (2024-06-30 to 2025-07-01)'
    }
]

# ================================================================
# Parse dates
# ================================================================
def parse_invoice_period(series: pd.Series) -> pd.Series:
    """Robustly parse invoice_period"""
    s = series.copy()
    s_str = s.astype(str).str.strip()
    s_str = s_str.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
    
    # Handle YYYYMM format
    mask_yyyymm = s_str.str.fullmatch(r"\d{6}", na=False)
    out = pd.Series(pd.NaT, index=s.index)
    
    if mask_yyyymm.any():
        out.loc[mask_yyyymm] = pd.to_datetime(s_str.loc[mask_yyyymm], format="%Y%m", errors="coerce")
    
    # Handle other formats
    mask_other = ~mask_yyyymm
    if mask_other.any():
        out.loc[mask_other] = pd.to_datetime(s_str.loc[mask_other], errors="coerce")
    
    return out

combined_df['invoice_period'] = parse_invoice_period(combined_df['invoice_period'])
combined_df = combined_df[combined_df['invoice_period'].notna()].copy()

print(f"Total invoices with valid dates: {len(combined_df):,}")
print(f"Date range: {combined_df['invoice_period'].min()} to {combined_df['invoice_period'].max()}")

# ================================================================
# Function to create plot for a given period
# ================================================================
def create_period_plot(df, period_info, save_path):
    """Create a bar plot for a specific time period"""
    
    period_start = period_info['start']
    period_end = period_info['end']
    period_name = period_info['name']
    period_title = period_info['title']
    
    # Filter data for this period
    period_df = df[
        (df['invoice_period'] >= period_start) & 
        (df['invoice_period'] <= period_end)
    ].copy()
    
    print(f"\n{period_name}:")
    print(f"  Date range: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
    print(f"  Invoices: {len(period_df):,}")
    
    if len(period_df) == 0:
        print(f"  ⚠ No data for this period - skipping plot")
        return None
    
    # Aggregate by month
    monthly_totals = period_df.groupby(period_df['invoice_period'].dt.to_period('M')).agg({
        'total_discounted_price': 'sum',
        'total_undiscounted_price': 'sum',
        'discount_amount': 'sum'
    }).reset_index()
    
    monthly_totals['invoice_period'] = monthly_totals['invoice_period'].dt.to_timestamp()
    monthly_totals['n_invoices'] = period_df.groupby(period_df['invoice_period'].dt.to_period('M')).size().values
    
    print(f"  Months with data: {len(monthly_totals)}")
    print(f"  Total discounted: ${monthly_totals['total_discounted_price'].sum():,.2f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Bar plot of monthly discounted price
    ax.bar(monthly_totals['invoice_period'], 
           monthly_totals['total_discounted_price'],
           width=20,  # width in days
           color='#4472C4',
           alpha=0.7,
           edgecolor='black',
           linewidth=0.5)
    
    # Formatting
    ax.set_title(period_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Total Discounted Price ($)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for idx, row in monthly_totals.iterrows():
        ax.text(row['invoice_period'], 
                row['total_discounted_price'],
                f"${row['total_discounted_price']:,.0f}\n({row['n_invoices']:,})",
                ha='center', 
                va='bottom',
                fontsize=8,
                fontweight='bold')
    
    # Add summary text box
    total_discounted = monthly_totals['total_discounted_price'].sum()
    total_invoices = monthly_totals['n_invoices'].sum()
    
    summary_text = f"Period Total: ${total_discounted:,.0f}\nTotal Invoices: {total_invoices:,}"
    ax.text(0.02, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    
    plt.close()
    
    return monthly_totals

# ================================================================
# Create plots for each period
# ================================================================
print("\n" + "="*70)
print("CREATING PERIOD-SPECIFIC PLOTS")
print("="*70)

all_monthly_data = {}

for period in PERIODS:
    save_path = visualisations_dir / f"monthly_discounted_{period['name']}.png"
    monthly_data = create_period_plot(combined_df, period, save_path)
    if monthly_data is not None:
        all_monthly_data[period['name']] = monthly_data
        # Save to CSV
        csv_path = visualisations_dir / f"monthly_totals_{period['name']}.csv"
        monthly_data.to_csv(csv_path, index=False)
        print(f"  ✓ Saved monthly data CSV: {csv_path.name}")

# ================================================================
# Create plot for ENTIRE period (2023-12-01 to 2025-12-31)
# ================================================================
print("\n" + "="*70)
print("CREATING PLOT FOR ENTIRE PERIOD (2023-12-01 to 2025-12-31)")
print("="*70)

entire_period_info = {
    'name': 'Period_4_Entire',
    'start': pd.Timestamp("2023-12-01"),
    'end': pd.Timestamp("2025-12-31"),
    'title': 'Monthly Total Discounted Price\nEntire Period (2023-12-01 to 2025-12-31)'
}

save_path_entire = visualisations_dir / f"monthly_discounted_{entire_period_info['name']}.png"
entire_monthly_data = create_period_plot(combined_df, entire_period_info, save_path_entire)
if entire_monthly_data is not None:
    all_monthly_data[entire_period_info['name']] = entire_monthly_data
    # Save to CSV
    csv_path = visualisations_dir / f"monthly_totals_{entire_period_info['name']}.csv"
    entire_monthly_data.to_csv(csv_path, index=False)
    print(f"  ✓ Saved monthly data CSV: {csv_path.name}")

# ================================================================
# Create comparison summary
# ================================================================
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)

summary_data = []

for period_name, monthly_df in all_monthly_data.items():
    summary_data.append({
        'Period': period_name,
        'Total_Discounted': monthly_df['total_discounted_price'].sum(),
        'Total_Undiscounted': monthly_df['total_undiscounted_price'].sum(),
        'Total_Discount_Amount': monthly_df['discount_amount'].sum(),
        'Total_Invoices': monthly_df['n_invoices'].sum(),
        'Num_Months': len(monthly_df)
    })

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(profile_dir / '9.4_period_comparison_summary.csv', index=False)
print(f"\n✓ Saved summary to: period_comparison_summary.csv")

print("\n" + "="*70)
print("ALL VISUALIZATIONS COMPLETE")
print("="*70)
print("\nFiles created:")
for i, period in enumerate(PERIODS + [entire_period_info], 1):
    print(f"  {i}. monthly_discounted_{period['name']}.png")
    print(f"     monthly_totals_{period['name']}.csv")
print(f"  {len(PERIODS)+2}. period_comparison_summary.csv")
print("="*70)