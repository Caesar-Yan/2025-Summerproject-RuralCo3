'''
Docstring for 09.4_monthly_invoices_total

this script generates monthly totals for different periods, and gives statistics for invoice data


inputs:
- ats_grouped_transformed_with_discounts.csv
- invoice_grouped_transformed_with_discounts.csv

outputs:
- 9.4_monthly_discounted_bar_Period_X.png
- 9.4_monthly_discounted_line_Period_X.png
- 9.4_monthly_discounted_cumulative_Period_X.png
- 9.4_monthly_invoice_count_bar_Period_X.png
- 9.4_monthly_invoice_count_line_Period_X.png
- 9.4_monthly_invoice_count_cumulative_Period_X.png
- 9.4_period_comparison_summary.csv

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
        'start': pd.Timestamp("2024-01-01"),
        'end': pd.Timestamp("2024-12-31"),
        'title': 'Monthly Total Discounted Price\n2024-01-01 to 2024-12-31'
    },
    {
        'name': 'Period_2_2024-2025',
        'start': pd.Timestamp("2025-01-01"),
        'end': pd.Timestamp("2025-12-31"),
        'title': 'Monthly Total Discounted Price\n2025-01-011 to 2025-12-31'
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
# Helper function to prepare monthly data
# ================================================================
def prepare_monthly_data(df, period_info):
    """Prepare monthly aggregated data for a given period"""
    period_start = period_info['start']
    period_end = period_info['end']
    
    # Filter data for this period
    period_df = df[
        (df['invoice_period'] >= period_start) & 
        (df['invoice_period'] <= period_end)
    ].copy()
    
    if len(period_df) == 0:
        return None
    
    # Aggregate by month
    monthly_totals = period_df.groupby(period_df['invoice_period'].dt.to_period('M')).agg({
        'total_discounted_price': 'sum',
        'total_undiscounted_price': 'sum',
        'discount_amount': 'sum'
    }).reset_index()
    
    monthly_totals['invoice_period'] = monthly_totals['invoice_period'].dt.to_timestamp()
    monthly_totals['n_invoices'] = period_df.groupby(period_df['invoice_period'].dt.to_period('M')).size().values
    
    return monthly_totals

# ================================================================
# Function to create BAR plot
# ================================================================
def create_period_plot_bar(df, period_info, save_path):
    """Create a bar plot for a specific time period"""
    
    period_name = period_info['name']
    period_title = period_info['title']
    
    # Get monthly data
    monthly_totals = prepare_monthly_data(df, period_info)
    
    if monthly_totals is None:
        print(f"  ⚠ No data for this period - skipping bar plot")
        return None
    
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
    ax.set_title(period_title + ' (Bar Chart)', fontsize=16, fontweight='bold', pad=20)
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Bar plot saved to: {save_path.name}")
    plt.close()
    
    return monthly_totals

# ================================================================
# Function to create LINE plot
# ================================================================
def create_period_plot_line(df, period_info, save_path):
    """Create a line plot for a specific time period"""
    
    period_name = period_info['name']
    period_title = period_info['title']
    
    # Get monthly data
    monthly_totals = prepare_monthly_data(df, period_info)
    
    if monthly_totals is None:
        print(f"  ⚠ No data for this period - skipping line plot")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Line plot of monthly discounted price
    ax.plot(monthly_totals['invoice_period'], 
            monthly_totals['total_discounted_price'],
            color='#4472C4',
            linewidth=2.5,
            marker='o',
            markersize=8,
            markerfacecolor='#4472C4',
            markeredgecolor='white',
            markeredgewidth=2)
    
    # Formatting
    ax.set_title(period_title + ' (Line Chart)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Total Discounted Price ($)', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on data points
    for idx, row in monthly_totals.iterrows():
        ax.text(row['invoice_period'], 
                row['total_discounted_price'],
                f"${row['total_discounted_price']:,.0f}",
                ha='center', 
                va='bottom',
                fontsize=8,
                fontweight='bold')
    
    # Add summary text box
    total_discounted = monthly_totals['total_discounted_price'].sum()
    total_invoices = monthly_totals['n_invoices'].sum()
    avg_monthly = monthly_totals['total_discounted_price'].mean()
    
    summary_text = f"Period Total: ${total_discounted:,.0f}\nTotal Invoices: {total_invoices:,}\nAvg Monthly: ${avg_monthly:,.0f}"
    ax.text(0.02, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Line plot saved to: {save_path.name}")
    plt.close()
    
    return monthly_totals

# ================================================================
# Function to create CUMULATIVE plot
# ================================================================
def create_period_plot_cumulative(df, period_info, save_path):
    """Create a cumulative plot for a specific time period"""
    
    period_name = period_info['name']
    period_title = period_info['title']
    
    # Get monthly data
    monthly_totals = prepare_monthly_data(df, period_info)
    
    if monthly_totals is None:
        print(f"  ⚠ No data for this period - skipping cumulative plot")
        return None
    
    # Calculate cumulative sum
    monthly_totals['cumulative_discounted'] = monthly_totals['total_discounted_price'].cumsum()
    monthly_totals['cumulative_invoices'] = monthly_totals['n_invoices'].cumsum()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Line plot of cumulative discounted price
    ax.plot(monthly_totals['invoice_period'], 
            monthly_totals['cumulative_discounted'],
            color='#27AE60',
            linewidth=3,
            marker='o',
            markersize=8,
            markerfacecolor='#27AE60',
            markeredgecolor='white',
            markeredgewidth=2)
    
    # Fill area under the curve
    ax.fill_between(monthly_totals['invoice_period'],
                     monthly_totals['cumulative_discounted'],
                     alpha=0.3,
                     color='#27AE60')
    
    # Formatting
    ax.set_title(period_title.replace('Monthly Total', 'Cumulative') + ' (Cumulative)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Cumulative Discounted Price ($)', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on data points (every other point to avoid crowding)
    for idx, row in monthly_totals.iterrows():
        if idx % 2 == 0 or idx == len(monthly_totals) - 1:  # Show every other point + last point
            ax.text(row['invoice_period'], 
                    row['cumulative_discounted'],
                    f"${row['cumulative_discounted']:,.0f}",
                    ha='center', 
                    va='bottom',
                    fontsize=8,
                    fontweight='bold')
    
    # Add summary text box
    final_total = monthly_totals['cumulative_discounted'].iloc[-1]
    total_invoices = monthly_totals['cumulative_invoices'].iloc[-1]
    months_count = len(monthly_totals)
    
    summary_text = f"Final Cumulative: ${final_total:,.0f}\nTotal Invoices: {total_invoices:,}\nMonths: {months_count}"
    ax.text(0.02, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Cumulative plot saved to: {save_path.name}")
    plt.close()
    
    return monthly_totals

# ================================================================
# Function to create BAR plot - INVOICE COUNT
# ================================================================
def create_invoice_count_plot_bar(df, period_info, save_path):
    """Create a bar plot of invoice counts for a specific time period"""
    
    period_name = period_info['name']
    period_title = period_info['title'].replace('Monthly Total Discounted Price', 'Monthly Invoice Count')
    
    # Get monthly data
    monthly_totals = prepare_monthly_data(df, period_info)
    
    if monthly_totals is None:
        print(f"  ⚠ No data for this period - skipping invoice count bar plot")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Bar plot of monthly invoice count
    ax.bar(monthly_totals['invoice_period'], 
           monthly_totals['n_invoices'],
           width=20,  # width in days
           color='#E67E22',
           alpha=0.7,
           edgecolor='black',
           linewidth=0.5)
    
    # Formatting
    ax.set_title(period_title + ' (Bar Chart)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Number of Invoices', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for idx, row in monthly_totals.iterrows():
        ax.text(row['invoice_period'], 
                row['n_invoices'],
                f"{row['n_invoices']:,}",
                ha='center', 
                va='bottom',
                fontsize=9,
                fontweight='bold')
    
    # Add summary text box
    total_invoices = monthly_totals['n_invoices'].sum()
    avg_monthly_invoices = monthly_totals['n_invoices'].mean()
    
    summary_text = f"Total Invoices: {total_invoices:,}\nAvg Monthly: {avg_monthly_invoices:,.0f}"
    ax.text(0.02, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Invoice count bar plot saved to: {save_path.name}")
    plt.close()
    
    return monthly_totals

# ================================================================
# Function to create LINE plot - INVOICE COUNT
# ================================================================
def create_invoice_count_plot_line(df, period_info, save_path):
    """Create a line plot of invoice counts for a specific time period"""
    
    period_name = period_info['name']
    period_title = period_info['title'].replace('Monthly Total Discounted Price', 'Monthly Invoice Count')
    
    # Get monthly data
    monthly_totals = prepare_monthly_data(df, period_info)
    
    if monthly_totals is None:
        print(f"  ⚠ No data for this period - skipping invoice count line plot")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Line plot of monthly invoice count
    ax.plot(monthly_totals['invoice_period'], 
            monthly_totals['n_invoices'],
            color='#E67E22',
            linewidth=2.5,
            marker='o',
            markersize=8,
            markerfacecolor='#E67E22',
            markeredgecolor='white',
            markeredgewidth=2)
    
    # Formatting
    ax.set_title(period_title + ' (Line Chart)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Number of Invoices', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on data points
    for idx, row in monthly_totals.iterrows():
        ax.text(row['invoice_period'], 
                row['n_invoices'],
                f"{row['n_invoices']:,}",
                ha='center', 
                va='bottom',
                fontsize=8,
                fontweight='bold')
    
    # Add summary text box
    total_invoices = monthly_totals['n_invoices'].sum()
    avg_monthly_invoices = monthly_totals['n_invoices'].mean()
    max_invoices = monthly_totals['n_invoices'].max()
    min_invoices = monthly_totals['n_invoices'].min()
    
    summary_text = f"Total: {total_invoices:,}\nAvg: {avg_monthly_invoices:,.0f}\nMax: {max_invoices:,}\nMin: {min_invoices:,}"
    ax.text(0.02, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#FFE5CC', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Invoice count line plot saved to: {save_path.name}")
    plt.close()
    
    return monthly_totals

# ================================================================
# Function to create CUMULATIVE plot - INVOICE COUNT
# ================================================================
def create_invoice_count_plot_cumulative(df, period_info, save_path):
    """Create a cumulative plot of invoice counts for a specific time period"""
    
    period_name = period_info['name']
    period_title = period_info['title'].replace('Monthly Total Discounted Price', 'Cumulative Invoice Count')
    
    # Get monthly data
    monthly_totals = prepare_monthly_data(df, period_info)
    
    if monthly_totals is None:
        print(f"  ⚠ No data for this period - skipping invoice count cumulative plot")
        return None
    
    # Calculate cumulative sum
    monthly_totals['cumulative_invoices'] = monthly_totals['n_invoices'].cumsum()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Line plot of cumulative invoice count
    ax.plot(monthly_totals['invoice_period'], 
            monthly_totals['cumulative_invoices'],
            color='#8E44AD',
            linewidth=3,
            marker='o',
            markersize=8,
            markerfacecolor='#8E44AD',
            markeredgecolor='white',
            markeredgewidth=2)
    
    # Fill area under the curve
    ax.fill_between(monthly_totals['invoice_period'],
                     monthly_totals['cumulative_invoices'],
                     alpha=0.3,
                     color='#8E44AD')
    
    # Formatting
    ax.set_title(period_title + ' (Cumulative)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Cumulative Number of Invoices', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on data points (every other point to avoid crowding)
    for idx, row in monthly_totals.iterrows():
        if idx % 2 == 0 or idx == len(monthly_totals) - 1:  # Show every other point + last point
            ax.text(row['invoice_period'], 
                    row['cumulative_invoices'],
                    f"{row['cumulative_invoices']:,}",
                    ha='center', 
                    va='bottom',
                    fontsize=8,
                    fontweight='bold')
    
    # Add summary text box
    final_total = monthly_totals['cumulative_invoices'].iloc[-1]
    months_count = len(monthly_totals)
    avg_monthly = monthly_totals['n_invoices'].mean()
    
    summary_text = f"Final Total: {final_total:,}\nMonths: {months_count}\nAvg Monthly: {avg_monthly:,.0f}"
    ax.text(0.02, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#E8DAEF', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Invoice count cumulative plot saved to: {save_path.name}")
    plt.close()
    
    return monthly_totals

# ================================================================
# Create all plot types for each period
# ================================================================
print("\n" + "="*70)
print("CREATING PERIOD-SPECIFIC PLOTS")
print("="*70)

all_monthly_data = {}
created_files = []

for period in PERIODS:
    print(f"\n{period['name']}:")
    print(f"  Date range: {period['start'].strftime('%Y-%m-%d')} to {period['end'].strftime('%Y-%m-%d')}")
    
    # PRICE PLOTS
    # Bar plot
    save_path_bar = visualisations_dir / f"9.4_monthly_discounted_bar_{period['name']}.png"
    monthly_data = create_period_plot_bar(combined_df, period, save_path_bar)
    if monthly_data is not None:
        all_monthly_data[period['name']] = monthly_data
        created_files.append(save_path_bar.name)
        print(f"  Invoices: {monthly_data['n_invoices'].sum():,}")
        print(f"  Total discounted: ${monthly_data['total_discounted_price'].sum():,.2f}")
    
    # Line plot
    save_path_line = visualisations_dir / f"9.4_monthly_discounted_line_{period['name']}.png"
    create_period_plot_line(combined_df, period, save_path_line)
    if monthly_data is not None:
        created_files.append(save_path_line.name)
    
    # Cumulative plot
    save_path_cumulative = visualisations_dir / f"9.4_monthly_discounted_cumulative_{period['name']}.png"
    create_period_plot_cumulative(combined_df, period, save_path_cumulative)
    if monthly_data is not None:
        created_files.append(save_path_cumulative.name)
    
    # INVOICE COUNT PLOTS
    # Bar plot
    save_path_count_bar = visualisations_dir / f"9.4_monthly_invoice_count_bar_{period['name']}.png"
    create_invoice_count_plot_bar(combined_df, period, save_path_count_bar)
    if monthly_data is not None:
        created_files.append(save_path_count_bar.name)
    
    # Line plot
    save_path_count_line = visualisations_dir / f"9.4_monthly_invoice_count_line_{period['name']}.png"
    create_invoice_count_plot_line(combined_df, period, save_path_count_line)
    if monthly_data is not None:
        created_files.append(save_path_count_line.name)
    
    # Cumulative plot
    save_path_count_cumulative = visualisations_dir / f"9.4_monthly_invoice_count_cumulative_{period['name']}.png"
    create_invoice_count_plot_cumulative(combined_df, period, save_path_count_cumulative)
    if monthly_data is not None:
        created_files.append(save_path_count_cumulative.name)

# ================================================================
# Create plots for ENTIRE period (2023-12-01 to 2025-12-31)
# ================================================================
print("\n" + "="*70)
print("CREATING PLOTS FOR ENTIRE PERIOD (2023-12-01 to 2025-12-31)")
print("="*70)

entire_period_info = {
    'name': 'Period_4_Entire',
    'start': pd.Timestamp("2023-12-01"),
    'end': pd.Timestamp("2025-12-31"),
    'title': 'Monthly Total Discounted Price\nEntire Period (2023-12-01 to 2025-12-31)'
}

print(f"\n{entire_period_info['name']}:")

# PRICE PLOTS
# Bar plot
save_path_bar_entire = visualisations_dir / f"9.4_monthly_discounted_bar_{entire_period_info['name']}.png"
entire_monthly_data = create_period_plot_bar(combined_df, entire_period_info, save_path_bar_entire)
if entire_monthly_data is not None:
    all_monthly_data[entire_period_info['name']] = entire_monthly_data
    created_files.append(save_path_bar_entire.name)
    print(f"  Invoices: {entire_monthly_data['n_invoices'].sum():,}")
    print(f"  Total discounted: ${entire_monthly_data['total_discounted_price'].sum():,.2f}")

# Line plot
save_path_line_entire = visualisations_dir / f"9.4_monthly_discounted_line_{entire_period_info['name']}.png"
create_period_plot_line(combined_df, entire_period_info, save_path_line_entire)
if entire_monthly_data is not None:
    created_files.append(save_path_line_entire.name)

# Cumulative plot
save_path_cumulative_entire = visualisations_dir / f"9.4_monthly_discounted_cumulative_{entire_period_info['name']}.png"
create_period_plot_cumulative(combined_df, entire_period_info, save_path_cumulative_entire)
if entire_monthly_data is not None:
    created_files.append(save_path_cumulative_entire.name)

# INVOICE COUNT PLOTS
# Bar plot
save_path_count_bar_entire = visualisations_dir / f"9.4_monthly_invoice_count_bar_{entire_period_info['name']}.png"
create_invoice_count_plot_bar(combined_df, entire_period_info, save_path_count_bar_entire)
if entire_monthly_data is not None:
    created_files.append(save_path_count_bar_entire.name)

# Line plot
save_path_count_line_entire = visualisations_dir / f"9.4_monthly_invoice_count_line_{entire_period_info['name']}.png"
create_invoice_count_plot_line(combined_df, entire_period_info, save_path_count_line_entire)
if entire_monthly_data is not None:
    created_files.append(save_path_count_line_entire.name)

# Cumulative plot
save_path_count_cumulative_entire = visualisations_dir / f"9.4_monthly_invoice_count_cumulative_{entire_period_info['name']}.png"
create_invoice_count_plot_cumulative(combined_df, entire_period_info, save_path_count_cumulative_entire)
if entire_monthly_data is not None:
    created_files.append(save_path_count_cumulative_entire.name)

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
        'Num_Months': len(monthly_df),
        'Avg_Monthly_Discounted': monthly_df['total_discounted_price'].mean(),
        'Avg_Monthly_Invoices': monthly_df['n_invoices'].mean()
    })

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(profile_dir / '9.4_period_comparison_summary.csv', index=False)
print(f"\n✓ Saved summary to: 9.4_period_comparison_summary.csv")
created_files.append('9.4_period_comparison_summary.csv')

print("\n" + "="*70)
print("ALL VISUALIZATIONS COMPLETE")
print("="*70)
print(f"\nTotal files created: {len(created_files)}")
print("\nFiles created:")
for i, filename in enumerate(created_files, 1):
    print(f"  {i}. {filename}")
print("="*70)