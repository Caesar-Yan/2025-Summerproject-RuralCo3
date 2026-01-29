"""
11.6_forecast_overlay - Overlay Discounted vs Undiscounted Forecasts

This script loads the forecasts from 11.3 (discounted) and 11.5 (undiscounted),
along with historical actuals from 9.4, and creates visualizations showing:
- Historical actual discounted and undiscounted amounts
- Forecasted discounted and undiscounted amounts
- Overlay comparison of actuals vs forecasts

Inputs:
-------
- visualisations/9.4_monthly_totals_Period_4_Entire.csv (historical actuals)
- visualisations/11.3_forecast_next_15_months.csv (discounted forecast)
- forecast/11.5_forecast_undiscounted_next_15_months.csv (undiscounted forecast)

Outputs:
--------
- visualisations/11.6_forecast_overlay.png (main overlay plot with actuals + forecasts)
- visualisations/11.6_forecast_comparison_table.csv (summary statistics)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path(r"\\file\Usersc$\cch155\Home\Desktop\2025\data605\2025-Summerproject-RuralCo3")
ALT_BASE = Path(r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3")

VIS_DIR = ALT_BASE / 'visualisations'
FORECAST_DIR = ALT_BASE / 'forecast'

ALT_VIS = ALT_BASE / 'visualisations'
ALT_FORECAST = ALT_BASE / 'forecast'

# Ensure output directories exist
VIS_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("FORECAST OVERLAY: DISCOUNTED vs UNDISCOUNTED")
print("Including Historical Actuals from 9.4")
print("="*80)
print(f"\n📁 Output directory: {VIS_DIR}")
print(f"   (Full path: {VIS_DIR.resolve()})")

# Load historical actuals
print("\n📁 Loading historical actuals...")

historical_path = VIS_DIR / '9.4_monthly_totals_Period_4_Entire.csv'
if not historical_path.exists():
    historical_path = ALT_VIS / '9.4_monthly_totals_Period_4_Entire.csv'

historical = pd.read_csv(historical_path)
historical['invoice_period'] = pd.to_datetime(historical['invoice_period'])
print(f"  ✓ Loaded 9.4 (historical actuals): {len(historical)} months")

# Load forecasts
print("\n📁 Loading forecasts...")

f11_3_path = VIS_DIR / '11.3_forecast_next_15_months.csv'
if not f11_3_path.exists():
    f11_3_path = ALT_VIS / '11.3_forecast_next_15_months.csv'

f11_5_path = FORECAST_DIR / '11.5_forecast_undiscounted_next_15_months.csv'
if not f11_5_path.exists():
    f11_5_path = ALT_FORECAST / '11.5_forecast_undiscounted_next_15_months.csv'

f11_3 = pd.read_csv(f11_3_path)
f11_5 = pd.read_csv(f11_5_path)

f11_3['invoice_period'] = pd.to_datetime(f11_3['invoice_period'])
f11_5['invoice_period'] = pd.to_datetime(f11_5['invoice_period'])

print(f"  ✓ Loaded 11.3 (discounted): {len(f11_3)} months")
print(f"  ✓ Loaded 11.5 (undiscounted): {len(f11_5)} months")

# Merge on invoice_period
merged = f11_3[['invoice_period', 'forecast_discounted_price']].merge(
    f11_5[['invoice_period', 'forecast_undiscounted_price']],
    on='invoice_period',
    how='inner'
)

merged = merged.sort_values('invoice_period').reset_index(drop=True)
# Prepare historical data: select only discounted and undiscounted prices
historical_subset = historical[['invoice_period', 'total_discounted_price', 'total_undiscounted_price']].copy()
historical_subset.columns = ['invoice_period', 'actual_discounted_price', 'actual_undiscounted_price']
historical_subset = historical_subset.sort_values('invoice_period').reset_index(drop=True)
# Calculate discount amount and rate
merged['forecast_discount_amount'] = merged['forecast_undiscounted_price'] - merged['forecast_discounted_price']
merged['discount_rate_pct'] = (merged['forecast_discount_amount'] / merged['forecast_undiscounted_price'] * 100)

print(f"\n📊 Merged data: {len(merged)} matching months")
print(f"  Period: {merged['invoice_period'].min().strftime('%Y-%m')} to {merged['invoice_period'].max().strftime('%Y-%m')}")

# Calculate period summaries
print(f"\n📅 Generating period summaries...")

# Period 1: Forecast period 2026-01-01 to 2027-01-01
forecast_2026_2027 = merged[(merged['invoice_period'] >= '2026-01-01') & 
                             (merged['invoice_period'] <= '2027-01-01')].copy()

# Period 2: Historical period 2024-07-01 to 2025-07-01
historical_2024_2025 = historical_subset[(historical_subset['invoice_period'] >= '2024-07-01') & 
                                         (historical_subset['invoice_period'] <= '2025-07-01')].copy()

# Create period summary table
period_summary = []

# Add forecast period
if not forecast_2026_2027.empty:
    period_summary.append({
        'period': '2026-01-01 to 2027-01-01 (Forecast)',
        'type': 'Forecast',
        'months': len(forecast_2026_2027),
        'total_undiscounted': forecast_2026_2027['forecast_undiscounted_price'].sum(),
        'total_discount': forecast_2026_2027['forecast_discount_amount'].sum(),
        'net_revenue': forecast_2026_2027['forecast_discounted_price'].sum(),
        'avg_discount_rate': forecast_2026_2027['discount_rate_pct'].mean()
    })

# Add historical period
if not historical_2024_2025.empty:
    historical_2024_2025['actual_discount_amount'] = (historical_2024_2025['actual_undiscounted_price'] - 
                                                      historical_2024_2025['actual_discounted_price'])
    period_summary.append({
        'period': '2024-07-01 to 2025-07-01 (Historical)',
        'type': 'Historical',
        'months': len(historical_2024_2025),
        'total_undiscounted': historical_2024_2025['actual_undiscounted_price'].sum(),
        'total_discount': historical_2024_2025['actual_discount_amount'].sum(),
        'net_revenue': historical_2024_2025['actual_discounted_price'].sum(),
        'avg_discount_rate': (historical_2024_2025['actual_discount_amount'].sum() / 
                             historical_2024_2025['actual_undiscounted_price'].sum() * 100)
    })

period_summary_df = pd.DataFrame(period_summary)
output_period_csv = VIS_DIR / '11.6_period_revenue_summary.csv'
period_summary_df.to_csv(output_period_csv, index=False)
print(f"  ✓ Generated period summary table")

print(f"\n📋 Period Revenue Summary:")
print(f"\n  Period 1: Forecast (2026-01-01 to 2027-01-01)")
if not forecast_2026_2027.empty:
    print(f"    Months: {len(forecast_2026_2027)}")
    print(f"    Total Undiscounted: ${forecast_2026_2027['forecast_undiscounted_price'].sum():>15,.2f}")
    print(f"    Total Discount:    ${forecast_2026_2027['forecast_discount_amount'].sum():>15,.2f}")
    print(f"    Net Revenue:       ${forecast_2026_2027['forecast_discounted_price'].sum():>15,.2f}")
    print(f"    Avg Discount Rate: {forecast_2026_2027['discount_rate_pct'].mean():>15.2f}%")
else:
    print(f"    No data in this period")

print(f"\n  Period 2: Historical (2024-07-01 to 2025-07-01)")
if not historical_2024_2025.empty:
    print(f"    Months: {len(historical_2024_2025)}")
    print(f"    Total Undiscounted: ${historical_2024_2025['actual_undiscounted_price'].sum():>15,.2f}")
    print(f"    Total Discount:    ${historical_2024_2025['actual_discount_amount'].sum():>15,.2f}")
    print(f"    Net Revenue:       ${historical_2024_2025['actual_discounted_price'].sum():>15,.2f}")
    print(f"    Avg Discount Rate: {(historical_2024_2025['actual_discount_amount'].sum() / historical_2024_2025['actual_undiscounted_price'].sum() * 100):>15.2f}%")
else:
    print(f"    No data in this period")

# Summary statistics
print(f"\n📈 Summary Statistics:")
print(f"\n  Discounted Forecast:")
print(f"    Mean:   ${merged['forecast_discounted_price'].mean():>15,.2f}")
print(f"    Min:    ${merged['forecast_discounted_price'].min():>15,.2f}")
print(f"    Max:    ${merged['forecast_discounted_price'].max():>15,.2f}")
print(f"    Total:  ${merged['forecast_discounted_price'].sum():>15,.2f}")

print(f"\n  Undiscounted Forecast:")
print(f"    Mean:   ${merged['forecast_undiscounted_price'].mean():>15,.2f}")
print(f"    Min:    ${merged['forecast_undiscounted_price'].min():>15,.2f}")
print(f"    Max:    ${merged['forecast_undiscounted_price'].max():>15,.2f}")
print(f"    Total:  ${merged['forecast_undiscounted_price'].sum():>15,.2f}")

print(f"\n  Discount Amount:")
print(f"    Mean:   ${merged['forecast_discount_amount'].mean():>15,.2f}")
print(f"    Min:    ${merged['forecast_discount_amount'].min():>15,.2f}")
print(f"    Max:    ${merged['forecast_discount_amount'].max():>15,.2f}")
print(f"    Total:  ${merged['forecast_discount_amount'].sum():>15,.2f}")

print(f"\n  Discount Rate:")
print(f"    Mean:   {merged['discount_rate_pct'].mean():>15.2f}%")
print(f"    Min:    {merged['discount_rate_pct'].min():>15.2f}%")
print(f"    Max:    {merged['discount_rate_pct'].max():>15.2f}%")

# Create visualization
fig, ax = plt.subplots(figsize=(14, 7))

# Plot 1: Overlay of actual and forecasted discounted/undiscounted
# Historical actuals
ax.plot(historical_subset['invoice_period'], historical_subset['actual_discounted_price'],
         marker='o', linewidth=2, markersize=6, label='Actual Discounted (9.4)', 
         color='#4472C4', linestyle='-', alpha=0.7)
ax.plot(historical_subset['invoice_period'], historical_subset['actual_undiscounted_price'],
         marker='s', linewidth=2, markersize=6, label='Actual Undiscounted (9.4)',
         color='#70AD47', linestyle='-', alpha=0.7)
# Forecasts
ax.plot(merged['invoice_period'], merged['forecast_discounted_price'],
         marker='o', linewidth=2.5, markersize=7, label='Forecast Discounted (11.3)', 
         color='#4472C4', linestyle='--', alpha=1.0)
ax.plot(merged['invoice_period'], merged['forecast_undiscounted_price'],
         marker='s', linewidth=2.5, markersize=7, label='Forecast Undiscounted (11.5)',
         color='#70AD47', linestyle='--', alpha=1.0)
ax.set_title('Actual vs Forecast: Discounted and Undiscounted Prices', fontsize=13, fontweight='bold')
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Monthly Total ($)', fontsize=11)
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

plt.tight_layout()
output_viz = VIS_DIR / '11.6_actual_vs_forecast_prices.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {output_viz.name}")
plt.close()

# Plot 2: Forecasted discount vs actual discount
fig, ax = plt.subplots(figsize=(14, 7))
# Calculate actual discount amounts
historical_subset['actual_discount_amount'] = (historical_subset['actual_undiscounted_price'] - 
                                               historical_subset['actual_discounted_price'])
# Forecast discount amounts
forecast_discount = merged['forecast_undiscounted_price'] - merged['forecast_discounted_price']
ax.bar(historical_subset['invoice_period'] - pd.Timedelta(days=5), 
        historical_subset['actual_discount_amount'],
        width=10, label='Actual Discount (9.4)', color='#4472C4', alpha=0.6, edgecolor='black', linewidth=1)
ax.bar(merged['invoice_period'] + pd.Timedelta(days=5),
        forecast_discount,
        width=10, label='Forecast Discount (11.3/11.5)', color='#C44E52', alpha=0.6, edgecolor='black', linewidth=1)
ax.set_title('Actual vs Forecasted Discount Amount', fontsize=13, fontweight='bold')
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Discount Amount ($)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

plt.tight_layout()
output_viz = VIS_DIR / '11.6_actual_vs_forecast_discount_amount.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {output_viz.name}")
plt.close()

# Plot 3: Actual vs Forecasted discount rate
fig, ax = plt.subplots(figsize=(14, 7))
historical_subset['actual_discount_rate'] = (historical_subset['actual_discount_amount'] / 
                                             historical_subset['actual_undiscounted_price'] * 100)
forecast_discount_rate = (forecast_discount / merged['forecast_undiscounted_price'] * 100)
ax.plot(historical_subset['invoice_period'], historical_subset['actual_discount_rate'],
         marker='o', linewidth=2, markersize=6, label='Actual Discount Rate (9.4)',
         color='#4472C4', linestyle='-')
ax.plot(merged['invoice_period'], forecast_discount_rate,
         marker='s', linewidth=2.5, markersize=7, label='Forecast Discount Rate (11.3/11.5)',
         color='#70AD47', linestyle='--')
ax.set_title('Actual vs Forecasted Discount Rate (% of Undiscounted)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Discount Rate (%)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

plt.tight_layout()
output_viz = VIS_DIR / '11.6_actual_vs_forecast_discount_rate.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {output_viz.name}")
plt.close()

# Plot 4: Forecast error analysis
fig, ax = plt.subplots(figsize=(14, 7))
# Calculate errors (only for overlapping months)
error_disc = merged['forecast_discounted_price'].values - merged['forecast_discounted_price'].values  # placeholder
error_undisc = merged['forecast_undiscounted_price'].values - merged['forecast_undiscounted_price'].values  # placeholder
# For now, show forecast discount rates side by side
width = 0.35
x_pos = np.arange(len(merged))
actual_rates_aligned = historical_subset['actual_discount_rate'].tail(len(merged)).values
ax.bar(x_pos - width/2, actual_rates_aligned, width, label='Actual Rate (last months)', 
        color='#4472C4', alpha=0.7, edgecolor='black', linewidth=1)
ax.bar(x_pos + width/2, forecast_discount_rate.values, width, label='Forecast Rate',
        color='#70AD47', alpha=0.7, edgecolor='black', linewidth=1)
ax.set_title('Discount Rate: Actuals vs Forecasts (Aligned)', fontsize=13, fontweight='bold')
ax.set_xlabel('Forecast Month', fontsize=11)
ax.set_ylabel('Discount Rate (%)', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels([d.strftime('%Y-%m') for d in merged['invoice_period']], rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

plt.tight_layout()
output_viz = VIS_DIR / '11.6_discount_rate_comparison.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {output_viz.name}")
plt.close()

# Save comparison table
output_csv = VIS_DIR / '11.6_forecast_comparison_table.csv'
merged.to_csv(output_csv, index=False)
print(f"✓ Saved comparison table: {output_csv.name}")

print("\n" + "="*80)
print("✅ FORECAST OVERLAY COMPLETE")
print("="*80)
print(f"\nFiles saved to: {VIS_DIR}")
print(f"  • 11.6_forecast_overlay.png")
print(f"  • 11.6_forecast_comparison_table.csv")
print(f"  • 11.6_period_revenue_summary.csv")
