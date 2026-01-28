'''
15.4_forecast_total_prices_estimated_revenue - Revenue Estimation Using Separate Forecasts

This script takes SEPARATELY forecasted discounted and undiscounted monthly totals 
and estimates revenue using statement-level decile distributions from script 15.2.

Unlike 15.3 which infers undiscounted using a constant multiplier, this script uses:
- Discounted forecast from 11.3
- Undiscounted forecast from 11.5 (direct forecast)
- Calculates discount amount as the difference

This allows for time-varying discount patterns rather than assuming constant ratios.

The approach:
1. Load both discounted (11.3) and undiscounted (11.5) forecasts
2. Calculate monthly discount amounts as difference
3. Allocate forecasted values across deciles
4. Apply decile-specific late payment rates (calibrated + seasonal)
5. Calculate interest revenue and retained discounts

Inputs:
-------
- visualisations/11.3_forecast_next_15_months.csv (discounted forecast)
- forecast/11.5_forecast_undiscounted_next_15_months.csv (undiscounted forecast)
- forecast/15.2_statement_distribution_summary.csv (statement-based decile patterns)
- payment_profile/decile_payment_profile.pkl (payment behaviors)
- visualisations/09.6_reconstructed_late_payment_rates.csv (seasonal adjustments)
- visualisations/10.6_calibrated_baseline_late_rate.csv (calibrated baseline)

Outputs:
--------
- forecast/15.4_monthly_revenue_forecast.csv
- forecast/15.4_decile_revenue_breakdown.csv
- forecast/15.4_forecast_comparison.csv (compare 15.3 vs 15.4)
- forecast/15.4_revenue_forecast_summary.xlsx
- forecast/15.4_revenue_forecast_visualization.png
- forecast/15.4_decile_contribution_analysis.png
- forecast/15.4_forecast_method_comparison.png

Author: Chris & Team
Date: January 2026
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================
BASE_PATH = Path(r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3")

visualisations_dir = BASE_PATH / "visualisations"
forecast_dir = BASE_PATH / "forecast"
profile_dir = BASE_PATH / "payment_profile"

# Revenue calculation parameters
ANNUAL_INTEREST_RATE = 0.2395
DAILY_INTEREST_RATE = ANNUAL_INTEREST_RATE / 365

CD_TO_DAYS = {
    2: 30, 
    3: 60,
    4: 90,
    5: 120,
    6: 150,
    7: 180,
    8: 210,
    9: 240
}

print("\n" + "="*80)
print("REVENUE ESTIMATION USING SEPARATE FORECASTS")
print("Discounted (11.3) + Undiscounted (11.5) → Revenue Forecast")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"Annual Interest Rate: {ANNUAL_INTEREST_RATE*100:.2f}%")
print(f"Daily Interest Rate: {DAILY_INTEREST_RATE:.6f}")
print("="*80)

# Check required inputs exist
required_files = {
    'discounted_forecast': visualisations_dir / "11.3_forecast_next_15_months.csv",
    'undiscounted_forecast': forecast_dir / "11.5_forecast_undiscounted_next_15_months.csv",
    'decile_dist': forecast_dir / "15.2_statement_distribution_summary.csv",
    'payment_profile': profile_dir / "decile_payment_profile.pkl",
    'seasonal_rates': visualisations_dir / "09.6_reconstructed_late_payment_rates.csv",
    'calibration': visualisations_dir / "10.6_calibrated_baseline_late_rate.csv"
}

missing_files = []
for name, file in required_files.items():
    if not file.exists():
        missing_files.append((name, file))

if missing_files:
    print(f"\n❌ ERROR: Required files not found:")
    for name, file in missing_files:
        print(f"   {name}: {file}")
    if any('11.5' in str(f) for _, f in missing_files):
        print(f"\n   Please run script 11.5 first to generate undiscounted forecast")
    if any('15.2' in str(f) for _, f in missing_files):
        print(f"   Please run script 15.2 first to generate statement-based distributions")
    exit(1)

# ================================================================
# STEP 1: Load Forecast Data (Both Sources)
# ================================================================
print("\n" + "="*80)
print("📁 [Step 1/7] LOADING FORECAST DATA FROM BOTH SOURCES")
print("="*80)

# Load discounted forecast from 11.3 (ONLY take what we need)
forecast_discounted_full = pd.read_csv(visualisations_dir / "11.3_forecast_next_15_months.csv")
forecast_discounted_full['invoice_period'] = pd.to_datetime(forecast_discounted_full['invoice_period'])

# Select ONLY the discounted columns from 11.3
forecast_discounted = forecast_discounted_full[['invoice_period', 'forecast_discounted_price']].copy()

print(f"  ✓ Loaded discounted forecast (11.3): {len(forecast_discounted)} months")
print(f"    Period: {forecast_discounted['invoice_period'].min().strftime('%Y-%m')} to {forecast_discounted['invoice_period'].max().strftime('%Y-%m')}")
print(f"    Total forecasted (discounted): ${forecast_discounted['forecast_discounted_price'].sum():,.2f}")

# Load undiscounted forecast from 11.5
forecast_undiscounted = pd.read_csv(forecast_dir / "11.5_forecast_undiscounted_next_15_months.csv")
forecast_undiscounted['invoice_period'] = pd.to_datetime(forecast_undiscounted['invoice_period'])

print(f"\n  ✓ Loaded undiscounted forecast (11.5): {len(forecast_undiscounted)} months")
print(f"    Period: {forecast_undiscounted['invoice_period'].min().strftime('%Y-%m')} to {forecast_undiscounted['invoice_period'].max().strftime('%Y-%m')}")
print(f"    Total forecasted (undiscounted): ${forecast_undiscounted['forecast_undiscounted_price'].sum():,.2f}")

# Merge forecasts on invoice_period (no suffix conflicts now!)
forecast_df = forecast_discounted.merge(
    forecast_undiscounted[['invoice_period', 'forecast_undiscounted_price', 'forecast_invoice_count']],
    on='invoice_period',
    how='inner'
)

# Calculate discount amount as difference
forecast_df['forecast_discount_amount'] = (
    forecast_df['forecast_undiscounted_price'] - forecast_df['forecast_discounted_price']
)

# Calculate effective discount rate
forecast_df['effective_discount_pct'] = (
    forecast_df['forecast_discount_amount'] / forecast_df['forecast_undiscounted_price'] * 100
)

print(f"\n  ✓ Merged forecasts: {len(forecast_df)} months")
print(f"\n  📊 Combined Forecast Summary:")
print(f"    Total forecasted (discounted from 11.3): ${forecast_df['forecast_discounted_price'].sum():,.2f}")
print(f"    Total forecasted (undiscounted from 11.5): ${forecast_df['forecast_undiscounted_price'].sum():,.2f}")
print(f"    Total discount amount (calculated): ${forecast_df['forecast_discount_amount'].sum():,.2f}")
print(f"    Average discount rate: {forecast_df['effective_discount_pct'].mean():.2f}%")
print(f"    Discount rate range: {forecast_df['effective_discount_pct'].min():.2f}% - {forecast_df['effective_discount_pct'].max():.2f}%")

# Show monthly breakdown
print(f"\n  Monthly Forecast Details:")
for _, row in forecast_df.iterrows():
    print(f"    {row['invoice_period'].strftime('%Y-%m')}: "
          f"Disc=${row['forecast_discounted_price']:>12,.2f}, "
          f"Undisc=${row['forecast_undiscounted_price']:>12,.2f}, "
          f"Discount={row['effective_discount_pct']:>5.2f}%")

# ================================================================
# STEP 2: Load Statement-Based Decile Distribution Patterns
# ================================================================
print("\n" + "="*80)
print("📊 [Step 2/7] LOADING STATEMENT-BASED DECILE DISTRIBUTIONS")
print("="*80)

decile_dist = pd.read_csv(forecast_dir / "15.2_statement_distribution_summary.csv")

print(f"  ✓ Loaded distribution for {len(decile_dist)} deciles (statement-based)")

# Normalize distributions to ensure they sum to exactly 100%
decile_dist['pct_of_total_statements'] = (
    decile_dist['pct_of_total_statements'] / decile_dist['pct_of_total_statements'].sum() * 100
)
decile_dist['pct_of_total_value'] = (
    decile_dist['pct_of_total_value'] / decile_dist['pct_of_total_value'].sum() * 100
)

print(f"\n  Normalized Statement-Based Decile Distribution:")
print(decile_dist[['decile', 'pct_of_total_statements', 'pct_of_total_value', 'avg_statement_value']].to_string(index=False))

# ================================================================
# STEP 3: Load Payment Profile and Calibration
# ================================================================
print("\n" + "="*80)
print("💳 [Step 3/7] LOADING PAYMENT PROFILE AND CALIBRATION")
print("="*80)

# Load payment profile
with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
    decile_profile = pickle.load(f)

n_deciles = decile_profile['metadata']['n_deciles']
print(f"  ✓ Loaded payment profile with {n_deciles} deciles")

# Load calibrated baseline
calibration_df = pd.read_csv(visualisations_dir / "10.6_calibrated_baseline_late_rate.csv")
calibrated_baseline_pct = calibration_df['calibrated_november_baseline_pct'].values[0]
scaling_factor = calibration_df['scaling_factor'].values[0]

print(f"  ✓ Loaded calibration:")
print(f"    Calibrated November baseline: {calibrated_baseline_pct:.2f}%")
print(f"    Scaling factor: {scaling_factor:.4f}")

# Load seasonal rates
seasonal_rates = pd.read_csv(visualisations_dir / "09.6_reconstructed_late_payment_rates.csv")
seasonal_rates['invoice_period'] = pd.to_datetime(seasonal_rates['invoice_period'])

# Get November 2025 baseline for seasonal adjustments
november_2025_row = seasonal_rates[
    (seasonal_rates['invoice_period'].dt.year == 2025) & 
    (seasonal_rates['invoice_period'].dt.month == 11)
]
november_reconstructed_rate = november_2025_row['reconstructed_late_rate_pct'].values[0] / 100

print(f"  ✓ Loaded seasonal rates (November 2025 baseline: {november_reconstructed_rate*100:.2f}%)")

# Calculate seasonal adjustment factors
seasonal_adjustment_factors = {}
for _, row in seasonal_rates.iterrows():
    year_month = (row['invoice_period'].year, row['invoice_period'].month)
    month_rate = row['reconstructed_late_rate_pct'] / 100
    adjustment_factor = month_rate / november_reconstructed_rate
    seasonal_adjustment_factors[year_month] = adjustment_factor

# Extract decile-specific metrics from payment profile
print(f"\n  Extracting decile-specific payment behaviors:")
decile_metrics = []

for decile_num in range(n_deciles):
    decile_key = f'decile_{decile_num}'
    
    if decile_key in decile_profile['deciles']:
        snapshot_late_rate = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
        cd_given_late = decile_profile['deciles'][decile_key]['delinquency_distribution']['cd_given_late']
        
        # Calculate expected days overdue
        if cd_given_late:
            expected_days = sum(CD_TO_DAYS.get(cd, 90) * prob for cd, prob in cd_given_late.items())
        else:
            expected_days = 60
    else:
        snapshot_late_rate = 0.02
        expected_days = 60
    
    # Apply calibration scaling
    calibrated_late_rate = snapshot_late_rate * scaling_factor
    
    decile_metrics.append({
        'decile': decile_num,
        'snapshot_late_rate': snapshot_late_rate,
        'calibrated_late_rate': calibrated_late_rate,
        'expected_days_overdue': expected_days
    })

decile_metrics_df = pd.DataFrame(decile_metrics)
print(decile_metrics_df.to_string(index=False))

# ================================================================
# STEP 4: Calculate Revenue for Each Month
# ================================================================
print("\n" + "="*80)
print("💰 [Step 4/7] CALCULATING REVENUE BY DIRECT VALUE ALLOCATION")
print("   Using Separately Forecasted Undiscounted + Discounted Amounts")
print("="*80)

all_monthly_revenue = []
all_decile_breakdown = []

for idx, forecast_row in forecast_df.iterrows():
    month = forecast_row['invoice_period']
    forecasted_total_discounted = forecast_row['forecast_discounted_price']
    forecasted_total_undiscounted = forecast_row['forecast_undiscounted_price']
    forecasted_discount_amount = forecast_row['forecast_discount_amount']
    effective_discount_pct = forecast_row['effective_discount_pct']
    
    year_month = (month.year, month.month)
    seasonal_factor = seasonal_adjustment_factors.get(year_month, 1.0)
    
    print(f"\n  Processing {month.strftime('%Y-%m')}:")
    print(f"    Forecasted total (discounted): ${forecasted_total_discounted:,.2f}")
    print(f"    Forecasted total (undiscounted): ${forecasted_total_undiscounted:,.2f}")
    print(f"    Forecasted discount amount: ${forecasted_discount_amount:,.2f} ({effective_discount_pct:.2f}%)")
    print(f"    Seasonal adjustment factor: {seasonal_factor:.4f}")
    
    # Initialize accumulators for this month
    month_interest_revenue = 0
    month_retained_discounts = 0
    
    # Process each decile
    for _, decile_row in decile_dist.iterrows():
        decile_num = int(decile_row['decile'])
        value_pct = decile_row['pct_of_total_value'] / 100
        
        # Step 1: Allocate forecasted values to this decile
        decile_value_discounted = forecasted_total_discounted * value_pct
        decile_value_undiscounted = forecasted_total_undiscounted * value_pct
        decile_discount_amount = forecasted_discount_amount * value_pct
        
        # Step 2: Get decile-specific payment behavior
        decile_metric = decile_metrics_df[decile_metrics_df['decile'] == decile_num].iloc[0]
        calibrated_late_rate = decile_metric['calibrated_late_rate']
        expected_days = decile_metric['expected_days_overdue']
        
        # Step 3: Apply seasonal adjustment
        adjusted_late_rate = calibrated_late_rate * seasonal_factor
        adjusted_late_rate = min(adjusted_late_rate, 1.0)  # Cap at 100%
        
        # Step 4: Calculate late payment value
        late_value_discounted = decile_value_discounted * adjusted_late_rate
        late_value_undiscounted = decile_value_undiscounted * adjusted_late_rate
        
        # Step 5: Calculate interest revenue
        # Interest = Late Value × Daily Rate × Days Overdue
        interest_revenue = late_value_discounted * DAILY_INTEREST_RATE * expected_days
        
        # Step 6: Calculate retained discounts (for "no discount" scenario)
        # Only late invoices have discounts retained
        retained_discounts = decile_discount_amount * adjusted_late_rate
        
        # Total revenue for this decile
        total_decile_revenue = interest_revenue + retained_discounts
        
        # Accumulate
        month_interest_revenue += interest_revenue
        month_retained_discounts += retained_discounts
        
        # Store decile-level breakdown
        all_decile_breakdown.append({
            'invoice_period': month,
            'decile': decile_num,
            'allocated_value_discounted': decile_value_discounted,
            'allocated_value_undiscounted': decile_value_undiscounted,
            'allocated_discount_amount': decile_discount_amount,
            'effective_discount_pct': (decile_discount_amount / decile_value_undiscounted * 100) if decile_value_undiscounted > 0 else 0,
            'calibrated_late_rate': calibrated_late_rate,
            'seasonal_factor': seasonal_factor,
            'adjusted_late_rate': adjusted_late_rate,
            'late_value_discounted': late_value_discounted,
            'expected_days_overdue': expected_days,
            'interest_revenue': interest_revenue,
            'retained_discounts': retained_discounts,
            'total_revenue': total_decile_revenue
        })
    
    # Store monthly totals
    total_month_revenue = month_interest_revenue + month_retained_discounts
    
    all_monthly_revenue.append({
        'invoice_period': month,
        'forecasted_total_discounted': forecasted_total_discounted,
        'forecasted_total_undiscounted': forecasted_total_undiscounted,
        'forecasted_discount_amount': forecasted_discount_amount,
        'effective_discount_pct': effective_discount_pct,
        'seasonal_factor': seasonal_factor,
        'interest_revenue': month_interest_revenue,
        'retained_discounts': month_retained_discounts,
        'total_revenue': total_month_revenue
    })
    
    print(f"    Interest revenue: ${month_interest_revenue:,.2f}")
    print(f"    Retained discounts: ${month_retained_discounts:,.2f}")
    print(f"    Total revenue: ${total_month_revenue:,.2f}")

# Convert to DataFrames
monthly_revenue_df = pd.DataFrame(all_monthly_revenue)
decile_breakdown_df = pd.DataFrame(all_decile_breakdown)

# Add cumulative revenue
monthly_revenue_df = monthly_revenue_df.sort_values('invoice_period')
monthly_revenue_df['cumulative_revenue'] = monthly_revenue_df['total_revenue'].cumsum()

print(f"\n  💰 15-Month Revenue Summary:")
print(f"    Total Interest Revenue: ${monthly_revenue_df['interest_revenue'].sum():,.2f}")
print(f"    Total Retained Discounts: ${monthly_revenue_df['retained_discounts'].sum():,.2f}")
print(f"    Total Revenue: ${monthly_revenue_df['total_revenue'].sum():,.2f}")
print(f"    Average Monthly Revenue: ${monthly_revenue_df['total_revenue'].mean():,.2f}")

# ================================================================
# STEP 5: Compare with 15.3 Forecast (if available)
# ================================================================
print("\n" + "="*80)
print("🔍 [Step 5/7] COMPARING WITH 15.3 FORECAST")
print("="*80)

comparison_file_153 = forecast_dir / '15.3_monthly_revenue_forecast.csv'

if comparison_file_153.exists():
    print("  ✓ Found 15.3 forecast - creating comparison")
    
    forecast_153 = pd.read_csv(comparison_file_153)
    forecast_153['invoice_period'] = pd.to_datetime(forecast_153['invoice_period'])
    
    # Merge with current forecast
    comparison_df = monthly_revenue_df[['invoice_period', 'total_revenue']].copy()
    comparison_df.columns = ['invoice_period', 'revenue_15.4']
    
    comparison_df = comparison_df.merge(
        forecast_153[['invoice_period', 'total_revenue']],
        on='invoice_period',
        how='left'
    )
    comparison_df.columns = ['invoice_period', 'revenue_15.4', 'revenue_15.3']
    
    # Calculate differences
    comparison_df['difference'] = comparison_df['revenue_15.4'] - comparison_df['revenue_15.3']
    comparison_df['difference_pct'] = (comparison_df['difference'] / comparison_df['revenue_15.3'] * 100)
    
    print("\n  Monthly Comparison (15.4 vs 15.3):")
    for _, row in comparison_df.iterrows():
        print(f"    {row['invoice_period'].strftime('%Y-%m')}: "
              f"15.4=${row['revenue_15.4']:>10,.2f}, "
              f"15.3=${row['revenue_15.3']:>10,.2f}, "
              f"Diff=${row['difference']:>10,.2f} ({row['difference_pct']:>6.2f}%)")
    
    print(f"\n  Summary Comparison:")
    print(f"    15.4 Total Revenue: ${comparison_df['revenue_15.4'].sum():,.2f}")
    print(f"    15.3 Total Revenue: ${comparison_df['revenue_15.3'].sum():,.2f}")
    print(f"    Total Difference: ${comparison_df['difference'].sum():,.2f} "
          f"({comparison_df['difference'].sum() / comparison_df['revenue_15.3'].sum() * 100:.2f}%)")
    print(f"    Average Monthly Difference: ${comparison_df['difference'].mean():,.2f} "
          f"({comparison_df['difference_pct'].mean():.2f}%)")
    
    # Save comparison
    comparison_output = forecast_dir / '15.4_forecast_comparison.csv'
    comparison_df.to_csv(comparison_output, index=False)
    print(f"\n  ✓ Saved comparison: {comparison_output.name}")
    
else:
    print("  ℹ️  15.3 forecast not found - skipping comparison")
    comparison_df = None

# ================================================================
# STEP 6: Analyze Decile Contributions
# ================================================================
print("\n" + "="*80)
print("📈 [Step 6/7] ANALYZING DECILE CONTRIBUTIONS")
print("="*80)

# Total contribution by decile across all months
decile_total_contribution = decile_breakdown_df.groupby('decile').agg({
    'allocated_value_discounted': 'sum',
    'allocated_value_undiscounted': 'sum',
    'allocated_discount_amount': 'sum',
    'late_value_discounted': 'sum',
    'interest_revenue': 'sum',
    'retained_discounts': 'sum',
    'total_revenue': 'sum'
}).reset_index()

decile_total_contribution['pct_of_total_revenue'] = (
    decile_total_contribution['total_revenue'] / decile_total_contribution['total_revenue'].sum() * 100
)

print("\n  Total Revenue Contribution by Decile (15 months):")
print(decile_total_contribution[['decile', 'total_revenue', 'pct_of_total_revenue', 
                                  'interest_revenue', 'retained_discounts']].to_string(index=False))

# Average metrics by decile
decile_avg_metrics = decile_breakdown_df.groupby('decile').agg({
    'effective_discount_pct': 'mean',
    'adjusted_late_rate': 'mean',
    'expected_days_overdue': 'mean',
    'interest_revenue': 'mean',
    'retained_discounts': 'mean',
    'total_revenue': 'mean'
}).reset_index()

print("\n  Average Monthly Metrics by Decile:")
print(decile_avg_metrics.to_string(index=False))

# ================================================================
# STEP 7: Save Results
# ================================================================
print("\n" + "="*80)
print("💾 [Step 7/7] SAVING RESULTS")
print("="*80)

# Save monthly revenue forecast
revenue_output = forecast_dir / '15.4_monthly_revenue_forecast.csv'
monthly_revenue_df.to_csv(revenue_output, index=False)
print(f"  ✓ Saved: {revenue_output.name}")

# Save decile breakdown
decile_output = forecast_dir / '15.4_decile_revenue_breakdown.csv'
decile_breakdown_df.to_csv(decile_output, index=False)
print(f"  ✓ Saved: {decile_output.name}")

# Save comprehensive Excel file
excel_output = forecast_dir / '15.4_revenue_forecast_summary.xlsx'
with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
    # Monthly revenue sheet
    monthly_revenue_df.to_excel(writer, sheet_name='Monthly_Revenue', index=False)
    
    # Decile breakdown by month
    decile_breakdown_df.to_excel(writer, sheet_name='Decile_Monthly_Breakdown', index=False)
    
    # Total decile contribution
    decile_total_contribution.to_excel(writer, sheet_name='Decile_Total_Contribution', index=False)
    
    # Average decile metrics
    decile_avg_metrics.to_excel(writer, sheet_name='Decile_Avg_Metrics', index=False)
    
    # Comparison with 15.3 (if available)
    if comparison_df is not None:
        comparison_df.to_excel(writer, sheet_name='Comparison_15.3_vs_15.4', index=False)
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Method',
            'Total 15-Month Revenue',
            'Interest Revenue',
            'Retained Discounts',
            'Average Monthly Revenue',
            'Forecasted Total Discounted',
            'Forecasted Total Undiscounted',
            'Total Discount Amount',
            'Average Discount Rate',
            'Average Late Probability',
            'Average Days Overdue',
            'Calibrated November Baseline',
            'Scaling Factor',
            'Annual Interest Rate',
            'Daily Interest Rate'
        ],
        'Value': [
            'Separate Forecasts (11.3 + 11.5)',
            f"${monthly_revenue_df['total_revenue'].sum():,.2f}",
            f"${monthly_revenue_df['interest_revenue'].sum():,.2f}",
            f"${monthly_revenue_df['retained_discounts'].sum():,.2f}",
            f"${monthly_revenue_df['total_revenue'].mean():,.2f}",
            f"${monthly_revenue_df['forecasted_total_discounted'].sum():,.2f}",
            f"${monthly_revenue_df['forecasted_total_undiscounted'].sum():,.2f}",
            f"${monthly_revenue_df['forecasted_discount_amount'].sum():,.2f}",
            f"{monthly_revenue_df['effective_discount_pct'].mean():.2f}%",
            f"{decile_breakdown_df['adjusted_late_rate'].mean()*100:.2f}%",
            f"{decile_breakdown_df['expected_days_overdue'].mean():.1f} days",
            f"{calibrated_baseline_pct:.2f}%",
            f"{scaling_factor:.4f}",
            f"{ANNUAL_INTEREST_RATE*100:.2f}%",
            f"{DAILY_INTEREST_RATE:.6f}"
        ]
    })
    summary_stats.to_excel(writer, sheet_name='Summary', index=False)

print(f"  ✓ Saved: {excel_output.name}")

# ================================================================
# VISUALIZATIONS
# ================================================================
print("\n" + "="*80)
print("🎨 CREATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Revenue Forecast Overview
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Monthly Revenue Breakdown
ax1 = axes[0, 0]
x = np.arange(len(monthly_revenue_df))
width = 0.35
ax1.bar(x - width/2, monthly_revenue_df['interest_revenue'], width,
        label='Interest Revenue', color='#4472C4', alpha=0.7, edgecolor='black')
ax1.bar(x + width/2, monthly_revenue_df['retained_discounts'], width,
        label='Retained Discounts', color='#FFC000', alpha=0.7, edgecolor='black')
ax1.set_title('Monthly Revenue Breakdown (15.4 - Separate Forecasts)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Revenue ($)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(x)
ax1.set_xticklabels(monthly_revenue_df['invoice_period'].dt.strftime('%Y-%m'), rotation=45)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: Cumulative Revenue
ax2 = axes[0, 1]
ax2.plot(monthly_revenue_df['invoice_period'], monthly_revenue_df['cumulative_revenue'],
         marker='o', linewidth=2.5, markersize=8, color='#70AD47')
ax2.fill_between(monthly_revenue_df['invoice_period'], 0, monthly_revenue_df['cumulative_revenue'],
                 alpha=0.3, color='#70AD47')
ax2.set_title('Cumulative Revenue Forecast', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Cumulative Revenue ($)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis='x', rotation=45)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add final value annotation
final_revenue = monthly_revenue_df['cumulative_revenue'].iloc[-1]
ax2.text(monthly_revenue_df['invoice_period'].iloc[-1], final_revenue,
         f'  ${final_revenue:,.0f}', fontsize=11, fontweight='bold',
         va='center', ha='left')

# Plot 3: Effective Discount Rate Over Time
ax3 = axes[1, 0]
ax3.plot(monthly_revenue_df['invoice_period'], monthly_revenue_df['effective_discount_pct'],
         marker='s', linewidth=2, markersize=8, color='coral')
ax3.axhline(y=monthly_revenue_df['effective_discount_pct'].mean(), 
            color='red', linestyle='--', linewidth=1.5, alpha=0.5, 
            label=f"Mean: {monthly_revenue_df['effective_discount_pct'].mean():.2f}%")
ax3.set_title('Effective Discount Rate by Month\n(From Separate Forecasts)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12)
ax3.set_ylabel('Discount Rate (%)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Revenue Components Pie Chart
ax4 = axes[1, 1]
total_interest = monthly_revenue_df['interest_revenue'].sum()
total_discounts = monthly_revenue_df['retained_discounts'].sum()
sizes = [total_interest, total_discounts]
labels = ['Interest Revenue', 'Retained Discounts']
colors = ['#4472C4', '#FFC000']
explode = (0.05, 0.05)

ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax4.set_title(f'15-Month Revenue Composition\nTotal: ${total_interest + total_discounts:,.2f}',
              fontsize=14, fontweight='bold')

plt.tight_layout()
revenue_viz_output = forecast_dir / '15.4_revenue_forecast_visualization.png'
plt.savefig(revenue_viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {revenue_viz_output.name}")
plt.close()

# Visualization 2: Decile Contribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Total Revenue by Decile
ax1 = axes[0, 0]
ax1.bar(decile_total_contribution['decile'], decile_total_contribution['total_revenue'],
        color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_title('Total Revenue by Decile (15 Months)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Decile', fontsize=12)
ax1.set_ylabel('Total Revenue ($)', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
for i, row in decile_total_contribution.iterrows():
    ax1.text(row['decile'], row['total_revenue'], 
             f"{row['pct_of_total_revenue']:.1f}%",
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 2: Revenue Breakdown by Decile
ax2 = axes[0, 1]
x = decile_total_contribution['decile']
width = 0.35
ax2.bar(x - width/2, decile_total_contribution['interest_revenue'], width,
        label='Interest', color='#4472C4', alpha=0.7, edgecolor='black')
ax2.bar(x + width/2, decile_total_contribution['retained_discounts'], width,
        label='Discounts', color='#FFC000', alpha=0.7, edgecolor='black')
ax2.set_title('Revenue Components by Decile', fontsize=14, fontweight='bold')
ax2.set_xlabel('Decile', fontsize=12)
ax2.set_ylabel('Revenue ($)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 3: Average Discount Rate by Decile
ax3 = axes[1, 0]
ax3.bar(decile_avg_metrics['decile'], decile_avg_metrics['effective_discount_pct'],
        color='coral', edgecolor='black', alpha=0.7)
ax3.set_title('Average Discount Rate by Decile', fontsize=14, fontweight='bold')
ax3.set_xlabel('Decile', fontsize=12)
ax3.set_ylabel('Discount Rate (%)', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')
for i, row in decile_avg_metrics.iterrows():
    ax3.text(row['decile'], row['effective_discount_pct'], 
             f"{row['effective_discount_pct']:.1f}%",
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 4: Average Days Overdue by Decile
ax4 = axes[1, 1]
ax4.bar(decile_avg_metrics['decile'], decile_avg_metrics['expected_days_overdue'],
        color='#70AD47', edgecolor='black', alpha=0.7)
ax4.set_title('Average Days Overdue by Decile', fontsize=14, fontweight='bold')
ax4.set_xlabel('Decile', fontsize=12)
ax4.set_ylabel('Days Overdue', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')
for i, row in decile_avg_metrics.iterrows():
    ax4.text(row['decile'], row['expected_days_overdue'], 
             f"{row['expected_days_overdue']:.0f}d",
             ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
decile_viz_output = forecast_dir / '15.4_decile_contribution_analysis.png'
plt.savefig(decile_viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {decile_viz_output.name}")
plt.close()

# Visualization 3: Comparison with 15.3 (if available)
if comparison_df is not None:
    print("  Creating comparison visualization with 15.3...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Revenue Comparison Line Chart
    ax1 = axes[0, 0]
    ax1.plot(comparison_df['invoice_period'], comparison_df['revenue_15.4'],
             marker='o', linewidth=2.5, markersize=8, label='15.4 (Separate Forecasts)', color='#4472C4')
    ax1.plot(comparison_df['invoice_period'], comparison_df['revenue_15.3'],
             marker='s', linewidth=2.5, markersize=8, label='15.3 (Multiplier)', color='#ED7D31')
    ax1.set_title('Revenue Forecast Comparison: 15.4 vs 15.3', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Revenue ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Difference Over Time
    ax2 = axes[0, 1]
    colors = ['green' if x >= 0 else 'red' for x in comparison_df['difference']]
    ax2.bar(comparison_df['invoice_period'], comparison_df['difference'],
            color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Difference: 15.4 - 15.3', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Difference ($)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 3: Percentage Difference
    ax3 = axes[1, 0]
    colors = ['green' if x >= 0 else 'red' for x in comparison_df['difference_pct']]
    ax3.bar(comparison_df['invoice_period'], comparison_df['difference_pct'],
            color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Percentage Difference: (15.4 - 15.3) / 15.3 × 100%', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel('Difference (%)', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    COMPARISON SUMMARY: 15.4 vs 15.3
    
    15.4 Method: Separate Forecasts (11.3 + 11.5)
    15.3 Method: Multiplier-Based (11.3 only)
    
    Total Revenue:
      15.4: ${comparison_df['revenue_15.4'].sum():,.2f}
      15.3: ${comparison_df['revenue_15.3'].sum():,.2f}
      Difference: ${comparison_df['difference'].sum():,.2f}
      Difference %: {comparison_df['difference'].sum() / comparison_df['revenue_15.3'].sum() * 100:.2f}%
    
    Average Monthly Revenue:
      15.4: ${comparison_df['revenue_15.4'].mean():,.2f}
      15.3: ${comparison_df['revenue_15.3'].mean():,.2f}
      Avg Difference: ${comparison_df['difference'].mean():,.2f}
    
    Interpretation:
      Positive difference → 15.4 predicts higher revenue
      Negative difference → 15.3 predicts higher revenue
    
    The difference indicates the impact of using
    separately forecasted undiscounted amounts
    vs. constant multiplier approach.
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    comparison_viz_output = forecast_dir / '15.4_forecast_method_comparison.png'
    plt.savefig(comparison_viz_output, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {comparison_viz_output.name}")
    plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ REVENUE FORECAST COMPLETE (SEPARATE FORECASTS METHOD)!")
print("="*80)

print(f"\n📊 Methodology:")
print(f"  ✓ Uses SEPARATE forecasts for discounted (11.3) and undiscounted (11.5)")
print(f"  ✓ Calculates discount amount as difference (not constant multiplier)")
print(f"  ✓ Statement-based decile distributions from script 15.2")
print(f"  ✓ Decile-specific late payment rates with calibration")
print(f"  ✓ Seasonal adjustments applied")

print(f"\n💰 Revenue Forecast Summary:")
print(f"  Total 15-Month Revenue: ${monthly_revenue_df['total_revenue'].sum():,.2f}")
print(f"    Interest Revenue: ${monthly_revenue_df['interest_revenue'].sum():,.2f} "
      f"({monthly_revenue_df['interest_revenue'].sum() / monthly_revenue_df['total_revenue'].sum() * 100:.1f}%)")
print(f"    Retained Discounts: ${monthly_revenue_df['retained_discounts'].sum():,.2f} "
      f"({monthly_revenue_df['retained_discounts'].sum() / monthly_revenue_df['total_revenue'].sum() * 100:.1f}%)")
print(f"  Average Monthly Revenue: ${monthly_revenue_df['total_revenue'].mean():,.2f}")
print(f"  Revenue Range: ${monthly_revenue_df['total_revenue'].min():,.2f} - ${monthly_revenue_df['total_revenue'].max():,.2f}")

print(f"\n📈 Key Metrics:")
print(f"  Total forecasted (discounted): ${monthly_revenue_df['forecasted_total_discounted'].sum():,.2f}")
print(f"  Total forecasted (undiscounted): ${monthly_revenue_df['forecasted_total_undiscounted'].sum():,.2f}")
print(f"  Total discount amount: ${monthly_revenue_df['forecasted_discount_amount'].sum():,.2f}")
print(f"  Average discount rate: {monthly_revenue_df['effective_discount_pct'].mean():.2f}%")
print(f"  Discount rate range: {monthly_revenue_df['effective_discount_pct'].min():.2f}% - {monthly_revenue_df['effective_discount_pct'].max():.2f}%")
print(f"  Average late payment rate: {decile_breakdown_df['adjusted_late_rate'].mean()*100:.2f}%")
print(f"  Average days overdue: {decile_breakdown_df['expected_days_overdue'].mean():.1f} days")
print(f"  Revenue as % of discounted value: {monthly_revenue_df['total_revenue'].sum() / monthly_revenue_df['forecasted_total_discounted'].sum() * 100:.2f}%")

if comparison_df is not None:
    print(f"\n🔍 Comparison with 15.3:")
    print(f"  15.4 Total Revenue: ${comparison_df['revenue_15.4'].sum():,.2f}")
    print(f"  15.3 Total Revenue: ${comparison_df['revenue_15.3'].sum():,.2f}")
    print(f"  Difference: ${comparison_df['difference'].sum():,.2f} "
          f"({comparison_df['difference'].sum() / comparison_df['revenue_15.3'].sum() * 100:+.2f}%)")
    print(f"  Interpretation: {'15.4 predicts HIGHER revenue' if comparison_df['difference'].sum() > 0 else '15.3 predicts HIGHER revenue'}")

print(f"\n🎯 Top Revenue-Contributing Deciles:")
top_deciles = decile_total_contribution.nlargest(3, 'total_revenue')
for i, row in top_deciles.iterrows():
    print(f"  Decile {int(row['decile'])}: ${row['total_revenue']:,.2f} ({row['pct_of_total_revenue']:.1f}%)")

print(f"\n📁 Output Files:")
print(f"  All files saved to: {forecast_dir}")
print("  • 15.4_monthly_revenue_forecast.csv - Monthly revenue totals")
print("  • 15.4_decile_revenue_breakdown.csv - Detailed decile breakdown by month")
print("  • 15.4_forecast_comparison.csv - Comparison with 15.3 (if available)")
print("  • 15.4_revenue_forecast_summary.xlsx - Comprehensive workbook")
print("  • 15.4_revenue_forecast_visualization.png - Revenue analysis")
print("  • 15.4_decile_contribution_analysis.png - Decile contribution breakdown")
if comparison_df is not None:
    print("  • 15.4_forecast_method_comparison.png - 15.3 vs 15.4 comparison")

print("\n" + "="*80)
print("ADVANTAGES OF 15.4 OVER 15.3:")
print("  ✓ Uses independently forecasted undiscounted amounts (11.5)")
print("  ✓ Captures time-varying discount patterns")
print("  ✓ No assumption of constant discount multiplier")
print("  ✓ More accurate when discount behavior changes over time")
print("  ✓ Provides validation against multiplier approach")
print("="*80)

# ================================================================
# ADDITIONAL ANALYSIS: Last 12 Months Revenue Components
# ================================================================
print("\n" + "="*80)
print("📊 ADDITIONAL ANALYSIS: LAST 12 MONTHS BREAKDOWN")
print("="*80)

# Get last 12 months of data
last_12_months = monthly_revenue_df.tail(12).copy()

print(f"\n  Analyzing last 12 months:")
print(f"  Period: {last_12_months['invoice_period'].min().strftime('%Y-%m')} to {last_12_months['invoice_period'].max().strftime('%Y-%m')}")

# Calculate interest on DISCOUNTED amounts only
total_interest_on_discounted = last_12_months['interest_revenue'].sum()

# Calculate retained discounts
total_retained_discounts = last_12_months['retained_discounts'].sum()

# Calculate what interest would be on UNDISCOUNTED amounts
# We need to recalculate using undiscounted late values
# Get last 12 months from decile breakdown for detailed calculation
last_12_decile_breakdown = decile_breakdown_df[
    decile_breakdown_df['invoice_period'].isin(last_12_months['invoice_period'])
].copy()

# Calculate late value on undiscounted amounts
last_12_decile_breakdown['late_value_undiscounted'] = (
    last_12_decile_breakdown['allocated_value_undiscounted'] * 
    last_12_decile_breakdown['adjusted_late_rate']
)

# Calculate interest on undiscounted late values
last_12_decile_breakdown['interest_on_undiscounted'] = (
    last_12_decile_breakdown['late_value_undiscounted'] * 
    DAILY_INTEREST_RATE * 
    last_12_decile_breakdown['expected_days_overdue']
)

# Aggregate
total_interest_on_undiscounted = last_12_decile_breakdown['interest_on_undiscounted'].sum()

# Also calculate total late values for context
total_late_value_discounted = last_12_decile_breakdown['late_value_discounted'].sum()
total_late_value_undiscounted = last_12_decile_breakdown['late_value_undiscounted'].sum()

print(f"\n  💰 Last 12 Months Revenue Components:")
print(f"  " + "-"*76)
print(f"  {'Component':<40} {'Amount':>15} {'% of Total':>15}")
print(f"  " + "-"*76)

total_all_components = total_interest_on_discounted + total_interest_on_undiscounted + total_retained_discounts

print(f"  {'Interest on Discounted Amounts':<40} ${total_interest_on_discounted:>14,.2f} {total_interest_on_discounted/total_all_components*100:>14.2f}%")
print(f"  {'Interest on Undiscounted Amounts':<40} ${total_interest_on_undiscounted:>14,.2f} {total_interest_on_undiscounted/total_all_components*100:>14.2f}%")
print(f"  {'Retained Discounts':<40} ${total_retained_discounts:>14,.2f} {total_retained_discounts/total_all_components*100:>14.2f}%")
print(f"  " + "-"*76)
print(f"  {'TOTAL':<40} ${total_all_components:>14,.2f} {100.0:>14.2f}%")
print(f"  " + "="*76)

print(f"\n  📋 Supporting Metrics (Last 12 Months):")
print(f"    Total forecasted (discounted): ${last_12_months['forecasted_total_discounted'].sum():,.2f}")
print(f"    Total forecasted (undiscounted): ${last_12_months['forecasted_total_undiscounted'].sum():,.2f}")
print(f"    Total discount amount: ${last_12_months['forecasted_discount_amount'].sum():,.2f}")
print(f"    Total late value (discounted): ${total_late_value_discounted:,.2f}")
print(f"    Total late value (undiscounted): ${total_late_value_undiscounted:,.2f}")
print(f"    Average late payment rate: {last_12_decile_breakdown['adjusted_late_rate'].mean()*100:.2f}%")
print(f"    Average days overdue: {last_12_decile_breakdown['expected_days_overdue'].mean():.1f} days")

# Create visualization
print("\n  Creating last 12 months analysis visualization...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Revenue components pie chart
ax1 = axes[0]
components = [total_interest_on_discounted, total_interest_on_undiscounted, total_retained_discounts]
labels = ['Interest on\nDiscounted', 'Interest on\nUndiscounted', 'Retained\nDiscounts']
colors = ['#4472C4', '#70AD47', '#FFC000']
explode = (0.05, 0.05, 0.05)

ax1.pie(components, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title(f'Last 12 Months Revenue Components\nTotal: ${sum(components):,.2f}',
              fontsize=14, fontweight='bold')

# Plot 2: Bar chart comparison
ax2 = axes[1]
x_pos = np.arange(len(labels))
bars = ax2.bar(x_pos, components, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax2.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
ax2.set_title('Revenue Components Comparison', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add value labels on bars
for bar, value in zip(bars, components):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'${value:,.0f}\n({value/sum(components)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
last12_viz_output = forecast_dir / '15.4_last_12_months_analysis.png'
plt.savefig(last12_viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {last12_viz_output.name}")
plt.close()

# Save to CSV
last_12_summary = pd.DataFrame({
    'Component': [
        'Interest on Discounted Amounts',
        'Interest on Undiscounted Amounts', 
        'Retained Discounts',
        'TOTAL'
    ],
    'Amount': [
        total_interest_on_discounted,
        total_interest_on_undiscounted,
        total_retained_discounts,
        total_all_components
    ],
    'Percentage': [
        total_interest_on_discounted/total_all_components*100,
        total_interest_on_undiscounted/total_all_components*100,
        total_retained_discounts/total_all_components*100,
        100.0
    ]
})

# ================================================================
# CUMULATIVE REVENUE VISUALIZATION - Last 12 Months Components
# ================================================================
print("\n  Creating cumulative revenue graph...")

# Calculate cumulative values for each component over last 12 months
last_12_months_sorted = last_12_months.sort_values('invoice_period').copy()

# Cumulative interest on discounted
last_12_months_sorted['cumulative_interest_discounted'] = last_12_months_sorted['interest_revenue'].cumsum()

# For cumulative interest on undiscounted, group decile breakdown by month
last_12_decile_monthly = last_12_decile_breakdown.groupby('invoice_period').agg({
    'interest_on_undiscounted': 'sum'
}).reset_index().sort_values('invoice_period')

last_12_decile_monthly['cumulative_interest_undiscounted'] = last_12_decile_monthly['interest_on_undiscounted'].cumsum()

# Merge with main dataframe
last_12_months_sorted = last_12_months_sorted.merge(
    last_12_decile_monthly[['invoice_period', 'cumulative_interest_undiscounted']],
    on='invoice_period',
    how='left'
)

# Cumulative retained discounts
last_12_months_sorted['cumulative_retained_discounts'] = last_12_months_sorted['retained_discounts'].cumsum()

# Combined: Interest on Undiscounted + Retained Discounts
last_12_months_sorted['cumulative_undiscounted_plus_retained'] = (
    last_12_months_sorted['cumulative_interest_undiscounted'] + 
    last_12_months_sorted['cumulative_retained_discounts']
)

# Create cumulative visualization (same style as 15.3)
fig, ax = plt.subplots(figsize=(14, 8))

# Line 1: Interest on Discounted
ax.plot(last_12_months_sorted['invoice_period'], 
        last_12_months_sorted['cumulative_interest_discounted'],
        marker='o', linewidth=2.5, markersize=8, 
        label='Interest on Discounted Amounts', color='#4472C4')
ax.fill_between(last_12_months_sorted['invoice_period'], 
                0, last_12_months_sorted['cumulative_interest_discounted'],
                alpha=0.3, color='#4472C4')

# Line 2: Interest on Undiscounted + Retained Discounts
ax.plot(last_12_months_sorted['invoice_period'], 
        last_12_months_sorted['cumulative_undiscounted_plus_retained'],
        marker='s', linewidth=2.5, markersize=8, 
        label='Interest on Undiscounted + Retained Discounts', color='#70AD47')
ax.fill_between(last_12_months_sorted['invoice_period'], 
                0, last_12_months_sorted['cumulative_undiscounted_plus_retained'],
                alpha=0.3, color='#70AD47')

# Formatting
ax.set_title('Cumulative Revenue - Last 12 Months\nInterest on Discounted vs (Interest on Undiscounted + Retained Discounts)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Cumulative Revenue ($)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add final value annotations
final_month = last_12_months_sorted['invoice_period'].iloc[-1]
final_interest_disc = last_12_months_sorted['cumulative_interest_discounted'].iloc[-1]
final_combined = last_12_months_sorted['cumulative_undiscounted_plus_retained'].iloc[-1]

ax.text(final_month, final_interest_disc, 
        f'  ${final_interest_disc:,.0f}', 
        fontsize=11, fontweight='bold', va='center', ha='left')

ax.text(final_month, final_combined, 
        f'  ${final_combined:,.0f}', 
        fontsize=11, fontweight='bold', va='center', ha='left')

plt.tight_layout()
cumulative_viz_output = forecast_dir / '15.4_cumulative_revenue_last_12_months.png'
plt.savefig(cumulative_viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {cumulative_viz_output.name}")
plt.close()

print(f"\n  📈 Cumulative Revenue (Last 12 Months):")
print(f"    Interest on Discounted: ${final_interest_disc:,.2f}")
print(f"    Interest on Undiscounted + Retained Discounts: ${final_combined:,.2f}")
print(f"    TOTAL COMBINED: ${final_interest_disc + final_combined:,.2f}")

last12_csv_output = forecast_dir / '15.4_last_12_months_summary.csv'
last_12_summary.to_csv(last12_csv_output, index=False)
print(f"  ✓ Saved: {last12_csv_output.name}")

print("\n" + "="*80)