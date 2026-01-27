'''
15.3_estimated_forecast_revenue_with_deciles_bundled - Revenue Estimation from Forecasted Totals (Statement-Based)

This script takes forecasted monthly totals and estimates revenue using the
statement-level decile distributions from script 15.2. Unlike 15.1 which worked
with individual invoices, this uses bundled statements as the unit of analysis.

The simplified approach:
1. Allocate forecasted monthly total across deciles by value %
2. For each decile, calculate late payment value using decile-specific late rate
3. Calculate interest revenue using late value × daily rate × expected days overdue
4. Calculate retained discounts for "no discount" scenario
5. Apply seasonal adjustments to late payment rates

No synthetic invoices needed - direct value-based calculation.

Inputs:
-------
- visualisations/11.3_forecast_next_15_months.csv (forecasted totals)
- forecast/15.2_statement_distribution_summary.csv (statement-based decile patterns)
- payment_profile/decile_payment_profile.pkl (payment behaviors)
- visualisations/09.6_reconstructed_late_payment_rates.csv (seasonal adjustments)
- visualisations/10.6_calibrated_baseline_late_rate.csv (calibrated baseline)

Outputs:
--------
- forecast/15.3_monthly_revenue_forecast.csv
- forecast/15.3_decile_revenue_breakdown.csv
- forecast/15.3_revenue_forecast_summary.xlsx
- forecast/15.3_revenue_forecast_visualization.png
- forecast/15.3_decile_contribution_analysis.png

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
print("REVENUE ESTIMATION FROM FORECASTED TOTALS (STATEMENT-BASED)")
print("Direct Value Allocation - No Synthetic Invoices Required")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"Annual Interest Rate: {ANNUAL_INTEREST_RATE*100:.2f}%")
print(f"Daily Interest Rate: {DAILY_INTEREST_RATE:.6f}")
print("="*80)

# Check required inputs exist
required_files = {
    'forecast': visualisations_dir / "11.3_forecast_next_15_months.csv",
    'decile_dist': forecast_dir / "15.2_statement_distribution_summary.csv",
    'payment_profile': profile_dir / "decile_payment_profile.pkl",
    'seasonal_rates': visualisations_dir / "09.6_reconstructed_late_payment_rates.csv",
    'calibration': visualisations_dir / "10.6_calibrated_baseline_late_rate.csv"
}

for name, file in required_files.items():
    if not file.exists():
        print(f"\n❌ ERROR: Required file not found: {name}")
        print(f"   Expected: {file}")
        if name == 'decile_dist':
            print(f"\n   Please run script 15.2 first to generate statement-based distributions")
        exit(1)

# ================================================================
# STEP 1: Load Forecast Data
# ================================================================
print("\n" + "="*80)
print("📁 [Step 1/6] LOADING FORECAST DATA")
print("="*80)

forecast_df = pd.read_csv(visualisations_dir / "11.3_forecast_next_15_months.csv")
forecast_df['invoice_period'] = pd.to_datetime(forecast_df['invoice_period'])

print(f"  ✓ Loaded {len(forecast_df)} months of forecast")
print(f"  Forecast period: {forecast_df['invoice_period'].min().strftime('%Y-%m')} to {forecast_df['invoice_period'].max().strftime('%Y-%m')}")
print(f"\n  Forecast Summary:")
print(f"    Total forecasted revenue: ${forecast_df['forecast_discounted_price'].sum():,.2f}")
print(f"    Total forecasted invoices: {forecast_df['forecast_invoice_count'].sum():,}")
print(f"    Avg monthly revenue: ${forecast_df['forecast_discounted_price'].mean():,.2f}")
print(f"    Avg monthly invoices: {forecast_df['forecast_invoice_count'].mean():.0f}")

# ================================================================
# STEP 2: Load Statement-Based Decile Distribution Patterns
# ================================================================
print("\n" + "="*80)
print("📊 [Step 2/6] LOADING STATEMENT-BASED DECILE DISTRIBUTIONS")
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
print("💳 [Step 3/6] LOADING PAYMENT PROFILE AND CALIBRATION")
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
print("💰 [Step 4/6] CALCULATING REVENUE BY DIRECT VALUE ALLOCATION")
print("="*80)

all_monthly_revenue = []
all_decile_breakdown = []

for idx, forecast_row in forecast_df.iterrows():
    month = forecast_row['invoice_period']
    forecasted_total_discounted = forecast_row['forecast_discounted_price']
    forecasted_total_undiscounted = forecast_row['forecast_undiscounted_price']
    forecasted_discount_amount = forecast_row['forecast_discount_amount']
    
    year_month = (month.year, month.month)
    seasonal_factor = seasonal_adjustment_factors.get(year_month, 1.0)
    
    print(f"\n  Processing {month.strftime('%Y-%m')}:")
    print(f"    Forecasted total (discounted): ${forecasted_total_discounted:,.2f}")
    print(f"    Seasonal adjustment factor: {seasonal_factor:.4f}")
    
    # Initialize accumulators for this month
    month_interest_revenue = 0
    month_retained_discounts = 0
    
    # Process each decile
    for _, decile_row in decile_dist.iterrows():
        decile_num = int(decile_row['decile'])
        value_pct = decile_row['pct_of_total_value'] / 100
        
        # Step 1: Allocate forecasted value to this decile
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
# STEP 5: Analyze Decile Contributions
# ================================================================
print("\n" + "="*80)
print("📈 [Step 5/6] ANALYZING DECILE CONTRIBUTIONS")
print("="*80)

# Total contribution by decile across all months
decile_total_contribution = decile_breakdown_df.groupby('decile').agg({
    'allocated_value_discounted': 'sum',
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
    'adjusted_late_rate': 'mean',
    'expected_days_overdue': 'mean',
    'interest_revenue': 'mean',
    'retained_discounts': 'mean',
    'total_revenue': 'mean'
}).reset_index()

print("\n  Average Monthly Metrics by Decile:")
print(decile_avg_metrics.to_string(index=False))

# ================================================================
# STEP 6: Save Results
# ================================================================
print("\n" + "="*80)
print("💾 [Step 6/6] SAVING RESULTS")
print("="*80)

# Save monthly revenue forecast
revenue_output = forecast_dir / '15.3_monthly_revenue_forecast.csv'
monthly_revenue_df.to_csv(revenue_output, index=False)
print(f"  ✓ Saved: {revenue_output.name}")

# Save decile breakdown
decile_output = forecast_dir / '15.3_decile_revenue_breakdown.csv'
decile_breakdown_df.to_csv(decile_output, index=False)
print(f"  ✓ Saved: {decile_output.name}")

# Save comprehensive Excel file
excel_output = forecast_dir / '15.3_revenue_forecast_summary.xlsx'
with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
    # Monthly revenue sheet
    monthly_revenue_df.to_excel(writer, sheet_name='Monthly_Revenue', index=False)
    
    # Decile breakdown by month
    decile_breakdown_df.to_excel(writer, sheet_name='Decile_Monthly_Breakdown', index=False)
    
    # Total decile contribution
    decile_total_contribution.to_excel(writer, sheet_name='Decile_Total_Contribution', index=False)
    
    # Average decile metrics
    decile_avg_metrics.to_excel(writer, sheet_name='Decile_Avg_Metrics', index=False)
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total 15-Month Revenue',
            'Interest Revenue',
            'Retained Discounts',
            'Average Monthly Revenue',
            'Forecasted Total Value',
            'Average Late Probability',
            'Average Days Overdue',
            'Calibrated November Baseline',
            'Scaling Factor',
            'Annual Interest Rate',
            'Daily Interest Rate'
        ],
        'Value': [
            f"${monthly_revenue_df['total_revenue'].sum():,.2f}",
            f"${monthly_revenue_df['interest_revenue'].sum():,.2f}",
            f"${monthly_revenue_df['retained_discounts'].sum():,.2f}",
            f"${monthly_revenue_df['total_revenue'].mean():,.2f}",
            f"${monthly_revenue_df['forecasted_total_discounted'].sum():,.2f}",
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
ax1.set_title('Monthly Revenue Breakdown', fontsize=14, fontweight='bold')
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

# Plot 3: Seasonal Adjustment Impact
ax3 = axes[1, 0]
ax3.plot(monthly_revenue_df['invoice_period'], monthly_revenue_df['seasonal_factor'],
         marker='s', linewidth=2, markersize=8, color='coral')
ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Baseline')
ax3.set_title('Seasonal Adjustment Factor by Month', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12)
ax3.set_ylabel('Seasonal Factor', fontsize=12)
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
revenue_viz_output = forecast_dir / '15.3_revenue_forecast_visualization.png'
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

# Plot 3: Average Late Rate by Decile
ax3 = axes[1, 0]
ax3.bar(decile_avg_metrics['decile'], decile_avg_metrics['adjusted_late_rate'] * 100,
        color='coral', edgecolor='black', alpha=0.7)
ax3.set_title('Average Late Payment Rate by Decile', fontsize=14, fontweight='bold')
ax3.set_xlabel('Decile', fontsize=12)
ax3.set_ylabel('Late Payment Rate (%)', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')
for i, row in decile_avg_metrics.iterrows():
    ax3.text(row['decile'], row['adjusted_late_rate'] * 100, 
             f"{row['adjusted_late_rate']*100:.1f}%",
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
decile_viz_output = forecast_dir / '15.3_decile_contribution_analysis.png'
plt.savefig(decile_viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {decile_viz_output.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ REVENUE FORECAST COMPLETE (STATEMENT-BASED)!")
print("="*80)

print(f"\n📊 Methodology:")
print(f"  ✓ Direct value allocation (no synthetic invoices)")
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
print(f"  Average late payment rate (adjusted): {decile_breakdown_df['adjusted_late_rate'].mean()*100:.2f}%")
print(f"  Average days overdue (when late): {decile_breakdown_df['expected_days_overdue'].mean():.1f} days")
print(f"  Total forecasted value: ${monthly_revenue_df['forecasted_total_discounted'].sum():,.2f}")
print(f"  Revenue as % of forecasted value: {monthly_revenue_df['total_revenue'].sum() / monthly_revenue_df['forecasted_total_discounted'].sum() * 100:.2f}%")

print(f"\n🎯 Top Revenue-Contributing Deciles:")
top_deciles = decile_total_contribution.nlargest(3, 'total_revenue')
for i, row in top_deciles.iterrows():
    print(f"  Decile {int(row['decile'])}: ${row['total_revenue']:,.2f} ({row['pct_of_total_revenue']:.1f}%)")

print(f"\n📁 Output Files:")
print(f"  All files saved to: {forecast_dir}")
print("  • 15.3_monthly_revenue_forecast.csv - Monthly revenue totals")
print("  • 15.3_decile_revenue_breakdown.csv - Detailed decile breakdown by month")
print("  • 15.3_revenue_forecast_summary.xlsx - Comprehensive workbook")
print("  • 15.3_revenue_forecast_visualization.png - Revenue analysis")
print("  • 15.3_decile_contribution_analysis.png - Decile contribution breakdown")

print("\n" + "="*80)
print("ADVANTAGES OF THIS APPROACH:")
print("  ✓ No synthetic invoice creation needed")
print("  ✓ Direct value-based calculation")
print("  ✓ Uses realistic statement-based distributions")
print("  ✓ Computationally efficient")
print("  ✓ Easy to understand and validate")
print("="*80)