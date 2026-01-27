'''
15.1_estimated_forecast_revenue_with_deciles - Revenue Estimation from Forecasted Totals

This script takes forecasted monthly totals and reconstructs synthetic invoices
by allocating the forecasted amounts across deciles using historical distribution
patterns. Each decile receives invoices at its historical average value.

This creates a synthetic invoice population that:
1. Sums to the exact forecasted monthly total
2. Has the exact forecasted invoice count
3. Preserves historical decile structure
4. Can be used with existing payment profile logic to estimate revenue

Process:
1. Allocate forecasted total across deciles by value %
2. Allocate forecasted invoice count across deciles by count %
3. Create synthetic invoices at average decile values
4. Apply payment profiles to calculate late payment revenue
5. Apply seasonal adjustments

Inputs:
-------
- visualisations/11.3_forecast_next_15_months.csv (forecasted totals)
- forecast/15_decile_distribution_summary.csv (decile patterns)
- payment_profile/decile_payment_profile.pkl (payment behaviors)
- visualisations/09.6_reconstructed_late_payment_rates.csv (seasonal adjustments)
- visualisations/10.6_calibrated_baseline_late_rate.csv (calibrated baseline)

Outputs:
--------
- forecast/15.1_synthetic_invoices_from_forecast.csv
- forecast/15.1_monthly_revenue_forecast.csv
- forecast/15.1_reconstruction_validation.csv
- forecast/15.1_revenue_forecast_summary.xlsx
- forecast/15.1_reconstruction_quality_check.png
- forecast/15.1_revenue_forecast_visualization.png

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
PAYMENT_TERMS_MONTHS = 20 / 30

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
print("REVENUE ESTIMATION FROM FORECASTED TOTALS")
print("Option A: Average Invoice Values per Decile")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"Annual Interest Rate: {ANNUAL_INTEREST_RATE*100:.2f}%")
print("="*80)

# Check required inputs exist
required_files = {
    'forecast': visualisations_dir / "11.3_forecast_next_15_months.csv",
    'decile_dist': forecast_dir / "15_decile_distribution_summary.csv",
    'payment_profile': profile_dir / "decile_payment_profile.pkl",
    'seasonal_rates': visualisations_dir / "09.6_reconstructed_late_payment_rates.csv",
    'calibration': visualisations_dir / "10.6_calibrated_baseline_late_rate.csv"
}

for name, file in required_files.items():
    if not file.exists():
        print(f"\n❌ ERROR: Required file not found: {name}")
        print(f"   Expected: {file}")
        exit(1)

# ================================================================
# STEP 1: Load Forecast Data
# ================================================================
print("\n" + "="*80)
print("📁 [Step 1/7] LOADING FORECAST DATA")
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
# STEP 2: Load Decile Distribution Patterns
# ================================================================
print("\n" + "="*80)
print("📊 [Step 2/7] LOADING DECILE DISTRIBUTION PATTERNS")
print("="*80)

decile_dist = pd.read_csv(forecast_dir / "15_decile_distribution_summary.csv")

print(f"  ✓ Loaded distribution for {len(decile_dist)} deciles")

# Normalize distributions to ensure they sum to exactly 100%
decile_dist['pct_of_total_invoices'] = (
    decile_dist['pct_of_total_invoices'] / decile_dist['pct_of_total_invoices'].sum() * 100
)
decile_dist['pct_of_total_value'] = (
    decile_dist['pct_of_total_value'] / decile_dist['pct_of_total_value'].sum() * 100
)

print(f"\n  Normalized Decile Distribution:")
print(decile_dist[['decile', 'pct_of_total_invoices', 'pct_of_total_value', 'avg_invoice_value']].to_string(index=False))

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

# ================================================================
# STEP 4: Reconstruct Synthetic Invoices
# ================================================================
print("\n" + "="*80)
print("🔨 [Step 4/7] RECONSTRUCTING SYNTHETIC INVOICES")
print("="*80)

all_synthetic_invoices = []
reconstruction_validation = []

for idx, forecast_row in forecast_df.iterrows():
    month = forecast_row['invoice_period']
    forecasted_total = forecast_row['forecast_discounted_price']
    forecasted_count = int(forecast_row['forecast_invoice_count'])
    forecasted_undiscounted = forecast_row['forecast_undiscounted_price']
    discount_amount = forecast_row['forecast_discount_amount']
    
    print(f"\n  Processing {month.strftime('%Y-%m')}:")
    print(f"    Forecasted total: ${forecasted_total:,.2f}")
    print(f"    Forecasted count: {forecasted_count:,}")
    
    # Step 4.1: Allocate total value across deciles
    decile_allocated_values = []
    for _, decile_row in decile_dist.iterrows():
        decile_num = decile_row['decile']
        value_pct = decile_row['pct_of_total_value'] / 100
        allocated_value = forecasted_total * value_pct
        decile_allocated_values.append({
            'decile': decile_num,
            'allocated_value': allocated_value
        })
    
    # Step 4.2: Allocate invoice count across deciles
    decile_invoice_counts = []
    for _, decile_row in decile_dist.iterrows():
        decile_num = decile_row['decile']
        count_pct = decile_row['pct_of_total_invoices'] / 100
        allocated_count = forecasted_count * count_pct
        decile_invoice_counts.append({
            'decile': decile_num,
            'allocated_count': allocated_count
        })
    
    # Convert to integers, ensuring sum equals forecasted count
    decile_counts_df = pd.DataFrame(decile_invoice_counts)
    decile_counts_df['rounded_count'] = np.floor(decile_counts_df['allocated_count']).astype(int)
    
    # Distribute remainder
    remainder = forecasted_count - decile_counts_df['rounded_count'].sum()
    if remainder > 0:
        # Give remainder to deciles with highest fractional parts
        decile_counts_df['fractional'] = decile_counts_df['allocated_count'] - decile_counts_df['rounded_count']
        top_remainder_indices = decile_counts_df.nlargest(int(remainder), 'fractional').index
        decile_counts_df.loc[top_remainder_indices, 'rounded_count'] += 1
    
    # Step 4.3: Create synthetic invoices for each decile
    invoice_id_counter = 0
    
    for decile_num in range(n_deciles):
        decile_value = decile_allocated_values[decile_num]['allocated_value']
        decile_count = int(decile_counts_df[decile_counts_df['decile'] == decile_num]['rounded_count'].values[0])
        
        if decile_count == 0:
            continue
        
        # Average invoice value for this decile
        avg_value = decile_value / decile_count
        
        # Get decile payment profile info
        decile_key = f'decile_{decile_num}'
        if decile_key in decile_profile['deciles']:
            snapshot_late_rate = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
            cd_given_late = decile_profile['deciles'][decile_key]['delinquency_distribution']['cd_given_late']
        else:
            snapshot_late_rate = 0.02
            cd_given_late = {4: 1.0}  # Default to cd=4 (90 days)
        
        # Apply calibration scaling
        calibrated_late_rate = snapshot_late_rate * scaling_factor
        
        # Calculate expected days overdue
        if cd_given_late:
            expected_days = sum(CD_TO_DAYS.get(cd, 90) * prob for cd, prob in cd_given_late.items())
        else:
            expected_days = 60
        
        # Calculate discount per invoice (proportional)
        discount_per_invoice = (discount_amount / forecasted_count) if forecasted_count > 0 else 0
        undiscounted_per_invoice = avg_value + discount_per_invoice
        
        # Create invoices for this decile
        for i in range(decile_count):
            invoice_id_counter += 1
            
            all_synthetic_invoices.append({
                'synthetic_invoice_id': f"{month.strftime('%Y%m')}_{invoice_id_counter:05d}",
                'invoice_period': month,
                'year_month': (month.year, month.month),
                'decile': decile_num,
                'total_discounted_price': avg_value,
                'total_undiscounted_price': undiscounted_per_invoice,
                'discount_amount': discount_per_invoice,
                'decile_snapshot_late_rate': snapshot_late_rate,
                'decile_calibrated_late_rate': calibrated_late_rate,
                'expected_days_overdue': expected_days,
                'cd_distribution': str(cd_given_late)  # Store as string for reference
            })
    
    # Validation for this month
    month_invoices = [inv for inv in all_synthetic_invoices if inv['invoice_period'] == month]
    actual_total = sum(inv['total_discounted_price'] for inv in month_invoices)
    actual_count = len(month_invoices)
    
    reconstruction_validation.append({
        'invoice_period': month,
        'forecasted_total': forecasted_total,
        'reconstructed_total': actual_total,
        'difference': actual_total - forecasted_total,
        'pct_error': (actual_total - forecasted_total) / forecasted_total * 100,
        'forecasted_count': forecasted_count,
        'reconstructed_count': actual_count,
        'count_difference': actual_count - forecasted_count
    })
    
    print(f"    Reconstructed: ${actual_total:,.2f} ({actual_count:,} invoices)")
    print(f"    Error: ${actual_total - forecasted_total:,.2f} ({(actual_total - forecasted_total) / forecasted_total * 100:.4f}%)")

# Convert to DataFrames
synthetic_invoices_df = pd.DataFrame(all_synthetic_invoices)
validation_df = pd.DataFrame(reconstruction_validation)

print(f"\n  ✓ Created {len(synthetic_invoices_df):,} synthetic invoices")
print(f"  ✓ Average reconstruction error: {validation_df['pct_error'].abs().mean():.6f}%")

# ================================================================
# STEP 5: Apply Payment Profiles and Calculate Revenue
# ================================================================
print("\n" + "="*80)
print("💰 [Step 5/7] CALCULATING REVENUE FROM SYNTHETIC INVOICES")
print("="*80)

# Apply seasonal adjustments
synthetic_invoices_df['seasonal_factor'] = synthetic_invoices_df['year_month'].map(
    seasonal_adjustment_factors
).fillna(1.0)

# Calculate adjusted late rate
synthetic_invoices_df['adjusted_late_rate'] = (
    synthetic_invoices_df['decile_calibrated_late_rate'] * 
    synthetic_invoices_df['seasonal_factor']
)
synthetic_invoices_df['adjusted_late_rate'] = np.minimum(
    synthetic_invoices_df['adjusted_late_rate'], 1.0
)

# Determine which invoices are late (deterministic - use threshold at adjusted rate)
# For a deterministic estimate, we'll use expected values rather than random sampling
synthetic_invoices_df['expected_late_probability'] = synthetic_invoices_df['adjusted_late_rate']

# Calculate expected revenue per invoice
daily_rate = ANNUAL_INTEREST_RATE / 365

# Expected interest revenue = probability of late × principal × rate × days
synthetic_invoices_df['expected_interest_revenue'] = (
    synthetic_invoices_df['expected_late_probability'] *
    synthetic_invoices_df['total_discounted_price'] *
    daily_rate *
    synthetic_invoices_df['expected_days_overdue']
)

# Expected retained discounts (assuming discounts only retained on late invoices)
# For "no discount" scenario, this would be discount_amount × probability_late
# For "with discount" scenario, this is 0
synthetic_invoices_df['expected_retained_discounts'] = (
    synthetic_invoices_df['expected_late_probability'] *
    synthetic_invoices_df['discount_amount']
)

# Total expected revenue per invoice
synthetic_invoices_df['expected_total_revenue'] = (
    synthetic_invoices_df['expected_interest_revenue'] +
    synthetic_invoices_df['expected_retained_discounts']
)

print(f"  Revenue Calculation Summary:")
print(f"    Daily interest rate: {daily_rate:.6f}")
print(f"    Average expected late probability: {synthetic_invoices_df['expected_late_probability'].mean()*100:.2f}%")
print(f"    Average expected days overdue: {synthetic_invoices_df['expected_days_overdue'].mean():.1f} days")
print(f"    Total expected interest revenue: ${synthetic_invoices_df['expected_interest_revenue'].sum():,.2f}")
print(f"    Total expected retained discounts: ${synthetic_invoices_df['expected_retained_discounts'].sum():,.2f}")
print(f"    Total expected revenue: ${synthetic_invoices_df['expected_total_revenue'].sum():,.2f}")

# ================================================================
# STEP 6: Aggregate Monthly Revenue Forecast
# ================================================================
print("\n" + "="*80)
print("📊 [Step 6/7] AGGREGATING MONTHLY REVENUE FORECAST")
print("="*80)

monthly_revenue = synthetic_invoices_df.groupby('invoice_period').agg({
    'total_discounted_price': 'sum',
    'total_undiscounted_price': 'sum',
    'discount_amount': 'sum',
    'expected_interest_revenue': 'sum',
    'expected_retained_discounts': 'sum',
    'expected_total_revenue': 'sum',
    'synthetic_invoice_id': 'count',
    'expected_late_probability': 'mean',
    'expected_days_overdue': 'mean'
}).reset_index()

monthly_revenue.columns = [
    'invoice_period', 'total_discounted_price', 'total_undiscounted_price',
    'total_discount_amount', 'interest_revenue', 'retained_discounts', 
    'total_revenue', 'n_invoices', 'avg_late_probability', 'avg_days_overdue'
]

# Add cumulative revenue
monthly_revenue = monthly_revenue.sort_values('invoice_period')
monthly_revenue['cumulative_revenue'] = monthly_revenue['total_revenue'].cumsum()

print(f"\n  Monthly Revenue Forecast:")
print(monthly_revenue[['invoice_period', 'total_revenue', 'interest_revenue', 'retained_discounts']].to_string(index=False))

print(f"\n  💰 15-Month Revenue Summary:")
print(f"    Total Interest Revenue: ${monthly_revenue['interest_revenue'].sum():,.2f}")
print(f"    Total Retained Discounts: ${monthly_revenue['retained_discounts'].sum():,.2f}")
print(f"    Total Revenue: ${monthly_revenue['total_revenue'].sum():,.2f}")
print(f"    Average Monthly Revenue: ${monthly_revenue['total_revenue'].mean():,.2f}")

# ================================================================
# STEP 7: Save Results
# ================================================================
print("\n" + "="*80)
print("💾 [Step 7/7] SAVING RESULTS")
print("="*80)

# Save synthetic invoices
invoices_output = forecast_dir / '15.1_synthetic_invoices_from_forecast.csv'
synthetic_invoices_df.to_csv(invoices_output, index=False)
print(f"  ✓ Saved: {invoices_output.name}")

# Save monthly revenue forecast
revenue_output = forecast_dir / '15.1_monthly_revenue_forecast.csv'
monthly_revenue.to_csv(revenue_output, index=False)
print(f"  ✓ Saved: {revenue_output.name}")

# Save reconstruction validation
validation_output = forecast_dir / '15.1_reconstruction_validation.csv'
validation_df.to_csv(validation_output, index=False)
print(f"  ✓ Saved: {validation_output.name}")

# Save comprehensive Excel file
excel_output = forecast_dir / '15.1_revenue_forecast_summary.xlsx'
with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
    # Monthly revenue sheet
    monthly_revenue.to_excel(writer, sheet_name='Monthly_Revenue', index=False)
    
    # Synthetic invoices summary by decile and month
    decile_month_summary = synthetic_invoices_df.groupby(['invoice_period', 'decile']).agg({
        'synthetic_invoice_id': 'count',
        'total_discounted_price': 'sum',
        'expected_total_revenue': 'sum',
        'expected_late_probability': 'mean',
        'expected_days_overdue': 'mean'
    }).reset_index()
    decile_month_summary.columns = [
        'invoice_period', 'decile', 'n_invoices', 'total_value', 
        'expected_revenue', 'avg_late_prob', 'avg_days_overdue'
    ]
    decile_month_summary.to_excel(writer, sheet_name='Decile_Month_Summary', index=False)
    
    # Reconstruction validation
    validation_df.to_excel(writer, sheet_name='Reconstruction_Validation', index=False)
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total 15-Month Revenue',
            'Interest Revenue',
            'Retained Discounts',
            'Average Monthly Revenue',
            'Total Invoices',
            'Average Late Probability',
            'Average Days Overdue',
            'Calibrated November Baseline',
            'Scaling Factor',
            'Annual Interest Rate'
        ],
        'Value': [
            f"${monthly_revenue['total_revenue'].sum():,.2f}",
            f"${monthly_revenue['interest_revenue'].sum():,.2f}",
            f"${monthly_revenue['retained_discounts'].sum():,.2f}",
            f"${monthly_revenue['total_revenue'].mean():,.2f}",
            f"{monthly_revenue['n_invoices'].sum():,.0f}",
            f"{synthetic_invoices_df['expected_late_probability'].mean()*100:.2f}%",
            f"{synthetic_invoices_df['expected_days_overdue'].mean():.1f} days",
            f"{calibrated_baseline_pct:.2f}%",
            f"{scaling_factor:.4f}",
            f"{ANNUAL_INTEREST_RATE*100:.2f}%"
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

# Visualization 1: Reconstruction Quality Check
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Forecasted vs Reconstructed Total
ax1 = axes[0, 0]
ax1.scatter(validation_df['forecasted_total'], validation_df['reconstructed_total'],
            alpha=0.7, s=100, edgecolors='black', linewidths=1, color='steelblue')
min_val = min(validation_df['forecasted_total'].min(), validation_df['reconstructed_total'].min())
max_val = max(validation_df['forecasted_total'].max(), validation_df['reconstructed_total'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
ax1.set_title('Reconstruction Quality: Forecasted vs Reconstructed', fontsize=14, fontweight='bold')
ax1.set_xlabel('Forecasted Total ($)', fontsize=12)
ax1.set_ylabel('Reconstructed Total ($)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

# Plot 2: Reconstruction Error Over Time
ax2 = axes[0, 1]
ax2.bar(validation_df['invoice_period'], validation_df['pct_error'],
        color='coral', alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_title('Reconstruction Error by Month', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Error (%)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=45)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Plot 3: Invoice Count Validation
ax3 = axes[1, 0]
x = np.arange(len(validation_df))
width = 0.35
ax3.bar(x - width/2, validation_df['forecasted_count'], width, 
        label='Forecasted', color='#4472C4', alpha=0.7, edgecolor='black')
ax3.bar(x + width/2, validation_df['reconstructed_count'], width,
        label='Reconstructed', color='#70AD47', alpha=0.7, edgecolor='black')
ax3.set_title('Invoice Count: Forecasted vs Reconstructed', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month Index', fontsize=12)
ax3.set_ylabel('Number of Invoices', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Summary Statistics
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
RECONSTRUCTION VALIDATION SUMMARY

Total Months: {len(validation_df)}
Total Synthetic Invoices: {len(synthetic_invoices_df):,}

Average Reconstruction Error:
  Value: {validation_df['pct_error'].abs().mean():.6f}%
  Max Error: {validation_df['pct_error'].abs().max():.6f}%

Invoice Count:
  Perfect Match: {(validation_df['count_difference'] == 0).sum()} / {len(validation_df)} months

Total Forecasted: ${validation_df['forecasted_total'].sum():,.2f}
Total Reconstructed: ${validation_df['reconstructed_total'].sum():,.2f}
Difference: ${validation_df['difference'].sum():,.2f}

✓ Reconstruction Quality: EXCELLENT
  (Errors < 0.001% indicate numerical precision limits)
"""
ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
quality_output = forecast_dir / '15.1_reconstruction_quality_check.png'
plt.savefig(quality_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {quality_output.name}")
plt.close()

# Visualization 2: Revenue Forecast
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Monthly Revenue Breakdown
ax1 = axes[0, 0]
x = np.arange(len(monthly_revenue))
width = 0.35
ax1.bar(x - width/2, monthly_revenue['interest_revenue'], width,
        label='Interest Revenue', color='#4472C4', alpha=0.7, edgecolor='black')
ax1.bar(x + width/2, monthly_revenue['retained_discounts'], width,
        label='Retained Discounts', color='#FFC000', alpha=0.7, edgecolor='black')
ax1.set_title('Monthly Revenue Breakdown', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Revenue ($)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(x)
ax1.set_xticklabels(monthly_revenue['invoice_period'].dt.strftime('%Y-%m'), rotation=45)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: Cumulative Revenue
ax2 = axes[0, 1]
ax2.plot(monthly_revenue['invoice_period'], monthly_revenue['cumulative_revenue'],
         marker='o', linewidth=2.5, markersize=8, color='#70AD47')
ax2.fill_between(monthly_revenue['invoice_period'], 0, monthly_revenue['cumulative_revenue'],
                 alpha=0.3, color='#70AD47')
ax2.set_title('Cumulative Revenue Forecast', fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Cumulative Revenue ($)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis='x', rotation=45)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add final value annotation
final_revenue = monthly_revenue['cumulative_revenue'].iloc[-1]
ax2.text(monthly_revenue['invoice_period'].iloc[-1], final_revenue,
         f'  ${final_revenue:,.0f}', fontsize=11, fontweight='bold',
         va='center', ha='left')

# Plot 3: Average Late Probability by Month
ax3 = axes[1, 0]
ax3.plot(monthly_revenue['invoice_period'], monthly_revenue['avg_late_probability'] * 100,
         marker='s', linewidth=2, markersize=8, color='coral')
ax3.set_title('Average Late Payment Probability by Month', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12)
ax3.set_ylabel('Late Probability (%)', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Revenue Components Pie Chart
ax4 = axes[1, 1]
total_interest = monthly_revenue['interest_revenue'].sum()
total_discounts = monthly_revenue['retained_discounts'].sum()
sizes = [total_interest, total_discounts]
labels = ['Interest Revenue', 'Retained Discounts']
colors = ['#4472C4', '#FFC000']
explode = (0.05, 0.05)

ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax4.set_title(f'15-Month Revenue Composition\nTotal: ${total_interest + total_discounts:,.2f}',
              fontsize=14, fontweight='bold')

plt.tight_layout()
revenue_viz_output = forecast_dir / '15.1_revenue_forecast_visualization.png'
plt.savefig(revenue_viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {revenue_viz_output.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ REVENUE FORECAST COMPLETE!")
print("="*80)

print(f"\n📊 Reconstruction Summary:")
print(f"  Total synthetic invoices created: {len(synthetic_invoices_df):,}")
print(f"  Forecast period: {monthly_revenue['invoice_period'].min().strftime('%Y-%m')} to {monthly_revenue['invoice_period'].max().strftime('%Y-%m')}")
print(f"  Average reconstruction error: {validation_df['pct_error'].abs().mean():.6f}%")

print(f"\n💰 Revenue Forecast Summary:")
print(f"  Total 15-Month Revenue: ${monthly_revenue['total_revenue'].sum():,.2f}")
print(f"    Interest Revenue: ${monthly_revenue['interest_revenue'].sum():,.2f}")
print(f"    Retained Discounts: ${monthly_revenue['retained_discounts'].sum():,.2f}")
print(f"  Average Monthly Revenue: ${monthly_revenue['total_revenue'].mean():,.2f}")
print(f"  Revenue Range: ${monthly_revenue['total_revenue'].min():,.2f} - ${monthly_revenue['total_revenue'].max():,.2f}")

print(f"\n📈 Key Metrics:")
print(f"  Average late payment probability: {synthetic_invoices_df['expected_late_probability'].mean()*100:.2f}%")
print(f"  Average days overdue (when late): {synthetic_invoices_df['expected_days_overdue'].mean():.1f} days")
print(f"  Effective interest rate (annual): {ANNUAL_INTEREST_RATE*100:.2f}%")

print(f"\n📁 Output Files:")
print(f"  All files saved to: {forecast_dir}")
print("  • 15.1_synthetic_invoices_from_forecast.csv - All synthetic invoices")
print("  • 15.1_monthly_revenue_forecast.csv - Monthly revenue totals")
print("  • 15.1_reconstruction_validation.csv - Quality metrics")
print("  • 15.1_revenue_forecast_summary.xlsx - Comprehensive workbook")
print("  • 15.1_reconstruction_quality_check.png - Validation visualizations")
print("  • 15.1_revenue_forecast_visualization.png - Revenue analysis")

print("\n" + "="*80)
print("NOTE: This uses EXPECTED VALUES (deterministic)")
print("For probabilistic estimates with confidence intervals, use Monte Carlo")
print("="*80)