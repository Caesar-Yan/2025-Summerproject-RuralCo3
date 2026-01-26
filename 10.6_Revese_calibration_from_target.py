'''
Docstring for 10.6_Reverse_calibration_from_target

This script works BACKWARDS from the historical target revenue ($1.043M) to determine
what the November baseline late payment rate should be, then applies seasonal adjustments
to reconstruct a more accurate revenue estimate.

CORRECTED: Uses DISCOUNTED price for interest calculation (customers get discount, then pay interest on that amount)

Inputs:
- ats_grouped_transformed_with_discounts.csv
- invoice_grouped_transformed_with_discounts.csv
- 09.6_reconstructed_late_payment_rates.csv (seasonal adjustment factors)
- decile_payment_profile.pkl (for cd level distributions only)

Outputs:
- 10.6_calibrated_baseline_late_rate.csv
- 10.6_calibration_analysis.png
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path
from scipy.optimize import minimize_scalar

# ================================================================
# CONFIGURATION
# ================================================================
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
data_cleaning_dir = base_dir / "data_cleaning"
visualisations_dir = base_dir / "visualisations"

ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30

FY2025_START = pd.Timestamp("2024-07-01")
FY2025_END = pd.Timestamp("2025-06-30")

TARGET_REVENUE = 1_043_000  # Historical target to match

OUTPUT_DIR = visualisations_dir

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

np.random.seed(RANDOM_SEED)

print("\n" + "="*70)
print("REVERSE CALIBRATION: FINDING BASELINE FROM TARGET REVENUE")
print("="*70)
print(f"Target Revenue: ${TARGET_REVENUE:,.2f}")
print(f"Approach: Work backwards to find November baseline late rate")
print(f"USING DISCOUNTED PRICE for interest calculation")

# ================================================================
# Load data
# ================================================================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Load seasonal rates
seasonal_rates = pd.read_csv(visualisations_dir / "09.6_reconstructed_late_payment_rates.csv")
seasonal_rates['invoice_period'] = pd.to_datetime(seasonal_rates['invoice_period'])

# Get November 2025 for reference
november_2025_row = seasonal_rates[
    (seasonal_rates['invoice_period'].dt.year == 2025) & 
    (seasonal_rates['invoice_period'].dt.month == 11)
]

if len(november_2025_row) == 0:
    print("ERROR: November 2025 not found")
    exit(1)

november_reconstructed_rate = november_2025_row['reconstructed_late_rate_pct'].values[0] / 100
print(f"November reconstructed rate (from spending): {november_reconstructed_rate*100:.2f}%")

# Load invoices
ats_grouped = pd.read_csv(data_cleaning_dir / 'ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv(data_cleaning_dir / 'invoice_grouped_transformed_with_discounts.csv')

ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"Total invoices loaded: {len(combined_df):,}")

# Filter negatives
combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()

# Parse dates
def parse_invoice_period(series: pd.Series) -> pd.Series:
    s = series.copy()
    s_str = s.astype(str).str.strip()
    s_str = s_str.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
    
    mask_yyyymm = s_str.str.fullmatch(r"\d{6}", na=False)
    out = pd.Series(pd.NaT, index=s.index)
    
    if mask_yyyymm.any():
        out.loc[mask_yyyymm] = pd.to_datetime(s_str.loc[mask_yyyymm], format="%Y%m", errors="coerce")
    
    mask_other = ~mask_yyyymm
    if mask_other.any():
        out.loc[mask_other] = pd.to_datetime(s_str.loc[mask_other], errors="coerce")
    
    return out

combined_df['invoice_period'] = parse_invoice_period(combined_df['invoice_period'])
combined_df = combined_df[combined_df['invoice_period'].notna()].copy()

# Filter to FY2025
fy2025_df = combined_df[
    (combined_df['invoice_period'] >= FY2025_START) & 
    (combined_df['invoice_period'] <= FY2025_END)
].copy()

print(f"FY2025 invoices: {len(fy2025_df):,}")

# Load decile profile
with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
    decile_profile = pickle.load(f)

n_deciles = decile_profile['metadata']['n_deciles']
print(f"Loaded {n_deciles} deciles")

# Map to deciles
fy2025_df = fy2025_df.sort_values('total_undiscounted_price').reset_index(drop=True)

fy2025_df['decile'] = pd.qcut(
    fy2025_df['total_undiscounted_price'], 
    q=n_deciles, 
    labels=False,
    duplicates='drop'
)

# ================================================================
# Calculate seasonal adjustment factors
# ================================================================
print("\n" + "="*70)
print("CALCULATING SEASONAL ADJUSTMENT FACTORS")
print("="*70)

seasonal_adjustment_factors = {}

for _, row in seasonal_rates.iterrows():
    year_month = (row['invoice_period'].year, row['invoice_period'].month)
    month_rate = row['reconstructed_late_rate_pct'] / 100
    adjustment_factor = month_rate / november_reconstructed_rate
    seasonal_adjustment_factors[year_month] = adjustment_factor

print(f"Calculated {len(seasonal_adjustment_factors)} monthly adjustment factors")

# ================================================================
# PRE-COMPUTE all static values for VECTORIZED calculation
# ================================================================
print("\n" + "="*70)
print("PRE-COMPUTING STATIC VALUES")
print("="*70)

# Create year-month column
fy2025_df['year_month'] = list(zip(fy2025_df['invoice_period'].dt.year, 
                                     fy2025_df['invoice_period'].dt.month))

# Map seasonal factors
fy2025_df['seasonal_factor'] = fy2025_df['year_month'].map(seasonal_adjustment_factors).fillna(1.0)

# Get decile base rates from profile
decile_snapshot_rates = {}
for i in range(n_deciles):
    decile_key = f'decile_{i}'
    if decile_key in decile_profile['deciles']:
        rate = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
        decile_snapshot_rates[i] = rate
    else:
        decile_snapshot_rates[i] = 0.02

fy2025_df['decile_snapshot_rate'] = fy2025_df['decile'].map(decile_snapshot_rates)

# Calculate expected days overdue for each decile
decile_expected_days = {}
for i in range(n_deciles):
    decile_key = f'decile_{i}'
    if decile_key in decile_profile['deciles']:
        cd_given_late = decile_profile['deciles'][decile_key]['delinquency_distribution']['cd_given_late']
        if cd_given_late:
            expected_days = sum(CD_TO_DAYS.get(cd, 90) * prob for cd, prob in cd_given_late.items())
        else:
            expected_days = 60
    else:
        expected_days = 60
    decile_expected_days[i] = expected_days

fy2025_df['expected_days_overdue'] = fy2025_df['decile'].map(decile_expected_days)

print(f"✓ Pre-computed values for {len(fy2025_df):,} invoices")

# Show summary of what we're using
print(f"\nPricing summary:")
print(f"  Total undiscounted value: ${fy2025_df['total_undiscounted_price'].sum():,.2f}")
print(f"  Total discounted value: ${fy2025_df['total_discounted_price'].sum():,.2f}")
print(f"  Total discount amount: ${fy2025_df['discount_amount'].sum():,.2f}")
print(f"\n  Using DISCOUNTED price for interest calculation")

# ================================================================
# VECTORIZED revenue calculation (WITH DISCOUNT scenario)
# ================================================================
def calculate_expected_revenue_vectorized(november_baseline_rate):
    """
    FAST vectorized calculation of expected revenue
    
    CORRECTED: Uses DISCOUNTED price for principal (interest charged on discounted amount)
    """
    # Scaling factor from 2% snapshot to new baseline
    old_baseline = 0.02
    scaling_factor = november_baseline_rate / old_baseline
    
    # Calculate adjusted late rate for each invoice (VECTORIZED)
    scaled_base_rate = fy2025_df['decile_snapshot_rate'] * scaling_factor
    adjusted_late_rate = scaled_base_rate * fy2025_df['seasonal_factor']
    adjusted_late_rate = np.minimum(adjusted_late_rate, 1.0)
    
    # Calculate expected interest using DISCOUNTED price (VECTORIZED)
    daily_rate = ANNUAL_INTEREST_RATE / 365
    expected_interest = (
        adjusted_late_rate * 
        fy2025_df['total_discounted_price'] *  # CORRECTED: Use discounted price
        daily_rate * 
        fy2025_df['expected_days_overdue']
    )
    
    # Calculate expected retained discounts (VECTORIZED)
    # Note: In "with discount" scenario, there are NO retained discounts
    # Customers get discount whether they pay on time or late
    # We only charge interest on late payments
    expected_retained = 0  # No retained discounts in "with discount" scenario
    
    # Total revenue (interest only, no retained discounts)
    total_revenue = expected_interest.sum()
    
    return total_revenue

# ================================================================
# Test the vectorized function
# ================================================================
print("\n" + "="*70)
print("TESTING VECTORIZED CALCULATION SPEED")
print("="*70)

import time

start = time.time()
test_revenue = calculate_expected_revenue_vectorized(0.15)
elapsed = time.time() - start

print(f"✓ Calculated revenue for 15% baseline in {elapsed:.3f} seconds")
print(f"  Expected revenue: ${test_revenue:,.2f}")

# ================================================================
# Test with different baseline rates
# ================================================================
print("\n" + "="*70)
print("TESTING DIFFERENT BASELINE RATES")
print("="*70)

test_baselines = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

print(f"{'Baseline Rate':<15} {'Expected Revenue':<20} {'Gap to Target':<20}")
print("-" * 60)

test_results = []
for test_rate in test_baselines:
    revenue = calculate_expected_revenue_vectorized(test_rate)
    gap = revenue - TARGET_REVENUE
    test_results.append({'baseline': test_rate, 'revenue': revenue, 'gap': gap})
    print(f"{test_rate*100:>6.1f}%          ${revenue:>15,.2f}      ${gap:>15,.2f}")

# ================================================================
# OPTIMIZATION: Find exact baseline that hits target
# ================================================================
print("\n" + "="*70)
print("OPTIMIZING TO FIND CALIBRATED BASELINE")
print("="*70)

def objective_function(baseline_rate):
    """Return absolute difference from target"""
    revenue = calculate_expected_revenue_vectorized(baseline_rate)
    return abs(revenue - TARGET_REVENUE)

# Optimize
print("Running optimization...")
start = time.time()
result = minimize_scalar(objective_function, bounds=(0.01, 0.99), method='bounded', 
                        options={'xatol': 1e-6})
elapsed = time.time() - start

calibrated_baseline = result.x
calibrated_revenue = calculate_expected_revenue_vectorized(calibrated_baseline)

print(f"✓ Optimization completed in {elapsed:.2f} seconds")
print(f"\n✓ CALIBRATION COMPLETE")
print(f"  Calibrated November baseline rate: {calibrated_baseline*100:.2f}%")
print(f"  Expected revenue at this rate: ${calibrated_revenue:,.2f}")
print(f"  Target revenue: ${TARGET_REVENUE:,.2f}")
print(f"  Difference: ${abs(calibrated_revenue - TARGET_REVENUE):,.2f}")

# Calculate scaling factor
old_baseline_snapshot = 0.02
scaling_factor = calibrated_baseline / old_baseline_snapshot
print(f"\n  Scaling factor from snapshot: {scaling_factor:.2f}x")
print(f"  (Your 2% snapshot rate × {scaling_factor:.2f} = {calibrated_baseline*100:.2f}%)")

# ================================================================
# Show calibrated decile rates
# ================================================================
print("\n" + "="*70)
print("CALIBRATED DECILE BASE RATES (November)")
print("="*70)

print(f"{'Decile':<10} {'Snapshot Rate':<15} {'Calibrated Rate':<20}")
print("-" * 50)

calibrated_decile_rates = {}
for i in range(n_deciles):
    snapshot_rate = decile_snapshot_rates.get(i, 0.02)
    calibrated_rate = snapshot_rate * scaling_factor
    calibrated_decile_rates[i] = calibrated_rate
    print(f"{i:<10} {snapshot_rate*100:>6.2f}%          {calibrated_rate*100:>6.2f}%")

# ================================================================
# Show seasonal adjusted rates
# ================================================================
print("\n" + "="*70)
print("SEASONAL ADJUSTED RATES (Using Calibrated Baseline)")
print("="*70)

print(f"{'Month':<12} {'Seasonal Factor':<18} {'Avg Calibrated Rate':<25}")
print("-" * 60)

monthly_calibrated_rates = {}
for year_month, factor in sorted(seasonal_adjustment_factors.items()):
    avg_calibrated = np.mean(list(calibrated_decile_rates.values())) * factor
    monthly_calibrated_rates[year_month] = avg_calibrated
    print(f"{year_month[0]}-{year_month[1]:02d}      {factor:>6.3f}x            {avg_calibrated*100:>8.2f}%")

# ================================================================
# Save calibration results
# ================================================================
print("\n" + "="*70)
print("SAVING CALIBRATION RESULTS")
print("="*70)

calibration_summary = pd.DataFrame([{
    'target_revenue': TARGET_REVENUE,
    'calibrated_november_baseline_pct': calibrated_baseline * 100,
    'expected_revenue_at_baseline': calibrated_revenue,
    'snapshot_baseline_pct': old_baseline_snapshot * 100,
    'scaling_factor': scaling_factor,
    'optimization_success': result.success,
    'optimization_time_seconds': elapsed,
    'FY2025_start': FY2025_START,
    'FY2025_end': FY2025_END,
    'n_invoices': len(fy2025_df),
    'total_invoice_value_undiscounted': fy2025_df['total_undiscounted_price'].sum(),
    'total_invoice_value_discounted': fy2025_df['total_discounted_price'].sum(),
    'calibration_uses_discounted_price': True
}])

calibration_file = os.path.join(OUTPUT_DIR, '10.6_calibrated_baseline_late_rate.csv')
calibration_summary.to_csv(calibration_file, index=False)
print(f"✓ Saved calibration summary to: {calibration_file}")

# ================================================================
# Create visualization
# ================================================================
print("\n" + "="*70)
print("CREATING CALIBRATION VISUALIZATION")
print("="*70)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Revenue vs Baseline Rate
baseline_range = [r['baseline'] for r in test_results]
revenue_curve = [r['revenue'] for r in test_results]

ax1.plot([b*100 for b in baseline_range], revenue_curve, 'o-', linewidth=2.5, 
        color='steelblue', markersize=8)
ax1.axhline(y=TARGET_REVENUE, color='red', linestyle='--', linewidth=2, 
           label='Target Revenue')
ax1.axvline(x=calibrated_baseline * 100, color='green', linestyle=':', linewidth=2, 
           label=f'Calibrated Baseline ({calibrated_baseline*100:.1f}%)')
ax1.axvline(x=old_baseline_snapshot * 100, color='orange', linestyle=':', linewidth=2,
           label=f'Snapshot Baseline ({old_baseline_snapshot*100:.1f}%)')

ax1.scatter([calibrated_baseline * 100], [calibrated_revenue], s=200, color='green', 
           zorder=5, edgecolor='black', linewidth=2)

ax1.set_title('Revenue vs November Baseline Late Rate\n(Using Discounted Price)', 
             fontsize=14, fontweight='bold')
ax1.set_xlabel('November Baseline Late Rate (%)', fontsize=12)
ax1.set_ylabel('Expected Revenue ($)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))

# Plot 2: Decile rates comparison
deciles = list(range(n_deciles))
snapshot_rates = [decile_snapshot_rates.get(i, 0) * 100 for i in deciles]
calibrated_rates = [calibrated_decile_rates.get(i, 0) * 100 for i in deciles]

x = np.arange(len(deciles))
width = 0.35

ax2.bar(x - width/2, snapshot_rates, width, label='Snapshot (2%)', 
       color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.bar(x + width/2, calibrated_rates, width, 
       label=f'Calibrated ({calibrated_baseline*100:.1f}%)', 
       color='green', alpha=0.7, edgecolor='black', linewidth=0.5)

ax2.set_title('Decile Base Rates: Snapshot vs Calibrated', fontsize=14, fontweight='bold')
ax2.set_xlabel('Decile', fontsize=12)
ax2.set_ylabel('Late Payment Rate (%)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([f'D{i}' for i in deciles])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Monthly calibrated rates
months = sorted(monthly_calibrated_rates.keys())
month_labels = [f"{m[0]}-{m[1]:02d}" for m in months]
month_rates = [monthly_calibrated_rates[m] * 100 for m in months]

ax3.plot(range(len(months)), month_rates, marker='o', linewidth=2.5, 
        markersize=8, color='purple', alpha=0.8)
ax3.axhline(y=calibrated_baseline * 100, color='green', linestyle='--', 
           linewidth=2, alpha=0.5, label=f'November Baseline ({calibrated_baseline*100:.1f}%)')

ax3.set_title('Seasonally Adjusted Rates (Calibrated)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12)
ax3.set_ylabel('Average Late Payment Rate (%)', fontsize=12)
ax3.set_xticks(range(len(months)))
ax3.set_xticklabels(month_labels, rotation=45, ha='right')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Summary statistics
ax4.axis('off')

summary_text = f"""
CALIBRATION SUMMARY

Target Revenue: ${TARGET_REVENUE:,.0f}
Expected Revenue: ${calibrated_revenue:,.0f}

BASELINE RATES:
  Snapshot (November): {old_baseline_snapshot*100:.2f}%
  Calibrated (November): {calibrated_baseline*100:.2f}%
  Scaling Factor: {scaling_factor:.2f}x

PRICING USED:
  Interest charged on: DISCOUNTED price
  Total discounted value: ${fy2025_df['total_discounted_price'].sum():,.0f}
  
  (Customers get discount, then pay
   interest on the discounted amount)

INTERPRETATION:
  Your 2% snapshot rate was too low
  True November baseline: ~{calibrated_baseline*100:.1f}%
  
  This suggests the snapshot captured
  "currently overdue" status, not the
  percentage of invoices that eventually
  become late.

SEASONAL RANGE:
  Lowest month: {min(month_rates):.1f}%
  Highest month: {max(month_rates):.1f}%
  Average: {np.mean(month_rates):.1f}%

FY2025 DATA:
  Total invoices: {len(fy2025_df):,}
  Discounted value: ${fy2025_df['total_discounted_price'].sum():,.0f}
"""

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, '10.6_calibration_analysis.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {viz_path}")
plt.close()

print("\n" + "="*70)
print("CALIBRATION COMPLETE")
print("="*70)
print(f"\nKey Finding:")
print(f"  Your November baseline should be {calibrated_baseline*100:.1f}%, not 2%")
print(f"  This is {scaling_factor:.1f}x higher than the snapshot rate")
print(f"\nPricing Note:")
print(f"  Calibration uses DISCOUNTED price for interest calculation")
print(f"  (Customers get discount, interest charged on discounted amount)")
print(f"\nNext Steps:")
print(f"  1. Use {calibrated_baseline*100:.1f}% as your November baseline")
print(f"  2. Apply seasonal adjustments to this calibrated rate")
print(f"  3. Re-run simulation (10.7) with calibrated decile rates")
print(f"\nFiles created:")
print(f"  - 10.6_calibrated_baseline_late_rate.csv")
print(f"  - 10.6_calibration_analysis.png")
print("="*70)