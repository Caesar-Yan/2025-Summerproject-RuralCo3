'''
Docstring for 10.7_Calculation_with_calibrated_baseline

This script makes estimates of revenue generated under discount and no-discount scenarios,
using CALIBRATED baseline late payment rates (from 10.6) with seasonal adjustments.
FULLY VECTORIZED for speed.

Inputs:
- ats_grouped_transformed_with_discounts.csv
- invoice_grouped_transformed_with_discounts.csv
- 09.6_reconstructed_late_payment_rates.csv (seasonal adjustment factors)
- 10.6_calibrated_baseline_late_rate.csv (calibrated November baseline)
- decile_payment_profile.pkl (for cd level distributions)

Outputs:
- 10.7_FY2025_calibrated_comparison_summary.csv
- 10.7_FY2025_calibrated_detailed_simulations.xlsx
- 10.7_cd_level_analysis_calibrated.csv
- 10.7_cumulative_revenue_calibrated.png
- 10.7_monthly_late_rates_comparison.png
- 10.7_decile_seasonal_adjustments.csv
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path

# ================================================================
# CONFIGURATION
# ================================================================
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
data_cleaning_dir = base_dir / "data_cleaning"
visualisations_dir = base_dir / "visualisations"

ANNUAL_INTEREST_RATE = 0.2395
RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30

FY2025_START = pd.Timestamp("2024-07-01")
FY2025_END = pd.Timestamp("2025-06-30")

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

print("\n" + "="*70)
print("SIMULATION WITH CALIBRATED BASELINE (VECTORIZED)")
print("="*70)

np.random.seed(RANDOM_SEED)

# ================================================================
# Load calibrated baseline
# ================================================================
print("\n" + "="*70)
print("LOADING CALIBRATED BASELINE")
print("="*70)

try:
    calibration_df = pd.read_csv(visualisations_dir / "10.6_calibrated_baseline_late_rate.csv")
    calibrated_baseline_pct = calibration_df['calibrated_november_baseline_pct'].values[0]
    calibrated_baseline = calibrated_baseline_pct / 100
    scaling_factor = calibration_df['scaling_factor'].values[0]
    
    print(f"✓ Loaded calibrated baseline from 10.6")
    print(f"  Calibrated November baseline: {calibrated_baseline*100:.2f}%")
    print(f"  Scaling factor from snapshot: {scaling_factor:.2f}x")
    
except FileNotFoundError:
    print("ERROR: Calibrated baseline not found!")
    print("Please run 10.6_Reverse_calibration_from_target.py first")
    exit(1)

# ================================================================
# Load seasonal rates
# ================================================================
print("\n" + "="*70)
print("LOADING SEASONAL RATES")
print("="*70)

seasonal_rates = pd.read_csv(visualisations_dir / "09.6_reconstructed_late_payment_rates.csv")
seasonal_rates['invoice_period'] = pd.to_datetime(seasonal_rates['invoice_period'])

november_2025_row = seasonal_rates[
    (seasonal_rates['invoice_period'].dt.year == 2025) & 
    (seasonal_rates['invoice_period'].dt.month == 11)
]

november_reconstructed_rate = november_2025_row['reconstructed_late_rate_pct'].values[0] / 100

# Calculate seasonal adjustment factors
seasonal_adjustment_factors = {}
for _, row in seasonal_rates.iterrows():
    year_month = (row['invoice_period'].year, row['invoice_period'].month)
    month_rate = row['reconstructed_late_rate_pct'] / 100
    adjustment_factor = month_rate / november_reconstructed_rate
    seasonal_adjustment_factors[year_month] = adjustment_factor

print(f"✓ Calculated {len(seasonal_adjustment_factors)} seasonal adjustment factors")

# ================================================================
# Load invoices
# ================================================================
print("\n" + "="*70)
print("LOADING INVOICE DATA")
print("="*70)

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

# ================================================================
# Load decile profile
# ================================================================
print("\n" + "="*70)
print("LOADING DECILE PAYMENT PROFILE")
print("="*70)

with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
    decile_profile = pickle.load(f)

n_deciles = decile_profile['metadata']['n_deciles']
print(f"✓ Loaded {n_deciles} deciles")

# Map to deciles
fy2025_df = fy2025_df.sort_values('total_undiscounted_price').reset_index(drop=True)
fy2025_df['decile'] = pd.qcut(
    fy2025_df['total_undiscounted_price'], 
    q=n_deciles, 
    labels=False,
    duplicates='drop'
)

print(f"✓ Mapped {len(fy2025_df):,} invoices to deciles")

# ================================================================
# PRE-COMPUTE all values for VECTORIZED simulation
# ================================================================
print("\n" + "="*70)
print("PRE-COMPUTING VALUES FOR VECTORIZED SIMULATION")
print("="*70)

# Create year-month column
fy2025_df['year_month'] = list(zip(fy2025_df['invoice_period'].dt.year, 
                                     fy2025_df['invoice_period'].dt.month))

# Map seasonal factors
fy2025_df['seasonal_factor'] = fy2025_df['year_month'].map(seasonal_adjustment_factors).fillna(1.0)

# Get decile snapshot rates and scale to calibrated
decile_snapshot_rates = {}
decile_calibrated_rates = {}
for i in range(n_deciles):
    decile_key = f'decile_{i}'
    if decile_key in decile_profile['deciles']:
        snapshot_rate = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
    else:
        snapshot_rate = 0.02
    
    decile_snapshot_rates[i] = snapshot_rate
    decile_calibrated_rates[i] = snapshot_rate * scaling_factor

fy2025_df['decile_snapshot_rate'] = fy2025_df['decile'].map(decile_snapshot_rates)
fy2025_df['decile_calibrated_rate'] = fy2025_df['decile'].map(decile_calibrated_rates)

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

# Pre-compute cd distribution for sampling
decile_cd_distributions = {}
for i in range(n_deciles):
    decile_key = f'decile_{i}'
    if decile_key in decile_profile['deciles']:
        cd_given_late = decile_profile['deciles'][decile_key]['delinquency_distribution']['cd_given_late']
        if cd_given_late:
            cd_levels = list(cd_given_late.keys())
            cd_probs = np.array(list(cd_given_late.values()))
            cd_probs = cd_probs / cd_probs.sum()
            decile_cd_distributions[i] = (cd_levels, cd_probs)
        else:
            decile_cd_distributions[i] = ([0], [1.0])
    else:
        decile_cd_distributions[i] = ([0], [1.0])

print(f"✓ Pre-computed all values for {len(fy2025_df):,} invoices")

# ================================================================
# VECTORIZED simulation function - SHARED late status
# ================================================================
def simulate_both_scenarios_with_shared_late_status(invoices_df):
    """
    Simulate BOTH scenarios using the SAME late payment assignments
    
    This ensures fair comparison - the exact same invoices are late in both scenarios,
    only the principal amount and retained discounts differ.
    
    Returns: (with_discount_df, no_discount_df)
    """
    print(f"\nRunning SHARED simulation for both scenarios...")
    print(f"  Simulating {len(invoices_df):,} invoices...")
    
    # ================================================================
    # STEP 1: Determine which invoices are late (SAME FOR BOTH)
    # ================================================================
    df = invoices_df.copy()
    
    # Calculate adjusted late rate (VECTORIZED)
    df['adjusted_late_rate'] = df['decile_calibrated_rate'] * df['seasonal_factor']
    df['adjusted_late_rate'] = np.minimum(df['adjusted_late_rate'], 1.0)
    
    # Determine late status (VECTORIZED random draw) - SHARED
    random_draws = np.random.random(len(df))
    df['is_late'] = random_draws < df['adjusted_late_rate']
    
    # Sample cd levels for late invoices (SHARED)
    df['cd_level'] = 0
    df['days_overdue'] = 0.0
    
    for decile_num in df['decile'].unique():
        decile_mask = (df['decile'] == decile_num) & df['is_late']
        n_late_in_decile = decile_mask.sum()
        
        if n_late_in_decile > 0:
            cd_levels, cd_probs = decile_cd_distributions.get(int(decile_num), ([0], [1.0]))
            sampled_cd = np.random.choice(cd_levels, size=n_late_in_decile, p=cd_probs)
            df.loc[decile_mask, 'cd_level'] = sampled_cd
            df.loc[decile_mask, 'days_overdue'] = [CD_TO_DAYS.get(cd, 90) for cd in sampled_cd]
    
    df['months_overdue'] = df['days_overdue'] / 30
    
    # Calculate dates (SHARED)
    df['due_date'] = df['invoice_period']
    df['payment_date'] = df['due_date'] + pd.to_timedelta(df['days_overdue'], unit='D')
    
    print(f"  ✓ Determined late status (shared across scenarios)")
    print(f"  Late invoices: {df['is_late'].sum():,} ({df['is_late'].sum()/len(df)*100:.1f}%)")
    
    # ================================================================
    # STEP 2: Create WITH DISCOUNT scenario
    # ================================================================
    with_discount = df.copy()
    
    # Principal = discounted price
    with_discount['principal_amount'] = with_discount['total_discounted_price']
    with_discount['retained_discounts'] = 0
    
    # Calculate interest
    daily_rate = ANNUAL_INTEREST_RATE / 365
    with_discount['interest_charged'] = (
        with_discount['principal_amount'] * daily_rate * with_discount['days_overdue']
    )
    
    # Total revenue
    with_discount['credit_card_revenue'] = (
        with_discount['interest_charged'] + with_discount['retained_discounts']
    )
    
    # ================================================================
    # STEP 3: Create NO DISCOUNT scenario
    # ================================================================
    no_discount = df.copy()
    
    # Principal = undiscounted price
    no_discount['principal_amount'] = no_discount['total_undiscounted_price']
    
    # Retained discounts = discount amount for late invoices only
    no_discount['retained_discounts'] = np.where(
        no_discount['is_late'], 
        no_discount['discount_amount'], 
        0
    )
    
    # Calculate interest
    no_discount['interest_charged'] = (
        no_discount['principal_amount'] * daily_rate * no_discount['days_overdue']
    )
    
    # Total revenue
    no_discount['credit_card_revenue'] = (
        no_discount['interest_charged'] + no_discount['retained_discounts']
    )
    
    print(f"\n  WITH DISCOUNT scenario:")
    print(f"    Total revenue: ${with_discount['credit_card_revenue'].sum():,.2f}")
    
    print(f"\n  NO DISCOUNT scenario:")
    print(f"    Interest: ${no_discount['interest_charged'].sum():,.2f}")
    print(f"    Retained: ${no_discount['retained_discounts'].sum():,.2f}")
    print(f"    Total revenue: ${no_discount['credit_card_revenue'].sum():,.2f}")
    
    return with_discount, no_discount

# ================================================================
# Run simulation (SHARED late status for both scenarios)
# ================================================================
print("\n" + "="*70)
print("RUNNING SIMULATION (SHARED LATE STATUS)")
print("="*70)

import time

start = time.time()
with_discount, no_discount = simulate_both_scenarios_with_shared_late_status(fy2025_df)
elapsed = time.time() - start

print(f"\n✓ Simulation complete in {elapsed:.2f} seconds")

# ================================================================
# Save decile seasonal adjustments
# ================================================================
print("\n" + "="*70)
print("SAVING DECILE SEASONAL ADJUSTMENTS")
print("="*70)

decile_seasonal_summary = []
for decile_num in range(n_deciles):
    snapshot_rate = decile_snapshot_rates.get(decile_num, 0.02)
    calibrated_base = decile_calibrated_rates.get(decile_num, 0.02)
    
    for year_month, factor in sorted(seasonal_adjustment_factors.items()):
        adjusted_rate = calibrated_base * factor
        
        decile_seasonal_summary.append({
            'decile': decile_num,
            'year': year_month[0],
            'month': year_month[1],
            'snapshot_rate_pct': snapshot_rate * 100,
            'calibrated_base_rate_pct': calibrated_base * 100,
            'seasonal_factor': factor,
            'adjusted_late_rate_pct': adjusted_rate * 100
        })

decile_seasonal_df = pd.DataFrame(decile_seasonal_summary)
decile_seasonal_file = os.path.join(OUTPUT_DIR, '10.7_decile_seasonal_adjustments.csv')
decile_seasonal_df.to_csv(decile_seasonal_file, index=False)
print(f"✓ Saved to: {decile_seasonal_file}")

# ================================================================
# Summary statistics
# ================================================================
print("\n" + "="*70)
print(f"REVENUE COMPARISON: FY2025")
print("="*70)

def print_scenario_summary(df, scenario_name):
    print(f"\n{scenario_name}")
    print("-" * 70)
    
    total_invoices = len(df)
    n_late = df['is_late'].sum()
    n_on_time = total_invoices - n_late
    
    print(f"Total invoices: {total_invoices:,}")
    print(f"  On time: {n_on_time:,} ({n_on_time/total_invoices*100:.1f}%)")
    print(f"  Late: {n_late:,} ({n_late/total_invoices*100:.1f}%)")
    
    if n_late > 0:
        print(f"  Avg days overdue (late): {df[df['is_late']]['days_overdue'].mean():.1f}")
    
    print(f"\nCalibration:")
    print(f"  Avg calibrated base: {df['decile_calibrated_rate'].mean()*100:.2f}%")
    print(f"  Avg adjusted rate: {df['adjusted_late_rate'].mean()*100:.2f}%")
    
    interest_total = df['interest_charged'].sum()
    retained_total = df['retained_discounts'].sum()
    revenue_total = df['credit_card_revenue'].sum()
    
    print(f"\nRevenue:")
    print(f"  Interest: ${interest_total:,.2f}")
    print(f"  Retained: ${retained_total:,.2f}")
    print(f"  TOTAL: ${revenue_total:,.2f}")
    
    # Monthly breakdown
    monthly_revenue = df.groupby(df['invoice_period'].dt.to_period('M')).agg({
        'credit_card_revenue': 'sum',
        'is_late': ['sum', 'count'],
        'adjusted_late_rate': 'mean'
    })
    monthly_revenue.columns = ['revenue', 'n_late', 'n_invoices', 'avg_adj_rate']
    monthly_revenue['actual_late_pct'] = monthly_revenue['n_late'] / monthly_revenue['n_invoices'] * 100
    print(f"\nMonthly breakdown:")
    print(monthly_revenue.to_string())
    
    return {
        'scenario': scenario_name,
        'total_invoices': total_invoices,
        'n_late': n_late,
        'pct_late': n_late/total_invoices*100,
        'interest_revenue': interest_total,
        'retained_discounts': retained_total,
        'total_revenue': revenue_total
    }

summary_with = print_scenario_summary(with_discount, "WITH DISCOUNT")
summary_no = print_scenario_summary(no_discount, "NO DISCOUNT")

# ================================================================
# Comparison
# ================================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

revenue_diff = summary_no['total_revenue'] - summary_with['total_revenue']

print(f"\nRevenue:")
print(f"  WITH DISCOUNT:  ${summary_with['total_revenue']:>15,.2f}")
print(f"  NO DISCOUNT:    ${summary_no['total_revenue']:>15,.2f}")
print(f"  Difference:     ${revenue_diff:>15,.2f}")

print(f"\nTarget Comparison:")
print(f"  Target:         ${1_043_000:>15,.2f}")
print(f"  Simulated:      ${summary_no['total_revenue']:>15,.2f}")
print(f"  Gap:            ${summary_no['total_revenue'] - 1_043_000:>15,.2f}")

# ================================================================
# Save results
# ================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

comparison_df = pd.DataFrame([summary_with, summary_no])
output_csv = os.path.join(OUTPUT_DIR, '10.7_FY2025_calibrated_comparison_summary.csv')
comparison_df.to_csv(output_csv, index=False)
print(f"✓ {output_csv}")

# Save detailed results as CSV files (much faster than Excel)
print("Saving detailed results as CSV files...")
with_discount.to_csv(os.path.join(OUTPUT_DIR, '10.7_with_discount_details.csv'), index=False)
print(f"✓ 10.7_with_discount_details.csv")

no_discount.to_csv(os.path.join(OUTPUT_DIR, '10.7_no_discount_details.csv'), index=False)
print(f"✓ 10.7_no_discount_details.csv")

decile_seasonal_df.to_csv(os.path.join(OUTPUT_DIR, '10.7_decile_seasonal_adjustments.csv'), index=False)
print(f"✓ 10.7_decile_seasonal_adjustments.csv")

# ================================================================
# Visualizations
# ================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

TARGET_REVENUE = 1_043_000

fig, ax = plt.subplots(figsize=(16, 8))

# Aggregate by payment month
with_discount_monthly = with_discount.groupby(with_discount['payment_date'].dt.to_period('M')).agg({
    'credit_card_revenue': 'sum'
}).reset_index()
with_discount_monthly['payment_date'] = with_discount_monthly['payment_date'].dt.to_timestamp()
with_discount_monthly['cumulative'] = with_discount_monthly['credit_card_revenue'].cumsum()

no_discount_monthly = no_discount.groupby(no_discount['payment_date'].dt.to_period('M')).agg({
    'interest_charged': 'sum',
    'retained_discounts': 'sum',
    'credit_card_revenue': 'sum'
}).reset_index()
no_discount_monthly['payment_date'] = no_discount_monthly['payment_date'].dt.to_timestamp()
no_discount_monthly['cumulative_interest'] = no_discount_monthly['interest_charged'].cumsum()
no_discount_monthly['cumulative_retained'] = no_discount_monthly['retained_discounts'].cumsum()
no_discount_monthly['cumulative_total'] = no_discount_monthly['credit_card_revenue'].cumsum()

# Plot
ax.plot(with_discount_monthly['payment_date'], with_discount_monthly['cumulative'], 
        marker='o', linewidth=3, label='With Discount', color='#70AD47', markersize=8)

ax.fill_between(no_discount_monthly['payment_date'], 0, no_discount_monthly['cumulative_interest'],
                 alpha=0.3, color='#4472C4', label='No Discount - Interest')
ax.fill_between(no_discount_monthly['payment_date'], no_discount_monthly['cumulative_interest'], 
                 no_discount_monthly['cumulative_total'],
                 alpha=0.3, color='#8FAADC', label='No Discount - Retained')

ax.plot(no_discount_monthly['payment_date'], no_discount_monthly['cumulative_total'], 
        marker='s', linewidth=3, label='No Discount (Total)', color='#4472C4', markersize=8)

ax.axhline(y=TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target (${TARGET_REVENUE:,.0f})', alpha=0.8)

ax.set_title(f'FY2025 Cumulative Revenue (Calibrated Baseline)\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Payment Month', fontsize=14)
ax.set_ylabel('Cumulative Revenue ($)', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, '10.7_cumulative_revenue_calibrated.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✓ {viz_path}")
plt.close()

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
# print(f"Total time: {elapsed_with + elapsed_no:.2f} seconds")
print(f"\nCalibrated baseline: {calibrated_baseline*100:.2f}%")
print(f"Simulated revenue: ${summary_no['total_revenue']:,.2f}")
print(f"Target revenue: $1,043,000")
print(f"Difference: ${summary_no['total_revenue'] - 1_043_000:,.2f}")
print("="*70)