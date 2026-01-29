'''
Docstring for 10.10_estimate_historic_revenue_with_bundled_invoices - Monte Carlo with Statement-Level Analysis

FIXED VERSION: Re-assigns deciles at STATEMENT level after aggregation

This script runs Monte Carlo simulations using STATEMENT-LEVEL data instead of
individual invoices. Statements are the natural unit of payment behavior since
customers pay statements, not individual invoices.

Key differences from 10.8:
- Uses bundled invoice data from script 9.7 (already aggregated to statements)
- RE-ASSIGNS deciles based on STATEMENT values (not invoice values)
- Treats each statement as a single payment unit
- Applies late payment probability at statement level
- More realistic modeling of customer payment behavior

Monte Carlo simulation approach:
- Run N simulations with different random seeds
- Each simulation uses calibrated baseline and seasonal adjustments
- Track distribution of outcomes for statistical inference
- Calculate confidence intervals and error margins

Inputs:
- data_cleaning/ats_grouped_transformed_with_discounts_bundled.csv
- data_cleaning/invoice_grouped_transformed_with_discounts_bundled.csv
- visualisations/09.6_reconstructed_late_payment_rates.csv (seasonal adjustment factors)
- visualisations/10.6_calibrated_baseline_late_rate.csv (calibrated November baseline)
- payment_profile/decile_payment_profile.pkl (for cd level distributions)

Outputs:
- visualisations/10.10_monte_carlo_summary_statistics_bundled.csv
- visualisations/10.10_simulation_results_distribution_bundled.csv
- visualisations/10.10_revenue_distribution_histogram_bundled.png
- visualisations/10.10_confidence_intervals_bundled.png
- visualisations/10.10_scenario_comparison_boxplot_bundled.png
- visualisations/10.10_monthly_revenue_uncertainty_bundled.png
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path
import time
from scipy import stats

# ================================================================
# CONFIGURATION
# ================================================================
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
data_cleaning_dir = base_dir / "data_cleaning"
visualisations_dir = base_dir / "visualisations"

# SIMULATION PARAMETERS
N_SIMULATIONS = 100  # Adjustable - increase for more precision, decrease for speed
CONFIDENCE_LEVEL = 0.95  # For confidence intervals

ANNUAL_INTEREST_RATE = 0.2395
BASE_RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30

FY2025_START = pd.Timestamp("2024-07-01")
FY2025_END = pd.Timestamp("2025-06-30")

OUTPUT_DIR = visualisations_dir
TARGET_REVENUE = 1_043_000

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
print("MONTE CARLO SIMULATION WITH BUNDLED STATEMENTS (FIXED)")
print("="*70)
print(f"Running {N_SIMULATIONS} simulations")
print(f"Confidence level: {CONFIDENCE_LEVEL*100}%")
print(f"Analysis unit: STATEMENTS (bundled from script 9.7)")
print(f"FIX: Re-assigns deciles based on STATEMENT values")

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
# Load BUNDLED invoices
# ================================================================
print("\n" + "="*70)
print("LOADING BUNDLED INVOICE DATA FROM 9.7")
print("="*70)

# Check if bundled files exist
bundled_ats_path = data_cleaning_dir / '9.7_ats_grouped_transformed_with_discounts_bundled.csv'
bundled_invoice_path = data_cleaning_dir / '9.7_invoice_grouped_transformed_with_discounts_bundled.csv'

if not bundled_ats_path.exists() or not bundled_invoice_path.exists():
    print("ERROR: Bundled invoice files not found!")
    print("Please run 9.7_bundle_invoices_to_statements.py first")
    print(f"Expected files:")
    print(f"  - {bundled_ats_path}")
    print(f"  - {bundled_invoice_path}")
    exit(1)

ats_grouped = pd.read_csv(bundled_ats_path)
invoice_grouped = pd.read_csv(bundled_invoice_path)

print(f"✓ Loaded {len(ats_grouped):,} ATS invoices (bundled)")
print(f"✓ Loaded {len(invoice_grouped):,} Invoice invoices (bundled)")

# Verify statement_id column exists
required_columns = ['statement_id']
for df, name in [(ats_grouped, 'ATS'), (invoice_grouped, 'Invoice')]:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"ERROR: {name} data missing columns: {missing}")
        print("Please re-run 9.7_bundle_invoices_to_statements.py")
        exit(1)

print(f"✓ Verified: statement_id column present")

# Combine datasets
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"\nCombined data:")
print(f"  Total invoices: {len(combined_df):,}")
print(f"  Total unique statements: {combined_df['statement_id'].nunique():,}")

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
fy2025_invoices = combined_df[
    (combined_df['invoice_period'] >= FY2025_START) & 
    (combined_df['invoice_period'] <= FY2025_END)
].copy()

print(f"\nFY2025 data:")
print(f"  Invoices: {len(fy2025_invoices):,}")
print(f"  Statements: {fy2025_invoices['statement_id'].nunique():,}")

# ================================================================
# Aggregate invoices to STATEMENT level (WITHOUT keeping invoice-level decile)
# ================================================================
print("\n" + "="*70)
print("AGGREGATING TO STATEMENT LEVEL")
print("="*70)

# CRITICAL FIX: Aggregate WITHOUT 'decile' in groupby
# We'll re-assign deciles based on statement values
statements_fy2025 = fy2025_invoices.groupby(['statement_id', 'invoice_period']).agg({
    'total_discounted_price': 'sum',
    'total_undiscounted_price': 'sum',
    'discount_amount': 'sum',
    'invoice_id': 'count',
    'customer_type': 'first'
}).reset_index()

statements_fy2025.columns = [
    'statement_id', 'invoice_period',
    'statement_discounted_price', 'statement_undiscounted_price',
    'statement_discount_amount', 'n_invoices_in_statement', 'customer_type'
]

print(f"✓ Aggregated to {len(statements_fy2025):,} statements for FY2025")
print(f"  Average invoices per statement: {statements_fy2025['n_invoices_in_statement'].mean():.2f}")
print(f"  Average statement value (discounted): ${statements_fy2025['statement_discounted_price'].mean():,.2f}")
print(f"  Average statement value (undiscounted): ${statements_fy2025['statement_undiscounted_price'].mean():,.2f}")

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

# ================================================================
# CRITICAL FIX: Re-assign deciles based on STATEMENT values
# ================================================================
print("\n" + "="*70)
print("RE-ASSIGNING DECILES BASED ON STATEMENT VALUES")
print("="*70)

statements_fy2025 = statements_fy2025.sort_values('statement_undiscounted_price').reset_index(drop=True)
statements_fy2025['decile'] = pd.qcut(
    statements_fy2025['statement_undiscounted_price'], 
    q=n_deciles, 
    labels=False,
    duplicates='drop'
)

print(f"✓ Re-assigned deciles based on statement values")
print(f"\nStatement value distribution by decile:")
for i in range(n_deciles):
    decile_data = statements_fy2025[statements_fy2025['decile'] == i]
    if len(decile_data) > 0:
        print(f"  Decile {i}: ${decile_data['statement_undiscounted_price'].mean():>10,.2f} avg, "
              f"${decile_data['statement_undiscounted_price'].median():>10,.2f} median, "
              f"{len(decile_data):>6,} statements")

# ================================================================
# PRE-COMPUTE all values for VECTORIZED simulation
# ================================================================
print("\n" + "="*70)
print("PRE-COMPUTING VALUES FOR VECTORIZED SIMULATION")
print("="*70)

# Create year-month column
statements_fy2025['year_month'] = list(zip(
    statements_fy2025['invoice_period'].dt.year,
    statements_fy2025['invoice_period'].dt.month
))

# Map seasonal factors
statements_fy2025['seasonal_factor'] = statements_fy2025['year_month'].map(
    seasonal_adjustment_factors
).fillna(1.0)

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

statements_fy2025['decile_snapshot_rate'] = statements_fy2025['decile'].map(decile_snapshot_rates)
statements_fy2025['decile_calibrated_rate'] = statements_fy2025['decile'].map(decile_calibrated_rates)

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

statements_fy2025['expected_days_overdue'] = statements_fy2025['decile'].map(decile_expected_days)

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

print(f"✓ Pre-computed all values for {len(statements_fy2025):,} statements")

# Print calibrated rates by decile
print(f"\nCalibrated late payment rates by decile:")
for i in range(n_deciles):
    snapshot = decile_snapshot_rates.get(i, 0.02)
    calibrated = decile_calibrated_rates.get(i, 0.02)
    print(f"  Decile {i}: {snapshot*100:>5.2f}% (snapshot) -> {calibrated*100:>5.2f}% (calibrated)")

# ================================================================
# VECTORIZED simulation function - SHARED late status
# ================================================================
def simulate_both_scenarios_with_shared_late_status(statements_df, seed):
    """
    Simulate BOTH scenarios using the SAME late payment assignments at STATEMENT level
    
    Revenue is attributed to the INVOICE MONTH, not payment month.
    """
    np.random.seed(seed)
    
    # ================================================================
    # STEP 1: Determine which STATEMENTS are late (SAME FOR BOTH)
    # ================================================================
    df = statements_df.copy()
    
    # Calculate adjusted late rate (VECTORIZED)
    df['adjusted_late_rate'] = df['decile_calibrated_rate'] * df['seasonal_factor']
    df['adjusted_late_rate'] = np.minimum(df['adjusted_late_rate'], 1.0)
    
    # Determine late status (VECTORIZED random draw) - SHARED
    random_draws = np.random.random(len(df))
    df['is_late'] = random_draws < df['adjusted_late_rate']
    
    # Sample cd levels for late statements (SHARED)
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
    
    # ================================================================
    # STEP 2: Create WITH DISCOUNT scenario
    # ================================================================
    with_discount = df.copy()
    
    # Principal = discounted price (at STATEMENT level)
    with_discount['principal_amount'] = with_discount['statement_discounted_price']
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
    
    # Principal = undiscounted price (at STATEMENT level)
    no_discount['principal_amount'] = no_discount['statement_undiscounted_price']
    
    # Retained discounts = discount amount for late statements only
    no_discount['retained_discounts'] = np.where(
        no_discount['is_late'], 
        no_discount['statement_discount_amount'], 
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
    
    # ================================================================
    # STEP 4: Calculate summary statistics
    # ================================================================
    with_discount_results = {
        'total_revenue': with_discount['credit_card_revenue'].sum(),
        'interest_revenue': with_discount['interest_charged'].sum(),
        'retained_discounts': with_discount['retained_discounts'].sum(),
        'n_late': with_discount['is_late'].sum(),
        'pct_late': with_discount['is_late'].sum() / len(with_discount) * 100,
        'avg_days_overdue': with_discount[with_discount['is_late']]['days_overdue'].mean() if with_discount['is_late'].sum() > 0 else 0
    }
    
    no_discount_results = {
        'total_revenue': no_discount['credit_card_revenue'].sum(),
        'interest_revenue': no_discount['interest_charged'].sum(),
        'retained_discounts': no_discount['retained_discounts'].sum(),
        'n_late': no_discount['is_late'].sum(),
        'pct_late': no_discount['is_late'].sum() / len(no_discount) * 100,
        'avg_days_overdue': no_discount[no_discount['is_late']]['days_overdue'].mean() if no_discount['is_late'].sum() > 0 else 0
    }
    
    # ================================================================
    # STEP 5: Monthly aggregation by INVOICE MONTH (not payment month)
    # ================================================================
    monthly_with = with_discount.groupby(with_discount['invoice_period'].dt.to_period('M')).agg({
        'credit_card_revenue': 'sum'
    }).reset_index()
    monthly_with['invoice_period'] = monthly_with['invoice_period'].dt.to_timestamp()
    monthly_with['cumulative'] = monthly_with['credit_card_revenue'].cumsum()
    monthly_with.rename(columns={'invoice_period': 'month'}, inplace=True)
    
    monthly_no = no_discount.groupby(no_discount['invoice_period'].dt.to_period('M')).agg({
        'credit_card_revenue': 'sum'
    }).reset_index()
    monthly_no['invoice_period'] = monthly_no['invoice_period'].dt.to_timestamp()
    monthly_no['cumulative'] = monthly_no['credit_card_revenue'].cumsum()
    monthly_no.rename(columns={'invoice_period': 'month'}, inplace=True)
    
    monthly_details = {
        'with_discount': monthly_with,
        'no_discount': monthly_no
    }
    
    return with_discount_results, no_discount_results, monthly_details

# ================================================================
# Run Monte Carlo simulation
# ================================================================
print("\n" + "="*70)
print(f"RUNNING {N_SIMULATIONS} MONTE CARLO SIMULATIONS")
print("="*70)

all_results_with = []
all_results_no = []
all_monthly_with = []
all_monthly_no = []

start_time = time.time()

for i in range(N_SIMULATIONS):
    seed = BASE_RANDOM_SEED + i
    
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Simulation {i+1}/{N_SIMULATIONS}...", end='\r')
    
    with_results, no_results, monthly = simulate_both_scenarios_with_shared_late_status(
        statements_fy2025, seed
    )
    
    with_results['simulation'] = i
    no_results['simulation'] = i
    
    all_results_with.append(with_results)
    all_results_no.append(no_results)
    
    # Track monthly data
    monthly_with = monthly['with_discount'].copy()
    monthly_with['simulation'] = i
    all_monthly_with.append(monthly_with)
    
    monthly_no = monthly['no_discount'].copy()
    monthly_no['simulation'] = i
    all_monthly_no.append(monthly_no)

elapsed = time.time() - start_time
print(f"\n✓ Completed {N_SIMULATIONS} simulations in {elapsed:.2f} seconds ({elapsed/N_SIMULATIONS:.2f}s per simulation)")

# ================================================================
# Convert to DataFrames
# ================================================================
results_with_df = pd.DataFrame(all_results_with)
results_no_df = pd.DataFrame(all_results_no)

monthly_with_df = pd.concat(all_monthly_with, ignore_index=True)
monthly_no_df = pd.concat(all_monthly_no, ignore_index=True)

# ================================================================
# Calculate summary statistics
# ================================================================
print("\n" + "="*70)
print("CALCULATING SUMMARY STATISTICS")
print("="*70)

def calculate_summary_stats(df, scenario_name):
    """Calculate mean, median, std, CI for all metrics"""
    stats_dict = {
        'scenario': scenario_name,
        'n_simulations': len(df),
    }
    
    for col in ['total_revenue', 'interest_revenue', 'retained_discounts', 'pct_late', 'avg_days_overdue']:
        values = df[col].values
        
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values, ddof=1)
        
        # Calculate confidence interval
        ci = stats.t.interval(
            CONFIDENCE_LEVEL,
            len(values) - 1,
            loc=mean_val,
            scale=stats.sem(values)
        )
        
        stats_dict[f'{col}_mean'] = mean_val
        stats_dict[f'{col}_median'] = median_val
        stats_dict[f'{col}_std'] = std_val
        stats_dict[f'{col}_ci_lower'] = ci[0]
        stats_dict[f'{col}_ci_upper'] = ci[1]
        stats_dict[f'{col}_min'] = np.min(values)
        stats_dict[f'{col}_max'] = np.max(values)
    
    return stats_dict

summary_with = calculate_summary_stats(results_with_df, 'with_discount')
summary_no = calculate_summary_stats(results_no_df, 'no_discount')

summary_df = pd.DataFrame([summary_with, summary_no])

# ================================================================
# Print summary
# ================================================================
print("\n" + "="*70)
print("MONTE CARLO RESULTS SUMMARY (STATEMENT-BASED, FIXED)")
print("="*70)

def print_scenario_stats(stats_dict, scenario_name):
    print(f"\n{scenario_name.upper()}")
    print("-" * 70)
    
    print(f"Total Revenue:")
    print(f"  Mean:      ${stats_dict['total_revenue_mean']:>15,.2f}")
    print(f"  Median:    ${stats_dict['total_revenue_median']:>15,.2f}")
    print(f"  Std Dev:   ${stats_dict['total_revenue_std']:>15,.2f}")
    print(f"  {CONFIDENCE_LEVEL*100:.0f}% CI:   ${stats_dict['total_revenue_ci_lower']:>15,.2f} - ${stats_dict['total_revenue_ci_upper']:>15,.2f}")
    print(f"  Range:     ${stats_dict['total_revenue_min']:>15,.2f} - ${stats_dict['total_revenue_max']:>15,.2f}")
    
    print(f"\nInterest Revenue:")
    print(f"  Mean:      ${stats_dict['interest_revenue_mean']:>15,.2f}")
    print(f"  {CONFIDENCE_LEVEL*100:.0f}% CI:   ${stats_dict['interest_revenue_ci_lower']:>15,.2f} - ${stats_dict['interest_revenue_ci_upper']:>15,.2f}")
    
    print(f"\nRetained Discounts:")
    print(f"  Mean:      ${stats_dict['retained_discounts_mean']:>15,.2f}")
    print(f"  {CONFIDENCE_LEVEL*100:.0f}% CI:   ${stats_dict['retained_discounts_ci_lower']:>15,.2f} - ${stats_dict['retained_discounts_ci_upper']:>15,.2f}")
    
    print(f"\nLate Payment Rate (of STATEMENTS):")
    print(f"  Mean:      {stats_dict['pct_late_mean']:>15.2f}%")
    print(f"  {CONFIDENCE_LEVEL*100:.0f}% CI:   {stats_dict['pct_late_ci_lower']:>15.2f}% - {stats_dict['pct_late_ci_upper']:>15.2f}%")

print_scenario_stats(summary_with, "WITH DISCOUNT")
print_scenario_stats(summary_no, "NO DISCOUNT")

# ================================================================
# Comparison
# ================================================================
print("\n" + "="*70)
print("SCENARIO COMPARISON")
print("="*70)

revenue_diff_mean = summary_no['total_revenue_mean'] - summary_with['total_revenue_mean']
revenue_diff_std = np.std(results_no_df['total_revenue'] - results_with_df['total_revenue'], ddof=1)

print(f"\nRevenue Difference (No Discount - With Discount):")
print(f"  Mean:      ${revenue_diff_mean:>15,.2f}")
print(f"  Std Dev:   ${revenue_diff_std:>15,.2f}")

print(f"\nTarget Comparison (No Discount scenario):")
print(f"  Target:         ${TARGET_REVENUE:>15,.2f}")
print(f"  Simulated Mean: ${summary_no['total_revenue_mean']:>15,.2f}")
print(f"  Gap:            ${summary_no['total_revenue_mean'] - TARGET_REVENUE:>15,.2f}")
print(f"  Gap as % of Target: {(summary_no['total_revenue_mean'] - TARGET_REVENUE) / TARGET_REVENUE * 100:.2f}%")

# Calculate probability of exceeding target
prob_exceed_target = (results_no_df['total_revenue'] >= TARGET_REVENUE).sum() / N_SIMULATIONS * 100
print(f"\nProbability of meeting/exceeding target: {prob_exceed_target:.1f}%")

# ================================================================
# Save results
# ================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Summary statistics
summary_output = os.path.join(OUTPUT_DIR, '10.10_monte_carlo_summary_statistics_bundled.csv')
summary_df.to_csv(summary_output, index=False)
print(f"✓ {summary_output}")

# All simulation results
results_with_df['scenario'] = 'with_discount'
results_no_df['scenario'] = 'no_discount'
all_results = pd.concat([results_with_df, results_no_df], ignore_index=True)

results_output = os.path.join(OUTPUT_DIR, '10.10_simulation_results_distribution_bundled.csv')
all_results.to_csv(results_output, index=False)
print(f"✓ {results_output}")

# ================================================================
# VISUALIZATIONS
# ================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. Revenue Distribution Histogram
print("  Creating revenue distribution histogram...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# With discount
ax1.hist(results_with_df['total_revenue'], bins=30, alpha=0.7, color='#70AD47', edgecolor='black')
ax1.axvline(summary_with['total_revenue_mean'], color='red', linestyle='--', linewidth=2, 
            label=f"Mean: ${summary_with['total_revenue_mean']:,.0f}")
ax1.axvline(summary_with['total_revenue_ci_lower'], color='orange', linestyle=':', linewidth=2, 
            label=f"{CONFIDENCE_LEVEL*100:.0f}% CI")
ax1.axvline(summary_with['total_revenue_ci_upper'], color='orange', linestyle=':', linewidth=2)
ax1.set_title('With Discount Revenue Distribution\n(Statement-Based, Fixed)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Total Revenue ($)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# No discount
ax2.hist(results_no_df['total_revenue'], bins=30, alpha=0.7, color='#4472C4', edgecolor='black')
ax2.axvline(summary_no['total_revenue_mean'], color='red', linestyle='--', linewidth=2, 
            label=f"Mean: ${summary_no['total_revenue_mean']:,.0f}")
ax2.axvline(summary_no['total_revenue_ci_lower'], color='orange', linestyle=':', linewidth=2, 
            label=f"{CONFIDENCE_LEVEL*100:.0f}% CI")
ax2.axvline(summary_no['total_revenue_ci_upper'], color='orange', linestyle=':', linewidth=2)
ax2.axvline(TARGET_REVENUE, color='green', linestyle='-', linewidth=2.5, 
            label=f"Target: ${TARGET_REVENUE:,.0f}", alpha=0.8)
ax2.set_title('No Discount Revenue Distribution\n(Statement-Based, Fixed)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Total Revenue ($)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
hist_output = os.path.join(OUTPUT_DIR, '10.10_revenue_distribution_histogram_bundled.png')
plt.savefig(hist_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {hist_output}")
plt.close()

# 2. Confidence Intervals Comparison
print("  Creating confidence intervals visualization...")
fig, ax = plt.subplots(figsize=(12, 8))

scenarios = ['With Discount', 'No Discount']
means = [summary_with['total_revenue_mean'], summary_no['total_revenue_mean']]
ci_lower = [summary_with['total_revenue_ci_lower'], summary_no['total_revenue_ci_lower']]
ci_upper = [summary_with['total_revenue_ci_upper'], summary_no['total_revenue_ci_upper']]
errors_lower = [m - l for m, l in zip(means, ci_lower)]
errors_upper = [u - m for m, u in zip(means, ci_upper)]

y_pos = np.arange(len(scenarios))
colors = ['#70AD47', '#4472C4']

ax.barh(y_pos, means, color=colors, alpha=0.6, height=0.5)
ax.errorbar(means, y_pos, xerr=[errors_lower, errors_upper], 
            fmt='none', ecolor='black', capsize=10, capthick=2, linewidth=2)

# Add mean value labels
for i, (mean, scenario) in enumerate(zip(means, scenarios)):
    ax.text(mean, i, f'  ${mean:,.0f}', va='center', fontsize=11, fontweight='bold')

ax.axvline(TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target: ${TARGET_REVENUE:,.0f}', alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(scenarios, fontsize=12)
ax.set_xlabel('Total Revenue ($)', fontsize=14)
ax.set_title(f'Revenue Comparison with {CONFIDENCE_LEVEL*100:.0f}% Confidence Intervals\n({N_SIMULATIONS} simulations, Statement-Based, Fixed)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
ci_output = os.path.join(OUTPUT_DIR, '10.10_confidence_intervals_bundled.png')
plt.savefig(ci_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {ci_output}")
plt.close()

# 3. Boxplot Comparison
print("  Creating boxplot comparison...")
fig, ax = plt.subplots(figsize=(12, 8))

data_to_plot = [results_with_df['total_revenue'], results_no_df['total_revenue']]
box_colors = ['#70AD47', '#4472C4']

bp = ax.boxplot(data_to_plot, tick_labels=scenarios, patch_artist=True,
                showmeans=True, meanline=True,
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(color='blue', linewidth=2, linestyle='--'))

for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(TARGET_REVENUE, color='green', linestyle='--', linewidth=2.5, 
           label=f'Target: ${TARGET_REVENUE:,.0f}', alpha=0.8)

ax.set_ylabel('Total Revenue ($)', fontsize=14)
ax.set_title(f'Revenue Distribution Comparison\n({N_SIMULATIONS} simulations, Statement-Based, Fixed)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
boxplot_output = os.path.join(OUTPUT_DIR, '10.10_scenario_comparison_boxplot_bundled.png')
plt.savefig(boxplot_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {boxplot_output}")
plt.close()

# 4. Monthly Revenue with Uncertainty Bands (Overlayed)
print("  Creating monthly revenue uncertainty bands...")
fig, ax = plt.subplots(figsize=(16, 8))

# Calculate monthly percentiles
def calculate_monthly_percentiles(monthly_df, percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]):
    """Calculate percentiles for each month across simulations"""
    monthly_stats = monthly_df.groupby('month')['cumulative'].quantile(percentiles).unstack()
    monthly_stats.columns = [int(p*100) for p in percentiles]  # Convert back to % for labeling
    return monthly_stats

monthly_with_stats = calculate_monthly_percentiles(monthly_with_df)
monthly_no_stats = calculate_monthly_percentiles(monthly_no_df)

# Plot WITH DISCOUNT uncertainty bands
ax.fill_between(monthly_with_stats.index, monthly_with_stats[5], monthly_with_stats[95],
                alpha=0.15, color='#70AD47', label='With Discount 90% CI')
ax.fill_between(monthly_with_stats.index, monthly_with_stats[25], monthly_with_stats[75],
                alpha=0.25, color='#70AD47', label='With Discount 50% CI')

# Plot NO DISCOUNT uncertainty bands
ax.fill_between(monthly_no_stats.index, monthly_no_stats[5], monthly_no_stats[95],
                alpha=0.15, color='#4472C4', label='No Discount 90% CI')
ax.fill_between(monthly_no_stats.index, monthly_no_stats[25], monthly_no_stats[75],
                alpha=0.25, color='#4472C4', label='No Discount 50% CI')

# Plot median lines on top
ax.plot(monthly_with_stats.index, monthly_with_stats[50], 
        color='#70AD47', linewidth=3, label='With Discount (Median)', marker='o', markersize=8)
ax.plot(monthly_no_stats.index, monthly_no_stats[50], 
        color='#4472C4', linewidth=3, label='No Discount (Median)', marker='s', markersize=8)

# Add target line
ax.axhline(TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target: ${TARGET_REVENUE:,.0f}', alpha=0.8)

ax.set_title(f'FY2025 Cumulative Revenue with Uncertainty Bands (By Invoice Month)\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")} ({N_SIMULATIONS} simulations)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Invoice Month', fontsize=14)
ax.set_ylabel('Cumulative Revenue ($)', fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
monthly_output = os.path.join(OUTPUT_DIR, '10.10_monthly_revenue_uncertainty_bundled.png')
plt.savefig(monthly_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {monthly_output}")
plt.close()

# ================================================================
# Final Summary
# ================================================================
print("\n" + "="*70)
print("MONTE CARLO SIMULATION COMPLETE (STATEMENT-BASED, FIXED)")
print("="*70)
print(f"\nSimulations run: {N_SIMULATIONS}")
print(f"Confidence level: {CONFIDENCE_LEVEL*100}%")
print(f"Total execution time: {elapsed:.2f} seconds")
print(f"Analysis unit: STATEMENTS ({len(statements_fy2025):,} statements)")

print(f"\nFIX APPLIED:")
print(f"  ✓ Deciles re-assigned based on STATEMENT values (not invoice values)")
print(f"  ✓ This corrects the systematic bias from invoice-level decile assignment")

print(f"\nNO DISCOUNT scenario:")
print(f"  Expected revenue: ${summary_no['total_revenue_mean']:,.2f}")
print(f"  {CONFIDENCE_LEVEL*100:.0f}% CI: ${summary_no['total_revenue_ci_lower']:,.2f} - ${summary_no['total_revenue_ci_upper']:,.2f}")
print(f"  Probability of meeting target: {prob_exceed_target:.1f}%")

print(f"\nRevenue improvement (No Discount vs With Discount):")
print(f"  Mean difference: ${revenue_diff_mean:,.2f}")
print(f"  Std deviation: ${revenue_diff_std:,.2f}")

print("\n" + "="*70)
print("KEY DIFFERENCE FROM ORIGINAL 10.10:")
print("  ORIGINAL: Kept invoice-level decile assignment → systematic bias")
print("  FIXED: Re-assigned deciles at statement level → correct calibration")
print("="*70)