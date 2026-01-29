'''
10.10_MC_estimate_historic_revenue_with_rebundling - Monte Carlo with Rebundling

This script runs Monte Carlo simulations with REBUNDLING at each iteration.
Each simulation creates fresh statement groupings, introducing bundling uncertainty.

Key features:
- Loads UNBUNDLED invoice data from 09.1
- Rebundles invoices at EACH iteration (realistic uncertainty)
- Uses calibrated baseline from 10.6
- Applies seasonal adjustments
- Statement-level payment simulation

Monte Carlo simulation approach:
- Run N simulations with different random seeds
- Each simulation rebundles invoices into statements
- Uses calibrated baseline and seasonal adjustments
- Track distribution of outcomes for statistical inference
- Calculate confidence intervals and error margins

Inputs:
- data_cleaning/ats_grouped_transformed_with_discounts.csv (UNBUNDLED from 09.1)
- data_cleaning/invoice_grouped_transformed_with_discounts.csv (UNBUNDLED from 09.1)
- visualisations/09.6_reconstructed_late_payment_rates.csv (seasonal adjustment factors)
- visualisations/10.6_calibrated_baseline_late_rate.csv (calibrated November baseline)
- payment_profile/decile_payment_profile.pkl (for cd level distributions)

Outputs:
- visualisations/10.10_MC_monte_carlo_summary_statistics_rebundled.csv
- visualisations/10.10_MC_simulation_results_distribution_rebundled.csv
- visualisations/10.10_MC_revenue_distribution_histogram_rebundled.png
- visualisations/10.10_MC_confidence_intervals_rebundled.png
- visualisations/10.10_MC_scenario_comparison_boxplot_rebundled.png
- visualisations/10.10_MC_monthly_revenue_uncertainty_rebundled.png
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

# Bundling parameters (from 9.7)
N_ACTIVE_USERS = 5920

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
print("MONTE CARLO SIMULATION WITH REBUNDLING AT EACH ITERATION")
print("="*70)
print(f"Running {N_SIMULATIONS} simulations")
print(f"Confidence level: {CONFIDENCE_LEVEL*100}%")
print(f"Analysis unit: STATEMENTS (rebundled each iteration)")
print(f"Bundling uncertainty: YES")

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
# Load UNBUNDLED invoices from 09.1
# ================================================================
print("\n" + "="*70)
print("LOADING UNBUNDLED INVOICE DATA FROM 09.1")
print("="*70)

unbundled_ats_path = data_cleaning_dir / 'ats_grouped_transformed_with_discounts.csv'
unbundled_invoice_path = data_cleaning_dir / 'invoice_grouped_transformed_with_discounts.csv'

if not unbundled_ats_path.exists() or not unbundled_invoice_path.exists():
    print("ERROR: Unbundled invoice files not found!")
    print("Please run 09.1_data_engineering_transformed.py first")
    print(f"Expected files:")
    print(f"  - {unbundled_ats_path}")
    print(f"  - {unbundled_invoice_path}")
    exit(1)

ats_grouped = pd.read_csv(unbundled_ats_path)
invoice_grouped = pd.read_csv(unbundled_invoice_path)

print(f"✓ Loaded {len(ats_grouped):,} ATS invoices (UNBUNDLED)")
print(f"✓ Loaded {len(invoice_grouped):,} Invoice invoices (UNBUNDLED)")

# Add customer type
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'

# Combine datasets
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"\nCombined data:")
print(f"  Total invoices: {len(combined_df):,}")

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

# Pre-compute decile snapshot rates
decile_snapshot_rates = {}
for i in range(n_deciles):
    decile_key = f'decile_{i}'
    if decile_key in decile_profile['deciles']:
        snapshot_rate = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
    else:
        snapshot_rate = 0.02
    decile_snapshot_rates[i] = snapshot_rate

# Pre-compute cd distributions
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
            decile_cd_distributions[i] = ([3], [1.0])
    else:
        decile_cd_distributions[i] = ([3], [1.0])

print(f"✓ Pre-computed decile rates and cd distributions")

# ================================================================
# Define bundling function
# ================================================================
def bundle_invoices_to_statements(invoices_df, seed):
    """
    Bundle invoices into statements using random assignment with Poisson distribution
    
    Args:
        invoices_df: DataFrame with invoice data
        seed: Random seed for this iteration
    
    Returns:
        DataFrame with statement-level aggregates
    """
    np.random.seed(seed)
    
    df = invoices_df.copy()
    df['year_month'] = df['invoice_period'].dt.to_period('M')
    
    all_statements = []
    
    for year_month in df['year_month'].unique():
        month_data = df[df['year_month'] == year_month].copy()
        n_invoices_this_month = len(month_data)
        
        # Calculate expected number of statements
        mean_invoices_per_statement = n_invoices_this_month / N_ACTIVE_USERS
        n_statements_this_month = max(1, int(n_invoices_this_month / mean_invoices_per_statement))
        
        # Randomly shuffle invoices
        month_data = month_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Generate statement sizes using Poisson distribution
        statement_sizes = np.random.poisson(mean_invoices_per_statement, n_statements_this_month)
        statement_sizes = np.maximum(statement_sizes, 1)
        
        # Adjust to match total invoice count
        total_assigned = statement_sizes.sum()
        if total_assigned > n_invoices_this_month:
            statement_sizes = (statement_sizes * n_invoices_this_month / total_assigned).astype(int)
            statement_sizes = np.maximum(statement_sizes, 1)
        
        # Handle remaining invoices
        remaining = n_invoices_this_month - statement_sizes.sum()
        if remaining > 0:
            random_indices = np.random.choice(len(statement_sizes), size=remaining, replace=True)
            for idx in random_indices:
                statement_sizes[idx] += 1
        elif remaining < 0:
            for i in range(abs(remaining)):
                if statement_sizes[-1] > 1:
                    statement_sizes[-1] -= 1
        
        # Create statement IDs
        statement_ids = []
        for stmt_idx, size in enumerate(statement_sizes):
            statement_ids.extend([f"{year_month}_stmt_{stmt_idx:04d}"] * size)
        
        statement_ids = statement_ids[:n_invoices_this_month]
        month_data['statement_id'] = statement_ids
        
        all_statements.append(month_data)
    
    bundled_df = pd.concat(all_statements, ignore_index=True)
    
    # Aggregate to statement level
    statements = bundled_df.groupby('statement_id').agg({
        'invoice_period': 'first',
        'year_month': 'first',
        'total_discounted_price': 'sum',
        'total_undiscounted_price': 'sum',
        'discount_amount': 'sum',
        'customer_type': 'first',
        'invoice_id': 'count'
    }).reset_index()
    
    statements.columns = [
        'statement_id', 'invoice_period', 'year_month',
        'statement_discounted_price', 'statement_undiscounted_price',
        'statement_discount_amount', 'n_invoices_in_statement', 'customer_type'
    ]
    
    # Assign deciles based on statement value
    statements = statements.sort_values('statement_undiscounted_price').reset_index(drop=True)
    statements['decile'] = pd.qcut(
        statements['statement_undiscounted_price'],
        q=n_deciles,
        labels=False,
        duplicates='drop'
    )
    
    return statements

# ================================================================
# Simulation function with rebundling
# ================================================================
def simulate_both_scenarios_with_rebundling(invoices_df, seed):
    """
    Simulate BOTH scenarios with REBUNDLING at each iteration
    
    Steps:
    1. Bundle invoices into statements (unique to this iteration)
    2. Apply seasonal adjustments and calibrated rates
    3. Simulate payment behavior at statement level
    4. Calculate revenue for both scenarios
    """
    np.random.seed(seed)
    
    # ================================================================
    # STEP 1: BUNDLE INVOICES (unique to this iteration)
    # ================================================================
    statements = bundle_invoices_to_statements(invoices_df, seed)
    
    # ================================================================
    # STEP 2: Apply seasonal adjustments and calibrated rates
    # ================================================================
    # Create year-month column
    statements['year_month_tuple'] = list(zip(
        statements['invoice_period'].dt.year,
        statements['invoice_period'].dt.month
    ))
    
    # Map seasonal factors
    statements['seasonal_factor'] = statements['year_month_tuple'].map(
        seasonal_adjustment_factors
    ).fillna(1.0)
    
    # Get decile snapshot rates and scale to calibrated
    statements['decile_snapshot_rate'] = statements['decile'].map(decile_snapshot_rates)
    
    # Handle missing deciles
    missing_decile = statements['decile_snapshot_rate'].isna().sum()
    if missing_decile > 0:
        avg_rate = np.mean(list(decile_snapshot_rates.values()))
        statements['decile_snapshot_rate'].fillna(avg_rate, inplace=True)
    
    # Apply scaling factor to get calibrated rate
    statements['decile_calibrated_rate'] = statements['decile_snapshot_rate'] * scaling_factor
    
    # ================================================================
    # STEP 3: Determine which STATEMENTS are late (SHARED)
    # ================================================================
    # Calculate adjusted late rate (VECTORIZED)
    statements['adjusted_late_rate'] = statements['decile_calibrated_rate'] * statements['seasonal_factor']
    statements['adjusted_late_rate'] = np.minimum(statements['adjusted_late_rate'], 1.0)
    
    # Determine late status (VECTORIZED random draw) - SHARED
    random_draws = np.random.random(len(statements))
    statements['is_late'] = random_draws < statements['adjusted_late_rate']
    
    # ================================================================
    # STEP 4: Sample cd levels for late statements (SHARED)
    # ================================================================
    statements['cd_level'] = 0
    statements['days_overdue'] = 0.0
    
    for decile_num in statements['decile'].unique():
        if pd.isna(decile_num):
            continue
            
        decile_mask = (statements['decile'] == decile_num) & statements['is_late']
        n_late_in_decile = decile_mask.sum()
        
        if n_late_in_decile > 0:
            cd_levels, cd_probs = decile_cd_distributions.get(int(decile_num), ([3], [1.0]))
            sampled_cd = np.random.choice(cd_levels, size=n_late_in_decile, p=cd_probs)
            statements.loc[decile_mask, 'cd_level'] = sampled_cd
            
            # Map cd levels to days overdue
            days_map = pd.Series(sampled_cd).map(CD_TO_DAYS).fillna(90).values
            statements.loc[decile_mask, 'days_overdue'] = days_map
    
    statements['months_overdue'] = statements['days_overdue'] / 30
    
    # ================================================================
    # STEP 5: Create WITH DISCOUNT scenario
    # ================================================================
    with_discount = statements.copy()
    with_discount['principal_amount'] = with_discount['statement_discounted_price']
    with_discount['retained_discounts'] = 0
    
    daily_rate = ANNUAL_INTEREST_RATE / 365
    with_discount['interest_charged'] = (
        with_discount['principal_amount'] * daily_rate * with_discount['days_overdue']
    )
    with_discount['credit_card_revenue'] = (
        with_discount['interest_charged'] + with_discount['retained_discounts']
    )
    
    # ================================================================
    # STEP 6: Create NO DISCOUNT scenario
    # ================================================================
    no_discount = statements.copy()
    no_discount['principal_amount'] = no_discount['statement_undiscounted_price']
    
    # Retained discounts = discount amount for late statements only
    no_discount['retained_discounts'] = np.where(
        no_discount['is_late'], 
        no_discount['statement_discount_amount'], 
        0
    )
    
    no_discount['interest_charged'] = (
        no_discount['principal_amount'] * daily_rate * no_discount['days_overdue']
    )
    no_discount['credit_card_revenue'] = (
        no_discount['interest_charged'] + no_discount['retained_discounts']
    )
    
    # ================================================================
    # STEP 7: Calculate summary statistics
    # ================================================================
    with_discount_results = {
        'total_revenue': with_discount['credit_card_revenue'].sum(),
        'interest_revenue': with_discount['interest_charged'].sum(),
        'retained_discounts': with_discount['retained_discounts'].sum(),
        'n_late': with_discount['is_late'].sum(),
        'n_statements': len(with_discount),
        'pct_late': with_discount['is_late'].sum() / len(with_discount) * 100,
        'avg_days_overdue': with_discount[with_discount['is_late']]['days_overdue'].mean() if with_discount['is_late'].sum() > 0 else 0,
        'avg_statement_value': with_discount['statement_undiscounted_price'].mean()
    }
    
    no_discount_results = {
        'total_revenue': no_discount['credit_card_revenue'].sum(),
        'interest_revenue': no_discount['interest_charged'].sum(),
        'retained_discounts': no_discount['retained_discounts'].sum(),
        'n_late': no_discount['is_late'].sum(),
        'n_statements': len(no_discount),
        'pct_late': no_discount['is_late'].sum() / len(no_discount) * 100,
        'avg_days_overdue': no_discount[no_discount['is_late']]['days_overdue'].mean() if no_discount['is_late'].sum() > 0 else 0,
        'avg_statement_value': no_discount['statement_undiscounted_price'].mean()
    }
    
    # ================================================================
    # STEP 8: Monthly aggregation by INVOICE MONTH
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
# Run Monte Carlo simulation with rebundling
# ================================================================
print("\n" + "="*70)
print(f"RUNNING {N_SIMULATIONS} MONTE CARLO SIMULATIONS WITH REBUNDLING")
print("="*70)

all_results_with = []
all_results_no = []
all_monthly_with = []
all_monthly_no = []

start_time = time.time()

for i in range(N_SIMULATIONS):
    seed = BASE_RANDOM_SEED + i
    
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Simulation {i+1}/{N_SIMULATIONS} (bundling + simulating)...", end='\r')
    
    with_results, no_results, monthly = simulate_both_scenarios_with_rebundling(
        fy2025_invoices, seed
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

# Show bundling variability
print(f"\nBundling Variability Metrics:")
print(f"  Number of statements range: {results_no_df['n_statements'].min():.0f} - {results_no_df['n_statements'].max():.0f}")
print(f"  Avg statement value range: ${results_no_df['avg_statement_value'].min():,.2f} - ${results_no_df['avg_statement_value'].max():,.2f}")
print(f"  This bundling variability introduces realistic uncertainty!")

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
print("MONTE CARLO RESULTS SUMMARY (STATEMENT-BASED WITH REBUNDLING)")
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
summary_output = os.path.join(OUTPUT_DIR, '10.10_MC_monte_carlo_summary_statistics_rebundled.csv')
summary_df.to_csv(summary_output, index=False)
print(f"✓ {summary_output}")

# All simulation results
results_with_df['scenario'] = 'with_discount'
results_no_df['scenario'] = 'no_discount'
all_results = pd.concat([results_with_df, results_no_df], ignore_index=True)

results_output = os.path.join(OUTPUT_DIR, '10.10_MC_simulation_results_distribution_rebundled.csv')
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
ax1.set_title('With Discount Revenue Distribution\n(Statement-Based with Rebundling)', fontsize=14, fontweight='bold')
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
ax2.set_title('No Discount Revenue Distribution\n(Statement-Based with Rebundling)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Total Revenue ($)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
hist_output = os.path.join(OUTPUT_DIR, '10.10_MC_revenue_distribution_histogram_rebundled.png')
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
ax.set_title(f'Revenue Comparison with {CONFIDENCE_LEVEL*100:.0f}% Confidence Intervals\n({N_SIMULATIONS} simulations, Statement-Based with Rebundling)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
ci_output = os.path.join(OUTPUT_DIR, '10.10_MC_confidence_intervals_rebundled.png')
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
ax.set_title(f'Revenue Distribution Comparison\n({N_SIMULATIONS} simulations, Statement-Based with Rebundling)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
boxplot_output = os.path.join(OUTPUT_DIR, '10.10_MC_scenario_comparison_boxplot_rebundled.png')
plt.savefig(boxplot_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {boxplot_output}")
plt.close()

# 4. Monthly Revenue with Uncertainty Bands
print("  Creating monthly revenue uncertainty bands...")
fig, ax = plt.subplots(figsize=(16, 9))

# Calculate monthly percentiles
def calculate_monthly_percentiles(monthly_df, percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]):
    """Calculate percentiles for each month across simulations"""
    monthly_stats = monthly_df.groupby('month')['cumulative'].quantile(percentiles).unstack()
    monthly_stats.columns = [int(p*100) for p in percentiles]
    return monthly_stats

monthly_with_stats = calculate_monthly_percentiles(monthly_with_df)
monthly_no_stats = calculate_monthly_percentiles(monthly_no_df)

# Plot WITH DISCOUNT uncertainty bands
ax.fill_between(monthly_with_stats.index, monthly_with_stats[5], monthly_with_stats[95],
                alpha=0.15, color='#70AD47', zorder=1)
ax.fill_between(monthly_with_stats.index, monthly_with_stats[25], monthly_with_stats[75],
                alpha=0.25, color='#70AD47', zorder=1)

# Plot NO DISCOUNT uncertainty bands
ax.fill_between(monthly_no_stats.index, monthly_no_stats[5], monthly_no_stats[95],
                alpha=0.15, color='#4472C4', zorder=1)
ax.fill_between(monthly_no_stats.index, monthly_no_stats[25], monthly_no_stats[75],
                alpha=0.25, color='#4472C4', zorder=1)

# Plot median lines on top
ax.plot(monthly_with_stats.index, monthly_with_stats[50], 
        color='#70AD47', linewidth=3.5, label='WITH DISCOUNT', marker='o', markersize=9, zorder=3)
ax.plot(monthly_no_stats.index, monthly_no_stats[50], 
        color='#4472C4', linewidth=3.5, label='NO DISCOUNT', marker='s', markersize=9, zorder=3)

# Add target line
ax.axhline(TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target: ${TARGET_REVENUE:,.0f}', alpha=0.8, zorder=2)

# Add revenue labels for final points only
with_final = monthly_with_stats[50].iloc[-1]
no_final = monthly_no_stats[50].iloc[-1]

# Format final values with K for <1M, M for >=1M
def format_revenue(val):
    if val < 1e6:
        return f'${val/1e3:.0f}K'
    else:
        return f'${val/1e6:.2f}M'

# Add label at end of WITH DISCOUNT
ax.text(monthly_with_stats.index[-1], with_final + 40000, 
        f'  {format_revenue(with_final)}', fontsize=12, fontweight='bold', color='#70AD47', va='bottom')

# Add label at end of NO DISCOUNT
ax.text(monthly_no_stats.index[-1], no_final - 40000, 
        f'  {format_revenue(no_final)}', fontsize=12, fontweight='bold', color='#4472C4', va='top')

ax.set_title(f'FY2025 Cumulative Revenue Historic Predictions Scaled\n({N_SIMULATIONS} Monte Carlo Sims)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Invoice Month', fontsize=15, fontweight='bold')
ax.set_ylabel('Cumulative Revenue ($)', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=13, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
monthly_output = os.path.join(OUTPUT_DIR, '10.10_MC_monthly_revenue_uncertainty_rebundled.png')
plt.savefig(monthly_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {monthly_output}")
plt.close()

# ================================================================
# Final Summary
# ================================================================
print("\n" + "="*70)
print("MONTE CARLO SIMULATION COMPLETE (WITH REBUNDLING)")
print("="*70)
print(f"\nSimulations run: {N_SIMULATIONS}")
print(f"Confidence level: {CONFIDENCE_LEVEL*100}%")
print(f"Total execution time: {elapsed:.2f} seconds")
print(f"Analysis approach: STATEMENTS with REBUNDLING each iteration")

print(f"\nNO DISCOUNT scenario:")
print(f"  Expected revenue: ${summary_no['total_revenue_mean']:,.2f}")
print(f"  {CONFIDENCE_LEVEL*100:.0f}% CI: ${summary_no['total_revenue_ci_lower']:,.2f} - ${summary_no['total_revenue_ci_upper']:,.2f}")
print(f"  Probability of meeting target: {prob_exceed_target:.1f}%")

print(f"\nRevenue improvement (No Discount vs With Discount):")
print(f"  Mean difference: ${revenue_diff_mean:,.2f}")
print(f"  Std deviation: ${revenue_diff_std:,.2f}")

print("\n" + "="*70)
print("KEY DIFFERENCES FROM ORIGINAL 10.10:")
print("  Original 10.10: Pre-bundled statements (same for all simulations)")
print("  This version: Rebundles invoices at EACH iteration")
print("  Captures MORE uncertainty: bundling + payment behavior")
print("  More realistic: future invoice groupings are uncertain")
print("="*70)