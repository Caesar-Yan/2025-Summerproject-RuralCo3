'''
10.5_Calculation_FAST_vectorized_100rounds_REBUNDLED.py

FAST VECTORIZED VERSION with RE-BUNDLING AT EACH ITERATION
Introduces realistic bundling uncertainty into Monte Carlo simulation

KEY CHANGES:
- Loads UNBUNDLED invoice data from 09.1 (not pre-bundled from 9.7)
- Re-bundles invoices at EACH iteration (introducing bundling variability)
- Captures MORE sources of uncertainty:
  * Which invoices bundle together
  * Statement value distributions
  * Decile assignments
  * Payment timing
  * Delinquency severity

This is more realistic because you don't know in advance exactly:
- Which invoices will appear on which future statements
- What the statement totals will be
- How statements will distribute across deciles

Inputs:
- ats_grouped_transformed_with_discounts.csv (UNBUNDLED from 09.1)
- invoice_grouped_transformed_with_discounts.csv (UNBUNDLED from 09.1)
- 09.6_reconstructed_late_payment_rates.csv (seasonal late payment rates)
- payment_profile/decile_payment_profile.pkl (for decile base rates and cd distributions)

Outputs:
- 10.5_FY2025_seasonal_comparison_summary_rebundled.csv
- 10.5_monte_carlo_all_rounds_rebundled.csv
- 10.5_revenue_distribution_histogram_rebundled.png
- 10.5_confidence_intervals_rebundled.png
- 10.5_scenario_comparison_boxplot_rebundled.png
- 10.5_monthly_revenue_uncertainty_rebundled.png
- 10.5_FY2025_seasonal_detailed_simulations_rebundled.xlsx

Author: Chris & Team
Date: January 2026
REFACTORED: January 2026 (re-bundles at each iteration)
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
BASE_PATH = Path(r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3")
data_cleaning_dir = BASE_PATH / "data_cleaning"
profile_dir = BASE_PATH / "payment_profile"
visualisations_dir = BASE_PATH / "visualisations"

ANNUAL_INTEREST_RATE = 0.2395
RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30

N_SIMULATIONS = 100
CONFIDENCE_LEVEL = 0.95

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

print("\n" + "="*80)
print("FAST VECTORIZED MONTE CARLO WITH RE-BUNDLING AT EACH ITERATION")
print("WITH SEASONALLY ADJUSTED DECILE RATES")
print("="*80)
print(f"Running {N_SIMULATIONS} simulation rounds")
print(f"RE-BUNDLING invoices at each iteration (captures bundling uncertainty)")
print(f"Using vectorized operations for speed")
print("="*80)

# ================================================================
# STEP 1: Load Seasonal Rates
# ================================================================
print("\n" + "="*80)
print("📊 [Step 1/5] LOADING SEASONAL RATES")
print("="*80)

seasonal_rates = pd.read_csv(visualisations_dir / "09.6_reconstructed_late_payment_rates.csv")
seasonal_rates['invoice_period'] = pd.to_datetime(seasonal_rates['invoice_period'])

november_2025_row = seasonal_rates[
    (seasonal_rates['invoice_period'].dt.year == 2025) & 
    (seasonal_rates['invoice_period'].dt.month == 11)
]

if len(november_2025_row) == 0:
    print("❌ ERROR: November 2025 not found in seasonal rates")
    exit(1)

november_baseline_rate = november_2025_row['reconstructed_late_rate_pct'].values[0] / 100

# Calculate seasonal adjustment factors
seasonal_adjustment_factors = {}
for _, row in seasonal_rates.iterrows():
    year_month = (row['invoice_period'].year, row['invoice_period'].month)
    month_rate = row['reconstructed_late_rate_pct'] / 100
    adjustment_factor = month_rate / november_baseline_rate
    seasonal_adjustment_factors[year_month] = adjustment_factor

print(f"  ✓ November 2025 baseline: {november_baseline_rate*100:.2f}%")
print(f"  ✓ Calculated {len(seasonal_adjustment_factors)} seasonal adjustment factors")

# ================================================================
# STEP 2: Load UNBUNDLED Invoice Data from 09.1
# ================================================================
print("\n" + "="*80)
print("📂 [Step 2/5] LOADING UNBUNDLED INVOICE DATA FROM 09.1")
print("="*80)

unbundled_ats_path = data_cleaning_dir / 'ats_grouped_transformed_with_discounts.csv'
unbundled_invoice_path = data_cleaning_dir / 'invoice_grouped_transformed_with_discounts.csv'

if not unbundled_ats_path.exists() or not unbundled_invoice_path.exists():
    print("❌ ERROR: Unbundled invoice files not found!")
    print("Please run 09.1_data_engineering_transformed.py first")
    exit(1)

ats_invoices = pd.read_csv(unbundled_ats_path)
invoice_invoices = pd.read_csv(unbundled_invoice_path)

ats_invoices['customer_type'] = 'ATS'
invoice_invoices['customer_type'] = 'Invoice'

print(f"  ✓ Loaded {len(ats_invoices):,} ATS invoices (UNBUNDLED)")
print(f"  ✓ Loaded {len(invoice_invoices):,} Invoice invoices (UNBUNDLED)")

# Combine
all_invoices = pd.concat([ats_invoices, invoice_invoices], ignore_index=True)
print(f"  ✓ Total unbundled invoices: {len(all_invoices):,}")

# ================================================================
# STEP 3: Parse Dates and Filter
# ================================================================
print("\n" + "="*80)
print("📅 [Step 3/5] PARSING DATES AND FILTERING")
print("="*80)

def parse_invoice_period(series: pd.Series) -> pd.Series:
    """Robustly parse invoice_period"""
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

all_invoices['invoice_period'] = parse_invoice_period(all_invoices['invoice_period'])
all_invoices = all_invoices[all_invoices['invoice_period'].notna()].copy()

# Filter to FY2025
fy2025_invoices = all_invoices[
    (all_invoices['invoice_period'] >= FY2025_START) & 
    (all_invoices['invoice_period'] <= FY2025_END)
].copy()

# Filter out negatives
fy2025_invoices = fy2025_invoices[fy2025_invoices['total_undiscounted_price'] >= 0].copy()
fy2025_invoices = fy2025_invoices[fy2025_invoices['total_discounted_price'] >= 0].copy()

print(f"  FY2025 ({FY2025_START.strftime('%d/%m/%Y')} - {FY2025_END.strftime('%d/%m/%Y')})")
print(f"  ✓ Filtered to {len(fy2025_invoices):,} invoices")

if len(fy2025_invoices) == 0:
    print("\n❌ WARNING: No invoices found in FY2025!")
    exit()

# ================================================================
# STEP 4: Load Payment Profile
# ================================================================
print("\n" + "="*80)
print("💳 [Step 4/5] LOADING DECILE PAYMENT PROFILE")
print("="*80)

try:
    with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
        decile_profile = pickle.load(f)
    
    n_deciles = decile_profile['metadata']['n_deciles']
    print(f"  ✓ Loaded {n_deciles} deciles")
    
except FileNotFoundError:
    print("❌ ERROR: Payment profile not found!")
    exit()

# Pre-compute decile base rates
decile_base_rates = {}
for i in range(n_deciles):
    decile_key = f'decile_{i}'
    if decile_key in decile_profile['deciles']:
        base_rate = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
    else:
        base_rate = 0.02
    decile_base_rates[i] = base_rate

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

print(f"  ✓ Pre-computed decile base rates and cd distributions")

# ================================================================
# STEP 5: Define Bundling and Simulation Functions
# ================================================================
print("\n" + "="*80)
print("🔧 [Step 5/5] DEFINING BUNDLING AND SIMULATION FUNCTIONS")
print("="*80)

def bundle_invoices_to_statements(invoices_df, seed):
    """
    Bundle invoices into statements using random assignment with Poisson distribution
    
    This is adapted from 9.7 but runs within each simulation iteration
    
    Args:
        invoices_df: DataFrame with invoice data
        seed: Random seed for this iteration
    
    Returns:
        DataFrame with statement_id and aggregated statement values
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
        
        # Generate statement sizes using Poisson
        statement_sizes = np.random.poisson(mean_invoices_per_statement, n_statements_this_month)
        statement_sizes = np.maximum(statement_sizes, 1)
        
        # Adjust to match total invoice count
        total_assigned = statement_sizes.sum()
        if total_assigned > n_invoices_this_month:
            statement_sizes = (statement_sizes * n_invoices_this_month / total_assigned).astype(int)
            statement_sizes = np.maximum(statement_sizes, 1)
        
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
        'total_discounted_price', 'total_undiscounted_price',
        'discount_amount', 'customer_type', 'n_invoices_in_statement'
    ]
    
    # Assign deciles based on statement value
    statements = statements.sort_values('total_undiscounted_price').reset_index(drop=True)
    statements['decile'] = pd.qcut(
        statements['total_undiscounted_price'],
        q=n_deciles,
        labels=False,
        duplicates='drop'
    )
    
    return statements

def simulate_both_scenarios_with_rebundling(invoices_df, seed):
    """
    FAST VECTORIZED simulation with RE-BUNDLING at each iteration
    
    Steps:
    1. Bundle invoices into statements (random groupings)
    2. Assign deciles based on statement totals
    3. Apply seasonal adjustments
    4. Simulate payment behavior
    5. Calculate revenue for both scenarios
    """
    np.random.seed(seed)
    
    # ================================================================
    # STEP 1: BUNDLE INVOICES (unique to this iteration)
    # ================================================================
    statements = bundle_invoices_to_statements(invoices_df, seed)
    
    # ================================================================
    # STEP 2: Apply seasonal adjustments
    # ================================================================
    statements['year_month_tuple'] = list(zip(
        statements['invoice_period'].dt.year,
        statements['invoice_period'].dt.month
    ))
    
    statements['seasonal_factor'] = statements['year_month_tuple'].map(
        seasonal_adjustment_factors
    ).fillna(1.0)
    
    statements['decile_base_rate'] = statements['decile'].map(decile_base_rates)
    
    # Handle missing deciles
    missing_decile = statements['decile_base_rate'].isna().sum()
    if missing_decile > 0:
        avg_rate = np.mean(list(decile_base_rates.values()))
        statements['decile_base_rate'].fillna(avg_rate, inplace=True)
    
    # Calculate adjusted late rates
    statements['adjusted_late_rate'] = statements['decile_base_rate'] * statements['seasonal_factor']
    statements['adjusted_late_rate'] = np.minimum(statements['adjusted_late_rate'], 1.0)
    
    # ================================================================
    # STEP 3: Determine late status (VECTORIZED)
    # ================================================================
    random_draws = np.random.random(len(statements))
    statements['is_late'] = random_draws < statements['adjusted_late_rate']
    
    # ================================================================
    # STEP 4: Sample cd levels (LOOP OVER DECILES)
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
            
            days_map = pd.Series(sampled_cd).map(CD_TO_DAYS).fillna(90).values
            statements.loc[decile_mask, 'days_overdue'] = days_map
    
    statements['months_overdue'] = statements['days_overdue'] / 30
    
    # ================================================================
    # STEP 5: Create WITH DISCOUNT scenario
    # ================================================================
    with_discount = statements.copy()
    
    with_discount['principal_amount'] = with_discount['total_discounted_price']
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
    
    no_discount['principal_amount'] = no_discount['total_undiscounted_price']
    no_discount['retained_discounts'] = np.where(
        no_discount['is_late'], 
        no_discount['discount_amount'], 
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
        'pct_late': with_discount['is_late'].sum() / len(with_discount) * 100,
        'n_statements': len(with_discount),
        'avg_statement_value': with_discount['total_undiscounted_price'].mean(),
    }
    
    no_discount_results = {
        'total_revenue': no_discount['credit_card_revenue'].sum(),
        'interest_revenue': no_discount['interest_charged'].sum(),
        'retained_discounts': no_discount['retained_discounts'].sum(),
        'n_late': no_discount['is_late'].sum(),
        'pct_late': no_discount['is_late'].sum() / len(no_discount) * 100,
        'n_statements': len(no_discount),
        'avg_statement_value': no_discount['total_undiscounted_price'].mean(),
    }
    
    # ================================================================
    # STEP 8: Monthly aggregation
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
    
    return with_discount_results, no_discount_results, monthly_details, with_discount, no_discount

print("  ✓ Bundling and simulation functions defined")

# ================================================================
# RUN MONTE CARLO SIMULATION WITH RE-BUNDLING
# ================================================================
print("\n" + "="*80)
print(f"🎲 RUNNING {N_SIMULATIONS} MONTE CARLO SIMULATIONS WITH RE-BUNDLING")
print("="*80)

all_results_with = []
all_results_no = []
all_monthly_with = []
all_monthly_no = []

first_round_with = None
first_round_no = None

start_time = time.time()

for i in range(N_SIMULATIONS):
    seed = RANDOM_SEED + i
    
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Simulation {i+1}/{N_SIMULATIONS} (bundling + simulating)...", end='\r')
    
    with_results, no_results, monthly, with_df, no_df = simulate_both_scenarios_with_rebundling(
        fy2025_invoices, seed
    )
    
    if i == 0:
        first_round_with = with_df
        first_round_no = no_df
    
    with_results['simulation'] = i
    no_results['simulation'] = i
    
    all_results_with.append(with_results)
    all_results_no.append(no_results)
    
    monthly_with = monthly['with_discount'].copy()
    monthly_with['simulation'] = i
    all_monthly_with.append(monthly_with)
    
    monthly_no = monthly['no_discount'].copy()
    monthly_no['simulation'] = i
    all_monthly_no.append(monthly_no)

elapsed = time.time() - start_time
print(f"\n  ✓ Completed {N_SIMULATIONS} simulations in {elapsed:.2f} seconds ({elapsed/N_SIMULATIONS:.3f}s per simulation)")
print(f"  Note: Slower than pre-bundled approach due to re-bundling overhead")
print(f"  BUT captures more realistic uncertainty!")

# ================================================================
# PROCESS RESULTS
# ================================================================
print("\n" + "="*80)
print("📊 PROCESSING RESULTS")
print("="*80)

results_with_df = pd.DataFrame(all_results_with)
results_no_df = pd.DataFrame(all_results_no)

monthly_with_df = pd.concat(all_monthly_with, ignore_index=True)
monthly_no_df = pd.concat(all_monthly_no, ignore_index=True)

summary_stats_df = pd.DataFrame({
    'round': range(N_SIMULATIONS),
    'with_discount_revenue': results_with_df['total_revenue'],
    'with_discount_interest': results_with_df['interest_revenue'],
    'with_discount_retained': results_with_df['retained_discounts'],
    'with_discount_late_rate': results_with_df['pct_late'],
    'with_discount_n_late': results_with_df['n_late'],
    'with_discount_n_statements': results_with_df['n_statements'],
    'with_discount_avg_statement_value': results_with_df['avg_statement_value'],
    'no_discount_revenue': results_no_df['total_revenue'],
    'no_discount_interest': results_no_df['interest_revenue'],
    'no_discount_retained': results_no_df['retained_discounts'],
    'no_discount_late_rate': results_no_df['pct_late'],
    'no_discount_n_late': results_no_df['n_late'],
    'no_discount_n_statements': results_no_df['n_statements'],
    'no_discount_avg_statement_value': results_no_df['avg_statement_value'],
    'revenue_difference': results_no_df['total_revenue'] - results_with_df['total_revenue']
})

print(f"  ✓ Processed {len(summary_stats_df)} simulation rounds")

# Show bundling variability
print(f"\n  Bundling Variability Metrics:")
print(f"    Number of statements range: {summary_stats_df['no_discount_n_statements'].min():.0f} - {summary_stats_df['no_discount_n_statements'].max():.0f}")
print(f"    Avg statement value range: ${summary_stats_df['no_discount_avg_statement_value'].min():,.2f} - ${summary_stats_df['no_discount_avg_statement_value'].max():,.2f}")
print(f"    This variability was NOT captured in pre-bundled approach!")

# ================================================================
# CALCULATE SUMMARY STATISTICS
# ================================================================
print("\n" + "="*80)
print(f"💰 MONTE CARLO RESULTS SUMMARY ({N_SIMULATIONS} ROUNDS)")
print("="*80)

def calculate_summary_stats(df, scenario_name):
    """Calculate mean, median, std, CI for all metrics"""
    stats_dict = {
        'scenario': scenario_name,
        'n_simulations': len(df),
    }
    
    col_mapping = {
        'total_revenue': 'revenue',
        'interest_revenue': 'interest',
        'retained_discounts': 'retained',
        'pct_late': 'late_rate',
        'n_late': 'n_late'
    }
    
    for col, col_suffix in col_mapping.items():
        if scenario_name == 'with_discount':
            col_name = f'with_discount_{col_suffix}'
        else:
            col_name = f'no_discount_{col_suffix}'
        
        values = df[col_name].values
        
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values, ddof=1)
        
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

summary_with = calculate_summary_stats(summary_stats_df, 'with_discount')
summary_no = calculate_summary_stats(summary_stats_df, 'no_discount')

monte_carlo_summary_df = pd.DataFrame([summary_with, summary_no])

def print_distribution_stats(stats_dict, label):
    """Print statistics for a distribution"""
    print(f"\n{label}")
    print(f"  Mean:        ${stats_dict['total_revenue_mean']:>15,.2f}")
    print(f"  Median:      ${stats_dict['total_revenue_median']:>15,.2f}")
    print(f"  Std Dev:     ${stats_dict['total_revenue_std']:>15,.2f}")
    print(f"  {CONFIDENCE_LEVEL*100:.0f}% CI:   ${stats_dict['total_revenue_ci_lower']:>15,.2f} - ${stats_dict['total_revenue_ci_upper']:>15,.2f}")
    print(f"  Range:       ${stats_dict['total_revenue_min']:>15,.2f} - ${stats_dict['total_revenue_max']:>15,.2f}")

print_distribution_stats(summary_with, "WITH DISCOUNT - Total Revenue Distribution")
print_distribution_stats(summary_no, "NO DISCOUNT - Total Revenue Distribution")

revenue_diff_mean = summary_stats_df['revenue_difference'].mean()
revenue_diff_std = summary_stats_df['revenue_difference'].std(ddof=1)

print(f"\n" + "="*80)
print("REVENUE DIFFERENCE (No Discount - With Discount)")
print("="*80)
print(f"  Mean:        ${revenue_diff_mean:>15,.2f}")
print(f"  Std Dev:     ${revenue_diff_std:>15,.2f}")

if revenue_diff_mean > 0:
    pct_more = (revenue_diff_mean / summary_with['total_revenue_mean']) * 100
    print(f"\n  ✓ NO DISCOUNT generates ${revenue_diff_mean:,.2f} MORE revenue on average ({pct_more:.1f}%)")

print(f"\nTarget Comparison (No Discount scenario):")
print(f"  Target:         ${TARGET_REVENUE:>15,.2f}")
print(f"  Simulated Mean: ${summary_no['total_revenue_mean']:>15,.2f}")
print(f"  Gap:            ${summary_no['total_revenue_mean'] - TARGET_REVENUE:>15,.2f}")

prob_exceed_target = (summary_stats_df['no_discount_revenue'] >= TARGET_REVENUE).sum() / N_SIMULATIONS * 100
print(f"\n  Probability of meeting/exceeding target: {prob_exceed_target:.1f}%")

# ================================================================
# SAVE RESULTS
# ================================================================
print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

summary_output = OUTPUT_DIR / '10.5_FY2025_seasonal_comparison_summary_rebundled.csv'
monte_carlo_summary_df.to_csv(summary_output, index=False)
print(f"  ✓ Saved: {summary_output.name}")

all_rounds_csv = OUTPUT_DIR / '10.5_monte_carlo_all_rounds_rebundled.csv'
summary_stats_df.to_csv(all_rounds_csv, index=False)
print(f"  ✓ Saved: {all_rounds_csv.name}")

output_excel = OUTPUT_DIR / '10.5_FY2025_seasonal_detailed_simulations_rebundled.xlsx'
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    first_round_with.to_excel(writer, sheet_name='With_Discount_Round0', index=False)
    first_round_no.to_excel(writer, sheet_name='No_Discount_Round0', index=False)
    monte_carlo_summary_df.to_excel(writer, sheet_name='Monte_Carlo_Summary', index=False)
    summary_stats_df.to_excel(writer, sheet_name='All_Rounds', index=False)

print(f"  ✓ Saved: {output_excel.name}")

# ================================================================
# VISUALIZATIONS
# ================================================================
print("\n" + "="*80)
print("🎨 CREATING VISUALIZATIONS")
print("="*80)

# 1. Revenue Distribution Histogram
print("  Creating revenue distribution histogram...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ax1.hist(summary_stats_df['with_discount_revenue'], bins=30, alpha=0.7, color='#70AD47', edgecolor='black')
ax1.axvline(summary_with['total_revenue_mean'], color='red', linestyle='--', linewidth=2.5, 
            label=f"Mean: ${summary_with['total_revenue_mean']:,.0f}")
ax1.axvline(summary_with['total_revenue_ci_lower'], color='orange', linestyle=':', linewidth=2, 
            label=f"{CONFIDENCE_LEVEL*100:.0f}% CI")
ax1.axvline(summary_with['total_revenue_ci_upper'], color='orange', linestyle=':', linewidth=2)
ax1.set_title(f'With Discount Revenue Distribution\n({N_SIMULATIONS} Sims with Re-Bundling)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Total Revenue ($)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

ax2.hist(summary_stats_df['no_discount_revenue'], bins=30, alpha=0.7, color='#4472C4', edgecolor='black')
ax2.axvline(summary_no['total_revenue_mean'], color='red', linestyle='--', linewidth=2.5, 
            label=f"Mean: ${summary_no['total_revenue_mean']:,.0f}")
ax2.axvline(summary_no['total_revenue_ci_lower'], color='orange', linestyle=':', linewidth=2, 
            label=f"{CONFIDENCE_LEVEL*100:.0f}% CI")
ax2.axvline(summary_no['total_revenue_ci_upper'], color='orange', linestyle=':', linewidth=2)
ax2.axvline(TARGET_REVENUE, color='green', linestyle='-', linewidth=2.5, 
            label=f"Target: ${TARGET_REVENUE:,.0f}", alpha=0.8)
ax2.set_title(f'No Discount Revenue Distribution\n({N_SIMULATIONS} Sims with Re-Bundling)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Total Revenue ($)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
hist_output = OUTPUT_DIR / '10.5_revenue_distribution_histogram_rebundled.png'
plt.savefig(hist_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {hist_output.name}")
plt.close()

# 2. Confidence Intervals
print("  Creating confidence intervals...")
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

for i, (mean, scenario) in enumerate(zip(means, scenarios)):
    ax.text(mean, i, f'  ${mean:,.0f}', va='center', fontsize=11, fontweight='bold')

ax.axvline(TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target: ${TARGET_REVENUE:,.0f}', alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(scenarios, fontsize=12)
ax.set_xlabel('Total Revenue ($)', fontsize=14)
ax.set_title(f'Revenue with {CONFIDENCE_LEVEL*100:.0f}% CI\n({N_SIMULATIONS} Sims with Re-Bundling)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
ci_output = OUTPUT_DIR / '10.5_confidence_intervals_rebundled.png'
plt.savefig(ci_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {ci_output.name}")
plt.close()

# 3. Boxplot
print("  Creating boxplot...")
fig, ax = plt.subplots(figsize=(12, 8))

data_to_plot = [summary_stats_df['with_discount_revenue'], summary_stats_df['no_discount_revenue']]
box_colors = ['#70AD47', '#4472C4']

bp = ax.boxplot(data_to_plot, labels=scenarios, patch_artist=True,
                showmeans=True, meanline=True,
                medianprops=dict(color='red', linewidth=2.5),
                meanprops=dict(color='blue', linewidth=2.5, linestyle='--'))

for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(TARGET_REVENUE, color='green', linestyle='--', linewidth=2.5, 
           label=f'Target: ${TARGET_REVENUE:,.0f}', alpha=0.8)

ax.set_ylabel('Total Revenue ($)', fontsize=14)
ax.set_title(f'Revenue Distribution Comparison\n({N_SIMULATIONS} Sims with Re-Bundling)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
boxplot_output = OUTPUT_DIR / '10.5_scenario_comparison_boxplot_rebundled.png'
plt.savefig(boxplot_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {boxplot_output.name}")
plt.close()

# 4. Monthly Revenue Uncertainty
print("  Creating monthly revenue uncertainty...")
fig, ax = plt.subplots(figsize=(16, 9))

def calculate_monthly_percentiles(monthly_df, percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]):
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

# Plot median lines
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

ax.set_title(f'FY2025 Cumulative Revenue Historic Predictions\n({N_SIMULATIONS} Monte Carlo Sims)', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Invoice Month', fontsize=15, fontweight='bold')
ax.set_ylabel('Cumulative Revenue ($)', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=13, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
monthly_output = OUTPUT_DIR / '10.5_monthly_revenue_uncertainty_rebundled.png'
plt.savefig(monthly_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {monthly_output.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE - MONTE CARLO WITH RE-BUNDLING")
print("="*80)

print(f"\n📊 Simulation Details:")
print(f"  Simulations run: {N_SIMULATIONS}")
print(f"  Invoices per simulation: {len(fy2025_invoices):,}")
print(f"  Execution time: {elapsed:.2f} seconds")

print(f"\n💰 NO DISCOUNT Scenario:")
print(f"  Expected revenue: ${summary_no['total_revenue_mean']:,.2f}")
print(f"  {CONFIDENCE_LEVEL*100:.0f}% CI: ${summary_no['total_revenue_ci_lower']:,.2f} - ${summary_no['total_revenue_ci_upper']:,.2f}")
print(f"  Probability of meeting target: {prob_exceed_target:.1f}%")

print(f"\n📁 Output Files:")
print(f"  • 10.5_FY2025_seasonal_comparison_summary_rebundled.csv")
print(f"  • 10.5_monte_carlo_all_rounds_rebundled.csv")
print(f"  • 10.5_FY2025_seasonal_detailed_simulations_rebundled.xlsx")
print(f"  • 10.5_revenue_distribution_histogram_rebundled.png")
print(f"  • 10.5_confidence_intervals_rebundled.png")
print(f"  • 10.5_scenario_comparison_boxplot_rebundled.png")
print(f"  • 10.5_monthly_revenue_uncertainty_rebundled.png")

print(f"\n  All files saved to: {OUTPUT_DIR}/")

print("\n" + "="*80)
print("KEY METHODOLOGY - RE-BUNDLING APPROACH:")
print("="*80)
print("  1. Loaded UNBUNDLED invoices from 09.1")
print("  2. For EACH of 100 iterations:")
print("     a. Randomly bundle invoices into statements")
print("     b. Assign deciles based on statement totals (varies per iteration)")
print("     c. Apply seasonal adjustments")
print("     d. Simulate payment behavior")
print("     e. Calculate revenue")
print("  3. Captures bundling uncertainty + payment uncertainty")
print("  4. More realistic forecast intervals")
print("\nSources of Uncertainty Captured:")
print("  ✓ Bundling variability (which invoices group together)")
print("  ✓ Statement value distribution")
print("  ✓ Decile composition")
print("  ✓ Payment timing (late vs on-time)")
print("  ✓ Delinquency severity (cd levels)")
print("="*80)