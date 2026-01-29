'''
Docstring for 10.5_Calculation_FAST_vectorized_100rounds

FAST VECTORIZED VERSION using 10.10's optimization techniques

This script makes estimates of revenue generated under discount and no-discount scenarios,
using SEASONALLY ADJUSTED DECILE-SPECIFIC late payment rates with 100 MONTE CARLO ROUNDS.

KEY OPTIMIZATION: Uses vectorized operations instead of row-by-row iteration
- Pre-computes all values before simulation loop
- Uses numpy vectorized operations for random draws
- Processes all invoices simultaneously
- Expected speedup: 50-100x faster than iterative version

Inputs:
- ats_grouped_transformed_with_discounts.csv
- invoice_grouped_transformed_with_discounts.csv
- 09.6_reconstructed_late_payment_rates.csv (seasonal late payment rates)
- decile_payment_profile.pkl (for decile base rates and cd level distributions)

Outputs:
- 10.5_FY2025_seasonal_comparison_summary.csv
- 10.5_monte_carlo_all_rounds.csv
- 10.5_revenue_distribution_histogram.png
- 10.5_confidence_intervals.png
- 10.5_scenario_comparison_boxplot.png
- 10.5_monthly_revenue_uncertainty.png
- (+ other standard outputs)
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

ANNUAL_INTEREST_RATE = 0.2395
RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30

N_SIMULATIONS = 100
CONFIDENCE_LEVEL = 0.95

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
print("FAST VECTORIZED MONTE CARLO SIMULATION")
print("WITH SEASONALLY ADJUSTED DECILE RATES")
print("="*70)
print(f"Running {N_SIMULATIONS} simulation rounds")
print(f"Using vectorized operations (50-100x faster than row-by-row)")

# ================================================================
# Load seasonal rates and calculate adjustment factors
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

if len(november_2025_row) == 0:
    print("ERROR: November 2025 not found in seasonal rates")
    exit(1)

november_baseline_rate = november_2025_row['reconstructed_late_rate_pct'].values[0] / 100

# Calculate seasonal adjustment factors
seasonal_adjustment_factors = {}
for _, row in seasonal_rates.iterrows():
    year_month = (row['invoice_period'].year, row['invoice_period'].month)
    month_rate = row['reconstructed_late_rate_pct'] / 100
    adjustment_factor = month_rate / november_baseline_rate
    seasonal_adjustment_factors[year_month] = adjustment_factor

print(f"✓ November 2025 baseline: {november_baseline_rate*100:.2f}%")
print(f"✓ Calculated {len(seasonal_adjustment_factors)} seasonal adjustment factors")

# ================================================================
# Load invoice data
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

# Filter out negatives
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
# Load decile payment profile
# ================================================================
print("\n" + "="*70)
print("LOADING DECILE PAYMENT PROFILE")
print("="*70)

with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
    decile_profile = pickle.load(f)

n_deciles = decile_profile['metadata']['n_deciles']
print(f"✓ Loaded {n_deciles} deciles")

# ================================================================
# Map invoices to deciles
# ================================================================
print("\n" + "="*70)
print("MAPPING INVOICES TO DECILES")
print("="*70)

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
fy2025_df['year_month'] = list(zip(
    fy2025_df['invoice_period'].dt.year,
    fy2025_df['invoice_period'].dt.month
))

# Map seasonal factors (VECTORIZED)
fy2025_df['seasonal_factor'] = fy2025_df['year_month'].map(
    seasonal_adjustment_factors
).fillna(1.0)

# Map decile base rates (VECTORIZED)
decile_base_rates = {}
for i in range(n_deciles):
    decile_key = f'decile_{i}'
    if decile_key in decile_profile['deciles']:
        base_rate = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
    else:
        base_rate = 0.02
    decile_base_rates[i] = base_rate

fy2025_df['decile_base_rate'] = fy2025_df['decile'].map(decile_base_rates)

# Pre-compute cd distributions for each decile
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
            decile_cd_distributions[i] = ([3], [1.0])  # Default to cd=3 (60 days)
    else:
        decile_cd_distributions[i] = ([3], [1.0])

print(f"✓ Pre-computed values for {len(fy2025_df):,} invoices")
print(f"  - Seasonal factors mapped")
print(f"  - Decile base rates mapped")
print(f"  - cd distributions prepared")

# ================================================================
# VECTORIZED simulation function
# ================================================================
def simulate_both_scenarios_vectorized(invoices_df, seed):
    """
    FAST VECTORIZED simulation of BOTH scenarios
    
    Uses same late status for both scenarios (consistent comparison)
    All operations are vectorized for maximum speed
    """
    np.random.seed(seed)
    
    df = invoices_df.copy()
    
    # ================================================================
    # STEP 1: Calculate adjusted late rates (VECTORIZED)
    # ================================================================
    df['adjusted_late_rate'] = df['decile_base_rate'] * df['seasonal_factor']
    df['adjusted_late_rate'] = np.minimum(df['adjusted_late_rate'], 1.0)
    
    # ================================================================
    # STEP 2: Determine late status for ALL invoices at once (VECTORIZED)
    # ================================================================
    random_draws = np.random.random(len(df))
    df['is_late'] = random_draws < df['adjusted_late_rate']
    
    # ================================================================
    # STEP 3: Sample cd levels for late invoices (LOOP ONLY OVER DECILES)
    # ================================================================
    df['cd_level'] = 0
    df['days_overdue'] = 0.0
    
    # Loop over DECILES (typically 10) not invoices (thousands)
    for decile_num in df['decile'].unique():
        decile_mask = (df['decile'] == decile_num) & df['is_late']
        n_late_in_decile = decile_mask.sum()
        
        if n_late_in_decile > 0:
            cd_levels, cd_probs = decile_cd_distributions.get(int(decile_num), ([3], [1.0]))
            sampled_cd = np.random.choice(cd_levels, size=n_late_in_decile, p=cd_probs)
            df.loc[decile_mask, 'cd_level'] = sampled_cd
            
            # Map cd to days (VECTORIZED)
            days_map = pd.Series(sampled_cd).map(CD_TO_DAYS).fillna(90).values
            df.loc[decile_mask, 'days_overdue'] = days_map
    
    df['months_overdue'] = df['days_overdue'] / 30
    
    # ================================================================
    # STEP 4: Create WITH DISCOUNT scenario
    # ================================================================
    with_discount = df.copy()
    
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
    # STEP 5: Create NO DISCOUNT scenario
    # ================================================================
    no_discount = df.copy()
    
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
    # STEP 6: Calculate summary statistics
    # ================================================================
    with_discount_results = {
        'total_revenue': with_discount['credit_card_revenue'].sum(),
        'interest_revenue': with_discount['interest_charged'].sum(),
        'retained_discounts': with_discount['retained_discounts'].sum(),
        'n_late': with_discount['is_late'].sum(),
        'pct_late': with_discount['is_late'].sum() / len(with_discount) * 100,
    }
    
    no_discount_results = {
        'total_revenue': no_discount['credit_card_revenue'].sum(),
        'interest_revenue': no_discount['interest_charged'].sum(),
        'retained_discounts': no_discount['retained_discounts'].sum(),
        'n_late': no_discount['is_late'].sum(),
        'pct_late': no_discount['is_late'].sum() / len(no_discount) * 100,
    }
    
    # ================================================================
    # STEP 7: Monthly aggregation
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

# ================================================================
# Run Monte Carlo simulation
# ================================================================
print("\n" + "="*70)
print(f"RUNNING {N_SIMULATIONS} VECTORIZED MONTE CARLO SIMULATIONS")
print("="*70)

all_results_with = []
all_results_no = []
all_monthly_with = []
all_monthly_no = []

# Store first round detailed results
first_round_with = None
first_round_no = None

start_time = time.time()

for i in range(N_SIMULATIONS):
    seed = RANDOM_SEED + i
    
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Simulation {i+1}/{N_SIMULATIONS}...", end='\r')
    
    with_results, no_results, monthly, with_df, no_df = simulate_both_scenarios_vectorized(
        fy2025_df, seed
    )
    
    # Store first round for detailed analysis
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
print(f"\n✓ Completed {N_SIMULATIONS} simulations in {elapsed:.2f} seconds ({elapsed/N_SIMULATIONS:.3f}s per simulation)")
print(f"  Speed: ~{len(fy2025_df) * N_SIMULATIONS / elapsed:,.0f} invoices/second")

# ================================================================
# Convert to DataFrames
# ================================================================
results_with_df = pd.DataFrame(all_results_with)
results_no_df = pd.DataFrame(all_results_no)

monthly_with_df = pd.concat(all_monthly_with, ignore_index=True)
monthly_no_df = pd.concat(all_monthly_no, ignore_index=True)

# Combine results
summary_stats_df = pd.DataFrame({
    'round': range(N_SIMULATIONS),
    'with_discount_revenue': results_with_df['total_revenue'],
    'with_discount_interest': results_with_df['interest_revenue'],
    'with_discount_retained': results_with_df['retained_discounts'],
    'with_discount_late_rate': results_with_df['pct_late'],
    'with_discount_n_late': results_with_df['n_late'],
    'no_discount_revenue': results_no_df['total_revenue'],
    'no_discount_interest': results_no_df['interest_revenue'],
    'no_discount_retained': results_no_df['retained_discounts'],
    'no_discount_late_rate': results_no_df['pct_late'],
    'no_discount_n_late': results_no_df['n_late'],
    'revenue_difference': results_no_df['total_revenue'] - results_with_df['total_revenue']
})

# ================================================================
# Calculate summary statistics
# ================================================================
print("\n" + "="*70)
print(f"MONTE CARLO RESULTS SUMMARY ({N_SIMULATIONS} ROUNDS)")
print("="*70)

def calculate_summary_stats(df, scenario_name):
    """Calculate mean, median, std, CI for all metrics"""
    stats_dict = {
        'scenario': scenario_name,
        'n_simulations': len(df),
    }
    
    # Map metric names to actual column names in summary_stats_df
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

print(f"\n" + "="*70)
print("REVENUE DIFFERENCE (No Discount - With Discount)")
print("="*70)
print(f"  Mean:        ${revenue_diff_mean:>15,.2f}")
print(f"  Std Dev:     ${revenue_diff_std:>15,.2f}")

if revenue_diff_mean > 0:
    pct_more = (revenue_diff_mean / summary_with['total_revenue_mean']) * 100
    print(f"\n✓ NO DISCOUNT generates ${revenue_diff_mean:,.2f} MORE revenue on average ({pct_more:.1f}%)")

print(f"\nTarget Comparison (No Discount scenario):")
print(f"  Target:         ${TARGET_REVENUE:>15,.2f}")
print(f"  Simulated Mean: ${summary_no['total_revenue_mean']:>15,.2f}")
print(f"  Gap:            ${summary_no['total_revenue_mean'] - TARGET_REVENUE:>15,.2f}")

prob_exceed_target = (summary_stats_df['no_discount_revenue'] >= TARGET_REVENUE).sum() / N_SIMULATIONS * 100
print(f"\nProbability of meeting/exceeding target: {prob_exceed_target:.1f}%")

# ================================================================
# Save results
# ================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Summary statistics
summary_output = os.path.join(OUTPUT_DIR, '10.5_FY2025_seasonal_comparison_summary.csv')
monte_carlo_summary_df.to_csv(summary_output, index=False)
print(f"✓ {summary_output}")

# All round results
all_rounds_csv = os.path.join(OUTPUT_DIR, '10.5_monte_carlo_all_rounds.csv')
summary_stats_df.to_csv(all_rounds_csv, index=False)
print(f"✓ {all_rounds_csv}")

# Detailed simulations (Round 0 only)
output_excel = os.path.join(OUTPUT_DIR, '10.5_FY2025_seasonal_detailed_simulations.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    first_round_with.to_excel(writer, sheet_name='With_Discount_Round0', index=False)
    first_round_no.to_excel(writer, sheet_name='No_Discount_Round0', index=False)
    monte_carlo_summary_df.to_excel(writer, sheet_name='Monte_Carlo_Summary', index=False)
    summary_stats_df.to_excel(writer, sheet_name='All_Rounds', index=False)

print(f"✓ {output_excel}")

# ================================================================
# VISUALIZATIONS
# ================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. Revenue Distribution Histogram
print("  Creating revenue distribution histogram...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

ax1.hist(summary_stats_df['with_discount_revenue'], bins=30, alpha=0.7, color='#70AD47', edgecolor='black')
ax1.axvline(summary_with['total_revenue_mean'], color='red', linestyle='--', linewidth=2.5, 
            label=f"Mean: ${summary_with['total_revenue_mean']:,.0f}")
ax1.axvline(summary_with['total_revenue_ci_lower'], color='orange', linestyle=':', linewidth=2, 
            label=f"{CONFIDENCE_LEVEL*100:.0f}% CI")
ax1.axvline(summary_with['total_revenue_ci_upper'], color='orange', linestyle=':', linewidth=2)
ax1.set_title(f'With Discount Revenue Distribution\n({N_SIMULATIONS} Simulations)', fontsize=14, fontweight='bold')
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
ax2.set_title(f'No Discount Revenue Distribution\n({N_SIMULATIONS} Simulations)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Total Revenue ($)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
hist_output = os.path.join(OUTPUT_DIR, '10.5_revenue_distribution_histogram.png')
plt.savefig(hist_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {hist_output}")
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
ax.set_title(f'Revenue Comparison with {CONFIDENCE_LEVEL*100:.0f}% Confidence Intervals\n({N_SIMULATIONS} Simulations)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
ci_output = os.path.join(OUTPUT_DIR, '10.5_confidence_intervals.png')
plt.savefig(ci_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {ci_output}")
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
ax.set_title(f'Revenue Distribution Comparison\n({N_SIMULATIONS} Simulations)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
boxplot_output = os.path.join(OUTPUT_DIR, '10.5_scenario_comparison_boxplot.png')
plt.savefig(boxplot_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {boxplot_output}")
plt.close()

# 4. Monthly Revenue Uncertainty
print("  Creating monthly revenue uncertainty...")
fig, ax = plt.subplots(figsize=(16, 8))

def calculate_monthly_percentiles(monthly_df, percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]):
    monthly_stats = monthly_df.groupby('month')['cumulative'].quantile(percentiles).unstack()
    monthly_stats.columns = [int(p*100) for p in percentiles]
    return monthly_stats

monthly_with_stats = calculate_monthly_percentiles(monthly_with_df)
monthly_no_stats = calculate_monthly_percentiles(monthly_no_df)

ax.fill_between(monthly_with_stats.index, monthly_with_stats[5], monthly_with_stats[95],
                alpha=0.15, color='#70AD47', label='With Discount 90% CI')
ax.fill_between(monthly_with_stats.index, monthly_with_stats[25], monthly_with_stats[75],
                alpha=0.25, color='#70AD47', label='With Discount 50% CI')

ax.fill_between(monthly_no_stats.index, monthly_no_stats[5], monthly_no_stats[95],
                alpha=0.15, color='#4472C4', label='No Discount 90% CI')
ax.fill_between(monthly_no_stats.index, monthly_no_stats[25], monthly_no_stats[75],
                alpha=0.25, color='#4472C4', label='No Discount 50% CI')

ax.plot(monthly_with_stats.index, monthly_with_stats[50], 
        color='#70AD47', linewidth=3, label='With Discount (Median)', marker='o', markersize=8)
ax.plot(monthly_no_stats.index, monthly_no_stats[50], 
        color='#4472C4', linewidth=3, label='No Discount (Median)', marker='s', markersize=8)

ax.axhline(TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target: ${TARGET_REVENUE:,.0f}', alpha=0.8)

ax.set_title(f'FY2025 Cumulative Revenue with Uncertainty Bands\n({N_SIMULATIONS} Simulations)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Invoice Month', fontsize=14)
ax.set_ylabel('Cumulative Revenue ($)', fontsize=14)
ax.legend(loc='upper left', fontsize=11, ncol=2)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
monthly_output = os.path.join(OUTPUT_DIR, '10.5_monthly_revenue_uncertainty.png')
plt.savefig(monthly_output, dpi=300, bbox_inches='tight')
print(f"  ✓ {monthly_output}")
plt.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE - FAST VECTORIZED MONTE CARLO")
print("="*70)
print(f"\nSimulations run: {N_SIMULATIONS}")
print(f"Execution time: {elapsed:.2f} seconds")
print(f"Speed: ~{len(fy2025_df) * N_SIMULATIONS / elapsed:,.0f} invoices/second")
print(f"\nNO DISCOUNT scenario:")
print(f"  Expected revenue: ${summary_no['total_revenue_mean']:,.2f}")
print(f"  {CONFIDENCE_LEVEL*100:.0f}% CI: ${summary_no['total_revenue_ci_lower']:,.2f} - ${summary_no['total_revenue_ci_upper']:,.2f}")
print(f"  Probability of meeting target: {prob_exceed_target:.1f}%")
print("="*70)