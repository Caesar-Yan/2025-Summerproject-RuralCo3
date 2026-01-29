'''
10.0.3_Calculation_with_cd_deciles_and_retentions_bundled

This script makes estimates of revenue generated under discount and no-discount scenarios,
i.e. the current and proposed models, using BUNDLED STATEMENTS as the payment unit.

Key differences from original 10:
- Uses bundled invoice data from script 9.7 (already aggregated to statements)
- Treats each statement as a single payment unit
- Uses THE SAME statement assignments for BOTH scenarios (fair comparison)
- More realistic modeling of customer payment behavior

Late rates and how late payments are when late are estimated using the real data 
from group2's data. These profiles can be found in payment_profile folder under ruralco3

Inputs:
- 9.7_ats_grouped_transformed_with_discounts_bundled.csv
- 9.7_invoice_grouped_transformed_with_discounts_bundled.csv
- payment_profile/decile_payment_profile.pkl

Outputs:
- 10.0.3_FY2025_cd_timing_comparison_summary_bundled.csv
- 10.0.3_FY2025_cd_timing_detailed_simulations_bundled.xlsx
- 10.0.3_cd_level_analysis_bundled.csv
- 10.0.3_cumulative_revenue_with_target_and_components_bundled.png
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
# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
data_cleaning_dir = base_dir / "data_cleaning"
visualisations_dir = base_dir / "visualisations"

ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30  # 20 days = 0.67 months

# New Zealand FY2025 definition
FY2025_START = pd.Timestamp("2024-07-01")
FY2025_END = pd.Timestamp("2025-06-30")

OUTPUT_DIR = visualisations_dir

# ================================================================
# CD LEVEL TO PAYMENT TIMING MAPPING
# ================================================================
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
print("BUNDLED STATEMENT-BASED REVENUE ESTIMATION")
print("CD LEVEL TO PAYMENT TIMING MAPPING")
print("="*70)
for cd, days in sorted(CD_TO_DAYS.items()):
    months = days / 30
    print(f"  cd = {cd}: {days} days ({months:.1f} months) overdue")

np.random.seed(RANDOM_SEED)

# ================================================================
# Load BUNDLED invoice data (already contains statement_id and decile)
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

# Load bundled data
ats_grouped = pd.read_csv(bundled_ats_path)
invoice_grouped = pd.read_csv(bundled_invoice_path)

print(f"✓ Loaded {len(ats_grouped):,} ATS invoices (bundled)")
print(f"✓ Loaded {len(invoice_grouped):,} Invoice invoices (bundled)")

# Verify required columns exist
required_columns = ['statement_id', 'decile']
for df, name in [(ats_grouped, 'ATS'), (invoice_grouped, 'Invoice')]:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"ERROR: {name} data missing columns: {missing}")
        print("Please re-run 9.7_bundle_invoices_to_statements.py")
        exit(1)

print(f"✓ Verified: statement_id and decile columns present")

# Combine datasets
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"\nTotal invoices loaded: {len(combined_df):,}")
print(f"Total unique statements: {combined_df['statement_id'].nunique():,}")

# ================================================================
# Filter out negative undiscounted prices
# ================================================================
initial_count = len(combined_df)
combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()
filtered_count = initial_count - len(combined_df)
if filtered_count > 0:
    print(f"⚠ Filtered out {filtered_count:,} invoices with negative undiscounted prices")
print(f"Remaining invoices: {len(combined_df):,}")
print(f"Remaining statements: {combined_df['statement_id'].nunique():,}")

# ================================================================
# Parse and filter dates for FY2025
# ================================================================
def parse_invoice_period(series: pd.Series) -> pd.Series:
    """Robustly parse invoice_period"""
    s = series.copy()
    s_str = s.astype(str).str.strip()
    s_str = s_str.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
    
    # Handle YYYYMM format
    mask_yyyymm = s_str.str.fullmatch(r"\d{6}", na=False)
    out = pd.Series(pd.NaT, index=s.index)
    
    if mask_yyyymm.any():
        out.loc[mask_yyyymm] = pd.to_datetime(s_str.loc[mask_yyyymm], format="%Y%m", errors="coerce")
    
    # Handle other formats
    mask_other = ~mask_yyyymm
    if mask_other.any():
        out.loc[mask_other] = pd.to_datetime(s_str.loc[mask_other], errors="coerce")
    
    return out

combined_df['invoice_period'] = parse_invoice_period(combined_df['invoice_period'])
combined_df = combined_df[combined_df['invoice_period'].notna()].copy()

# Filter to FY2025 only
fy2025_df = combined_df[
    (combined_df['invoice_period'] >= FY2025_START) & 
    (combined_df['invoice_period'] <= FY2025_END)
].copy()

print(f"\nFY2025 ({FY2025_START.strftime('%d/%m/%Y')} - {FY2025_END.strftime('%d/%m/%Y')}): {len(fy2025_df):,} invoices")
print(f"  Unique statements in FY2025: {fy2025_df['statement_id'].nunique():,}")

if len(fy2025_df) == 0:
    print("\n⚠ WARNING: No invoices found in FY2025!")
    print("Available date range in data:")
    print(f"  Min: {combined_df['invoice_period'].min()}")
    print(f"  Max: {combined_df['invoice_period'].max()}")
    exit()

# Show statement-level statistics
print(f"\nStatement-level statistics:")
invoices_per_statement = fy2025_df.groupby('statement_id').size()
print(f"  Average invoices per statement: {invoices_per_statement.mean():.2f}")
print(f"  Median invoices per statement: {invoices_per_statement.median():.0f}")
print(f"  Max invoices in a statement: {invoices_per_statement.max():.0f}")

# ================================================================
# Load decile payment profile
# ================================================================
print("\n" + "="*70)
print("LOADING DECILE PAYMENT PROFILE")
print("="*70)

try:
    with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
        decile_profile = pickle.load(f)
    
    n_deciles = decile_profile['metadata']['n_deciles']
    print(f"✓ Loaded decile payment profile")
    print(f"  Number of deciles: {n_deciles}")
    print(f"  Payment terms: {decile_profile['metadata']['payment_terms_months']:.2f} months")
    print(f"  Method: {decile_profile['metadata']['method']}")
    
    # Check if cd_given_late exists in the profile
    sample_decile = decile_profile['deciles']['decile_0']
    if 'cd_given_late' in sample_decile['delinquency_distribution']:
        print(f"  ✓ Profile contains cd distribution for late payments")
    else:
        print(f"  ⚠ WARNING: Profile does not contain cd_given_late distribution")
        print(f"    Please run the MODIFIED version of 09.3_Payment_profile_ML_clustering.py")
    
except FileNotFoundError:
    print("✗ ERROR: Decile payment profile not found!")
    print("  Expected location: payment_profile/decile_payment_profile.pkl")
    print("  Please run 09.3_Payment_profile_ML_clustering.py first")
    exit()

# ================================================================
# Show decile distribution (using pre-assigned deciles)
# ================================================================
print("\n" + "="*70)
print("DECILE DISTRIBUTION (PRE-ASSIGNED FROM 9.7)")
print("="*70)

print(f"\nInvoice distribution by decile:")
decile_dist = fy2025_df.groupby('decile').agg({
    'total_undiscounted_price': ['count', 'min', 'max', 'mean'],
    'statement_id': 'nunique'
})
decile_dist.columns = ['n_invoices', 'min_amount', 'max_amount', 'avg_amount', 'n_statements']
print(decile_dist.to_string())

# ================================================================
# Simulation function - STATEMENT LEVEL with shared assignments
# ================================================================
def simulate_statement_payments_shared(invoices_df, decile_profile_dict, cd_to_days_map, seed):
    """
    Simulate payment behavior at STATEMENT level with shared late status
    
    This function determines which statements are late ONCE, then applies
    different revenue calculations for with/without discount scenarios.
    
    Parameters:
    - invoices_df: DataFrame with bundled invoice data (has statement_id and decile)
    - decile_profile_dict: Decile payment profile dictionary
    - cd_to_days_map: Dictionary mapping cd level to days overdue
    - seed: Random seed for reproducibility
    
    Returns:
    - Tuple of (with_discount_df, no_discount_df) with shared payment behavior
    """
    np.random.seed(seed)
    
    # Get unique statements (each statement gets ONE payment decision)
    statements = invoices_df.groupby('statement_id').agg({
        'decile': 'first',
        'invoice_period': 'first',
        'total_discounted_price': 'sum',
        'total_undiscounted_price': 'sum',
        'discount_amount': 'sum',
        'customer_type': 'first'
    }).reset_index()
    
    n_statements = len(statements)
    print(f"\nSimulating payment behavior for {n_statements:,} statements...")
    
    # Initialize arrays for simulation results
    is_late_array = np.zeros(n_statements, dtype=bool)
    days_overdue_array = np.zeros(n_statements, dtype=float)
    cd_level_array = np.zeros(n_statements, dtype=int)
    
    # Simulate payment behavior for each statement based on its decile
    for idx, row in statements.iterrows():
        decile_num = row['decile']
        decile_key = f'decile_{int(decile_num)}'
        
        if decile_key not in decile_profile_dict['deciles']:
            decile_key = 'decile_0'
        
        decile_data = decile_profile_dict['deciles'][decile_key]
        
        # STEP 1: Determine if payment is late based on P(late)
        prob_late = decile_data['payment_behavior']['prob_late']
        is_late = np.random.random() < prob_late
        is_late_array[idx] = is_late
        
        if is_late:
            # STEP 2: If late, sample cd level from P(cd | late) distribution
            cd_given_late = decile_data['delinquency_distribution']['cd_given_late']
            
            if cd_given_late:
                cd_levels = list(cd_given_late.keys())
                cd_probs = list(cd_given_late.values())
                
                # Normalize probabilities
                cd_probs = np.array(cd_probs)
                cd_probs = cd_probs / cd_probs.sum()
                
                # Sample cd level
                cd_level = np.random.choice(cd_levels, p=cd_probs)
                cd_level_array[idx] = cd_level
                
                # STEP 3: Use cd level to determine days overdue
                days_overdue = cd_to_days_map.get(cd_level, 90)
                days_overdue_array[idx] = days_overdue
            else:
                cd_level_array[idx] = 0
                days_overdue_array[idx] = 60
        else:
            # On-time payment
            days_overdue_array[idx] = 0
            cd_level_array[idx] = 0
    
    # Add simulation results to statements
    statements['is_late'] = is_late_array
    statements['days_overdue'] = days_overdue_array
    statements['months_overdue'] = days_overdue_array / 30
    statements['cd_level'] = cd_level_array
    
    # ================================================================
    # Create WITH DISCOUNT scenario
    # ================================================================
    with_discount = statements.copy()
    with_discount['principal_amount'] = with_discount['total_discounted_price']
    with_discount['retained_discounts'] = 0
    
    # Calculate interest
    daily_rate = ANNUAL_INTEREST_RATE / 365
    with_discount['interest_charged'] = (
        with_discount['principal_amount'] * daily_rate * with_discount['days_overdue']
    )
    with_discount['credit_card_revenue'] = with_discount['interest_charged'] + with_discount['retained_discounts']
    
    # ================================================================
    # Create NO DISCOUNT scenario
    # ================================================================
    no_discount = statements.copy()
    no_discount['principal_amount'] = no_discount['total_undiscounted_price']
    
    # Retained discounts = discount amount for late statements only
    no_discount['retained_discounts'] = np.where(
        no_discount['is_late'], 
        no_discount['discount_amount'], 
        0
    )
    
    # Calculate interest
    no_discount['interest_charged'] = (
        no_discount['principal_amount'] * daily_rate * no_discount['days_overdue']
    )
    no_discount['credit_card_revenue'] = no_discount['interest_charged'] + no_discount['retained_discounts']
    
    return with_discount, no_discount

# ================================================================
# Run simulation for BOTH scenarios with SHARED payment behavior
# ================================================================
print("\n" + "="*70)
print("RUNNING SIMULATION FOR FY2025")
print("USING CD-BASED TIMING WITH SHARED PAYMENT BEHAVIOR")
print("="*70)

with_discount, no_discount = simulate_statement_payments_shared(
    fy2025_df,
    decile_profile,
    CD_TO_DAYS,
    RANDOM_SEED
)

print("✓ Simulation complete")
print(f"\nVerification - Same statements are late in both scenarios:")
print(f"  With Discount late statements: {with_discount['is_late'].sum():,}")
print(f"  No Discount late statements: {no_discount['is_late'].sum():,}")
print(f"  Match: {(with_discount['is_late'] == no_discount['is_late']).all()}")

# ================================================================
# Summary statistics
# ================================================================
print("\n" + "="*70)
print(f"REVENUE COMPARISON: FY2025 ({FY2025_START.strftime('%d/%m/%Y')} - {FY2025_END.strftime('%d/%m/%Y')})")
print("STATEMENT-BASED WITH RETAINED DISCOUNTS")
print("="*70)

def print_scenario_summary(df, scenario_name):
    """Print summary statistics for a scenario"""
    print(f"\n{scenario_name}")
    print("-" * 70)
    
    total_statements = len(df)
    n_late = df['is_late'].sum()
    n_on_time = total_statements - n_late
    
    # Statement statistics
    print(f"Total statements: {total_statements:,}")
    print(f"  Paid on time (≤{PAYMENT_TERMS_MONTHS:.2f} months): {n_on_time:,} ({n_on_time/total_statements*100:.1f}%)")
    print(f"  Paid late (>{PAYMENT_TERMS_MONTHS:.2f} months): {n_late:,} ({n_late/total_statements*100:.1f}%)")
    
    if n_late > 0:
        print(f"  Avg months overdue (late statements): {df[df['is_late']]['months_overdue'].mean():.2f}")
        print(f"  Avg days overdue (late statements): {df[df['is_late']]['days_overdue'].mean():.1f}")
        print(f"  Avg interest per late statement: ${df[df['is_late']]['interest_charged'].mean():,.2f}")
    
    # cd level distribution (for late payments)
    if n_late > 0:
        print(f"\n  Delinquency level (cd) distribution (late statements only):")
        cd_dist = df[df['is_late']]['cd_level'].value_counts().sort_index()
        for cd, count in cd_dist.items():
            days = CD_TO_DAYS.get(cd, 'N/A')
            print(f"    cd = {cd}: {count:,} ({count/n_late*100:.1f}%) [{days} days]")
    
    # Statement amounts
    print(f"\nTotal Statement Amounts (Customer Obligations):")
    print(f"  Undiscounted total: ${df['total_undiscounted_price'].sum():,.2f}")
    print(f"  Discounted total: ${df['total_discounted_price'].sum():,.2f}")
    print(f"  Discount amount: ${df['discount_amount'].sum():,.2f}")
    
    # Credit Card Revenue with breakdown
    interest_total = df['interest_charged'].sum()
    retained_total = df['retained_discounts'].sum()
    revenue_total = df['credit_card_revenue'].sum()
    
    print(f"\nCredit Card Company Revenue:")
    print(f"  Interest revenue: ${interest_total:,.2f}")
    print(f"  Retained discounts: ${retained_total:,.2f}")
    print(f"  TOTAL REVENUE: ${revenue_total:,.2f}")
    
    # Decile breakdown
    print(f"\nRevenue by Decile:")
    decile_summary = df.groupby('decile').agg({
        'interest_charged': 'sum',
        'retained_discounts': 'sum',
        'credit_card_revenue': 'sum',
        'is_late': 'sum',
        'statement_id': 'count'
    })
    decile_summary.columns = ['interest', 'retained', 'total_revenue', 'n_late', 'n_statements']
    decile_summary['pct_late'] = decile_summary['n_late'] / decile_summary['n_statements'] * 100
    print(decile_summary.to_string())
    
    return {
        'scenario': scenario_name,
        'total_statements': total_statements,
        'n_late': n_late,
        'n_on_time': n_on_time,
        'pct_late': n_late/total_statements*100,
        'avg_months_overdue': df[df['is_late']]['months_overdue'].mean() if n_late > 0 else 0,
        'avg_days_overdue': df[df['is_late']]['days_overdue'].mean() if n_late > 0 else 0,
        'total_undiscounted': df['total_undiscounted_price'].sum(),
        'total_discounted': df['total_discounted_price'].sum(),
        'discount_amount': df['discount_amount'].sum(),
        'interest_revenue': interest_total,
        'retained_discounts': retained_total,
        'total_revenue': revenue_total
    }

# Print summaries
summary_with = print_scenario_summary(with_discount, "FY2025 - WITH EARLY PAYMENT DISCOUNT (STATEMENT-BASED)")
summary_no = print_scenario_summary(no_discount, "FY2025 - NO DISCOUNT (STATEMENT-BASED + RETAINED DISCOUNTS)")

# ================================================================
# Comparison
# ================================================================
print("\n" + "="*70)
print("DIRECT COMPARISON - FY2025 (STATEMENT-BASED)")
print("="*70)

revenue_diff = summary_no['total_revenue'] - summary_with['total_revenue']

print(f"\nCredit Card Revenue:")
print(f"  WITH DISCOUNT:")
print(f"    Interest:          ${summary_with['interest_revenue']:>15,.2f}")
print(f"    Retained Disc.:    ${summary_with['retained_discounts']:>15,.2f}")
print(f"    TOTAL:             ${summary_with['total_revenue']:>15,.2f}")
print(f"\n  NO DISCOUNT:")
print(f"    Interest:          ${summary_no['interest_revenue']:>15,.2f}")
print(f"    Retained Disc.:    ${summary_no['retained_discounts']:>15,.2f}")
print(f"    TOTAL:             ${summary_no['total_revenue']:>15,.2f}")
print(f"\n  Difference:          ${revenue_diff:>15,.2f}")

if revenue_diff > 0:
    pct_more = (revenue_diff / summary_with['total_revenue']) * 100
    print(f"\n✓ NO DISCOUNT generates ${revenue_diff:,.2f} MORE revenue ({pct_more:.1f}%)")
else:
    pct_more = (abs(revenue_diff) / summary_no['total_revenue']) * 100
    print(f"\n✓ WITH DISCOUNT generates ${abs(revenue_diff):,.2f} MORE revenue ({pct_more:.1f}%)")

print(f"\nLate Payment Rates (at STATEMENT level):")
print(f"  With Discount: {summary_with['pct_late']:.1f}% late")
print(f"  No Discount: {summary_no['pct_late']:.1f}% late")
print(f"  (Same statements are late in both scenarios)")

# ================================================================
# Create comparison DataFrame
# ================================================================
comparison_df = pd.DataFrame([summary_with, summary_no])
output_csv = os.path.join(OUTPUT_DIR, '10.0.3_FY2025_cd_timing_comparison_summary_bundled.csv')
comparison_df.to_csv(output_csv, index=False)
print(f"\n✓ Saved comparison summary to: {output_csv}")

# ================================================================
# Save detailed simulations
# ================================================================
output_excel = os.path.join(OUTPUT_DIR, '10.0.3_FY2025_cd_timing_detailed_simulations_bundled.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    with_discount.to_excel(writer, sheet_name='With_Discount', index=False)
    no_discount.to_excel(writer, sheet_name='No_Discount', index=False)
    comparison_df.to_excel(writer, sheet_name='Summary_Comparison', index=False)

print(f"✓ Saved detailed simulations to: {output_excel}")

# ================================================================
# Create cd level analysis
# ================================================================
print("\n" + "="*70)
print("CD LEVEL ANALYSIS (STATEMENT-BASED)")
print("="*70)

cd_analysis = []
for scenario_name, df in [("With Discount", with_discount), ("No Discount", no_discount)]:
    late_statements = df[df['is_late']].copy()
    
    if len(late_statements) > 0:
        cd_summary = late_statements.groupby('cd_level').agg({
            'interest_charged': ['sum', 'mean', 'count'],
            'retained_discounts': 'sum',
            'credit_card_revenue': 'sum',
            'days_overdue': 'mean',
            'principal_amount': 'mean'
        })
        
        cd_summary.columns = ['total_interest', 'avg_interest', 'count', 'total_retained', 
                              'total_revenue', 'avg_days', 'avg_principal']
        cd_summary['scenario'] = scenario_name
        cd_summary['cd_level'] = cd_summary.index
        
        cd_analysis.append(cd_summary.reset_index(drop=True))

if cd_analysis:
    cd_analysis_df = pd.concat(cd_analysis, ignore_index=True)
    cd_analysis_file = os.path.join(OUTPUT_DIR, '10.0.3_cd_level_analysis_bundled.csv')
    cd_analysis_df.to_csv(cd_analysis_file, index=False)
    print(f"✓ Saved cd level analysis to: {cd_analysis_file}")
    
    print("\nCD Level Revenue Summary:")
    print(cd_analysis_df.to_string())

# ================================================================
# Create visualizations
# ================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

TARGET_REVENUE = 1_043_000  # $1.043M target from calibration

# ================================================================
# Visualization: Cumulative Revenue Over Time with Target
# Revenue attributed to INVOICE MONTH (accrual accounting)
# ================================================================
print("\nCreating visualization: Cumulative revenue over time with target...")

fig, ax = plt.subplots(figsize=(16, 8))

# Aggregate by INVOICE month for BOTH scenarios
with_discount_monthly = with_discount.groupby(with_discount['invoice_period'].dt.to_period('M')).agg({
    'credit_card_revenue': 'sum'
}).reset_index()
with_discount_monthly['invoice_period'] = with_discount_monthly['invoice_period'].dt.to_timestamp()
with_discount_monthly['cumulative'] = with_discount_monthly['credit_card_revenue'].cumsum()

no_discount_monthly = no_discount.groupby(no_discount['invoice_period'].dt.to_period('M')).agg({
    'interest_charged': 'sum',
    'retained_discounts': 'sum',
    'credit_card_revenue': 'sum'
}).reset_index()
no_discount_monthly['invoice_period'] = no_discount_monthly['invoice_period'].dt.to_timestamp()
no_discount_monthly['cumulative_interest'] = no_discount_monthly['interest_charged'].cumsum()
no_discount_monthly['cumulative_retained'] = no_discount_monthly['retained_discounts'].cumsum()
no_discount_monthly['cumulative_total'] = no_discount_monthly['credit_card_revenue'].cumsum()

# Plot WITH DISCOUNT (simple line)
ax.plot(with_discount_monthly['invoice_period'], with_discount_monthly['cumulative'], 
        marker='o', linewidth=3, label='With Discount (Interest Only)', 
        color='#70AD47', markersize=8)

# Plot NO DISCOUNT as stacked area to show components
ax.fill_between(no_discount_monthly['invoice_period'], 0, no_discount_monthly['cumulative_interest'],
                 alpha=0.3, color='#4472C4', label='No Discount - Interest')
ax.fill_between(no_discount_monthly['invoice_period'], no_discount_monthly['cumulative_interest'], 
                 no_discount_monthly['cumulative_total'],
                 alpha=0.3, color='#8FAADC', label='No Discount - Retained Discounts')

# Plot NO DISCOUNT total line
ax.plot(no_discount_monthly['invoice_period'], no_discount_monthly['cumulative_total'], 
        marker='s', linewidth=3, label='No Discount (Total)', 
        color='#4472C4', markersize=8)

# Add TARGET LINE at $1.043M
ax.axhline(y=TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target Revenue (${TARGET_REVENUE:,.0f})', alpha=0.8)

# Find when NO DISCOUNT scenario crosses target
cross_idx = None
for i, val in enumerate(no_discount_monthly['cumulative_total']):
    if val >= TARGET_REVENUE:
        cross_idx = i
        break

if cross_idx is not None:
    cross_date = no_discount_monthly['invoice_period'].iloc[cross_idx]
    cross_value = no_discount_monthly['cumulative_total'].iloc[cross_idx]
    
    # Add vertical line at crossing point
    ax.axvline(x=cross_date, color='red', linestyle=':', linewidth=2, alpha=0.5)
    
    # Add annotation
    ax.annotate(f'Target reached:\n{cross_date.strftime("%Y-%m")}\n(${cross_value:,.0f})',
               xy=(cross_date, TARGET_REVENUE),
               xytext=(20, 30), textcoords='offset points',
               fontsize=11, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='red', linewidth=2),
               arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

ax.set_title(f'FY2025 Cumulative Revenue Over Time (Statement-Based)\nInterest + Retained Discounts vs Target\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Invoice Month', fontsize=14)
ax.set_ylabel('Cumulative Revenue ($)', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

# Add final values
final_with = with_discount_monthly['cumulative'].iloc[-1]
final_no = no_discount_monthly['cumulative_total'].iloc[-1]
final_interest = no_discount_monthly['cumulative_interest'].iloc[-1]
final_retained = no_discount_monthly['cumulative_retained'].iloc[-1]

ax.annotate(f'With Discount\n${final_with:,.0f}', 
            xy=(with_discount_monthly['invoice_period'].iloc[-1], final_with),
            xytext=(10, -30), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#70AD47',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#70AD47', linewidth=2))

ax.annotate(f'No Discount Total\n${final_no:,.0f}\n(Int: ${final_interest:,.0f}\n+Ret: ${final_retained:,.0f})', 
            xy=(no_discount_monthly['invoice_period'].iloc[-1], final_no),
            xytext=(10, 10), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#4472C4',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#4472C4', linewidth=2))

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, '10.0.3_cumulative_revenue_with_target_and_components_bundled.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_path}")
plt.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE - STATEMENT-BASED REVENUE ESTIMATION")
print("="*70)
print(f"\nAll results saved to folder: {OUTPUT_DIR}/")
print("Files created:")
print(f"  1. 10.0.3_FY2025_cd_timing_comparison_summary_bundled.csv")
print(f"  2. 10.0.3_FY2025_cd_timing_detailed_simulations_bundled.xlsx")
print(f"  3. 10.0.3_cd_level_analysis_bundled.csv")
print(f"  4. 10.0.3_cumulative_revenue_with_target_and_components_bundled.png")
print("\nKey approach:")
print(f"  - Uses pre-bundled statements from script 9.7")
print(f"  - {len(with_discount):,} statements analyzed")
print(f"  - SAME payment behavior in both scenarios (fair comparison)")
print(f"  - Decile-specific late payment probabilities")
print(f"  - cd level sampled from P(cd | late, decile)")
print(f"  - Payment timing determined by cd level")
print(f"  - Revenue attributed to INVOICE MONTH (accrual accounting)")
print(f"  - NO DISCOUNT revenue = Interest + Retained Discounts")
print("="*70)