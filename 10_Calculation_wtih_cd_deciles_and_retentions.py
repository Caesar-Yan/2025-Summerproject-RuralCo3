'''
Docstring for 10_Calculation_wtih_cd_deciles_and_retentions

this script makes estiamtes of revenue generated under discount and no-discount scenarios,
i.e. the current and proposed models.
late rates and how late payments are when late are estimated using the real data 
from group2's data. these profiles can be found in payment profile folder under ruralco3

inputs:
- ats_grouped_transformed_with_discounts.csv
- invoice_grouped_transformed_with_discounts.csv

outputs:
- 10_FY2025_cd_timing_comparison_summary.csv
- 10_FY2025_cd_timing_detailed_simulations.xlsx
- 10_cd_level_analysis.csv
- 10_cumulative_revenue_with_target_and_components.png

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
print("CD LEVEL TO PAYMENT TIMING MAPPING")
print("="*70)
for cd, days in sorted(CD_TO_DAYS.items()):
    months = days / 30
    print(f"  cd = {cd}: {days} days ({months:.1f} months) overdue")

np.random.seed(RANDOM_SEED)

# ================================================================
# Load invoice data
# ================================================================
print("\n" + "="*70)
print("LOADING INVOICE DATA")
print("="*70)

# Load combined invoice data
ats_grouped = pd.read_csv(data_cleaning_dir / 'ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv(data_cleaning_dir / 'invoice_grouped_transformed_with_discounts.csv')

# Combine datasets
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"Total invoices loaded: {len(combined_df):,}")

# ================================================================
# Filter out negative undiscounted prices
# ================================================================
initial_count = len(combined_df)
combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()
filtered_count = initial_count - len(combined_df)
if filtered_count > 0:
    print(f"⚠ Filtered out {filtered_count:,} invoices with negative undiscounted prices")
print(f"Remaining invoices: {len(combined_df):,}")

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

if len(fy2025_df) == 0:
    print("\n⚠ WARNING: No invoices found in FY2025!")
    print("Available date range in data:")
    print(f"  Min: {combined_df['invoice_period'].min()}")
    print(f"  Max: {combined_df['invoice_period'].max()}")
    exit()

# ================================================================
# Load decile payment profile
# ================================================================
print("\n" + "="*70)
print("LOADING DECILE PAYMENT PROFILE")
print("="*70)

try:
    # Try to load MODIFIED profile first
    try:
        with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
            decile_profile = pickle.load(f)
        profile_version = "MODIFIED"
    except FileNotFoundError:
        with open('Payment Profile/decile_payment_profile.pkl', 'rb') as f:
            decile_profile = pickle.load(f)
        profile_version = "ORIGINAL"
    
    n_deciles = decile_profile['metadata']['n_deciles']
    print(f"✓ Loaded decile payment profile ({profile_version})")
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
    print("  Expected location: Payment Profile/decile_payment_profile_MODIFIED.pkl")
    print("  OR: Payment Profile/decile_payment_profile.pkl")
    print("  Please run 09.3_Payment_profile_ML_clustering.py first")
    exit()

# ================================================================
# Map invoices to deciles based on total_undiscounted_price
# ================================================================
print("\n" + "="*70)
print("MAPPING INVOICES TO DECILES")
print("="*70)

# Sort by undiscounted price and assign deciles
fy2025_df = fy2025_df.sort_values('total_undiscounted_price').reset_index(drop=True)

# Assign deciles using qcut to match training data approach
fy2025_df['decile'] = pd.qcut(
    fy2025_df['total_undiscounted_price'], 
    q=n_deciles, 
    labels=False,
    duplicates='drop'
)

print(f"✓ Mapped {len(fy2025_df):,} invoices to deciles")

# Show distribution
print(f"\nInvoice distribution by decile:")
decile_dist = fy2025_df.groupby('decile').agg({
    'total_undiscounted_price': ['count', 'min', 'max', 'mean']
})
decile_dist.columns = ['count', 'min_amount', 'max_amount', 'avg_amount']
print(decile_dist.to_string())

# ================================================================
# Simulation function using decile profiles with cd-based timing
# ================================================================
def simulate_with_cd_timing(invoices_df, decile_profile_dict, discount_scenario, cd_to_days_map):
    """
    Simulate invoice payments using decile-specific payment behavior
    with cd level determining payment timing
    
    MODIFICATION: For NO DISCOUNT scenario, add retained discounts to revenue
    
    Parameters:
    - invoices_df: DataFrame with invoice data (must have 'decile' column)
    - decile_profile_dict: Decile payment profile dictionary
    - discount_scenario: 'with_discount' or 'no_discount'
    - cd_to_days_map: Dictionary mapping cd level to days overdue
    
    Returns:
    - DataFrame with simulated payment details
    """
    simulated = invoices_df.copy()
    n_invoices = len(simulated)
    
    # Initialize arrays for simulation results
    is_late_array = np.zeros(n_invoices, dtype=bool)
    days_overdue_array = np.zeros(n_invoices, dtype=float)
    cd_level_array = np.zeros(n_invoices, dtype=int)
    
    print(f"\nSimulating {n_invoices:,} invoices...")
    
    # Simulate payment behavior for each invoice based on its decile
    for idx, row in simulated.iterrows():
        decile_num = row['decile']
        decile_key = f'decile_{int(decile_num)}'
        
        if decile_key not in decile_profile_dict['deciles']:
            print(f"⚠ Warning: Decile {decile_num} not found in profile, using decile 0")
            decile_key = 'decile_0'
        
        decile_data = decile_profile_dict['deciles'][decile_key]
        
        # ================================================================
        # STEP 1: Determine if payment is late based on P(late)
        # ================================================================
        prob_late = decile_data['payment_behavior']['prob_late']
        is_late = np.random.random() < prob_late
        is_late_array[idx] = is_late
        
        if is_late:
            # ================================================================
            # STEP 2: If late, sample cd level from P(cd | late) distribution
            # ================================================================
            cd_given_late = decile_data['delinquency_distribution']['cd_given_late']
            
            if cd_given_late:
                cd_levels = list(cd_given_late.keys())
                cd_probs = list(cd_given_late.values())
                
                # Normalize probabilities (in case they don't sum to 1)
                cd_probs = np.array(cd_probs)
                cd_probs = cd_probs / cd_probs.sum()
                
                # Sample cd level
                cd_level = np.random.choice(cd_levels, p=cd_probs)
                cd_level_array[idx] = cd_level
                
                # ================================================================
                # STEP 3: Use cd level to determine days overdue
                # ================================================================
                if cd_level in cd_to_days_map:
                    days_overdue = cd_to_days_map[cd_level]
                else:
                    # Default fallback if cd level not in mapping
                    print(f"⚠ Warning: cd level {cd_level} not in mapping, using 90 days")
                    days_overdue = 90
                
                days_overdue_array[idx] = days_overdue
            else:
                # No cd distribution available, use default
                cd_level_array[idx] = 0
                days_overdue_array[idx] = 60  # Default to 60 days
        else:
            # On-time payment
            days_overdue_array[idx] = 0
            cd_level_array[idx] = 0
    
    # Add simulation results to dataframe
    simulated['is_late'] = is_late_array
    simulated['days_overdue'] = days_overdue_array
    simulated['months_overdue'] = days_overdue_array / 30  # Convert to months
    simulated['cd_level'] = cd_level_array
    
    # Calculate dates
    simulated['due_date'] = simulated['invoice_period']
    simulated['payment_date'] = simulated['due_date'] + pd.to_timedelta(simulated['days_overdue'], unit='D')
    
    # ================================================================
    # Calculate amounts based on discount scenario
    # ================================================================
    if discount_scenario == 'with_discount':
        # With discount: On-time payers get discount, late payers pay discounted amount + interest
        simulated['principal_amount'] = simulated['total_discounted_price']
        simulated['paid_on_time'] = ~simulated['is_late']
        simulated['discount_applied'] = simulated['discount_amount']
        
        # No retained discounts in this scenario
        simulated['retained_discounts'] = 0
        
    else:  # no_discount
        # No discount: Everyone pays full undiscounted amount
        simulated['principal_amount'] = simulated['total_undiscounted_price']
        simulated['paid_on_time'] = ~simulated['is_late']
        simulated['discount_applied'] = 0
        
        # MODIFICATION: Late invoices don't get discount, so we retain that money
        simulated['retained_discounts'] = np.where(
            simulated['is_late'],
            simulated['discount_amount'],  # Retain discount for late invoices
            0  # On-time invoices would have gotten discount (but there is none)
        )
    
    # ================================================================
    # Calculate interest charges
    # ================================================================
    daily_rate = ANNUAL_INTEREST_RATE / 365
    
    # Interest = Principal × Daily Rate × Days Overdue
    simulated['interest_charged'] = (
        simulated['principal_amount'] * 
        daily_rate * 
        simulated['days_overdue']
    )
    
    # ================================================================
    # Calculate total revenue (interest + retained discounts)
    # ================================================================
    simulated['credit_card_revenue'] = simulated['interest_charged'] + simulated['retained_discounts']
    
    # Track total amounts
    simulated['total_invoice_amount_discounted'] = simulated['total_discounted_price']
    simulated['total_invoice_amount_undiscounted'] = simulated['total_undiscounted_price']
    
    return simulated

# ================================================================
# Run both scenarios for FY2025
# ================================================================
print("\n" + "="*70)
print("RUNNING SIMULATIONS FOR FY2025 USING CD-BASED TIMING")
print("WITH RETAINED DISCOUNTS FOR LATE PAYMENTS")
print("="*70)

# Scenario 1: With early payment discount
print("\nScenario 1: With early payment discount...")
with_discount = simulate_with_cd_timing(
    fy2025_df, 
    decile_profile, 
    discount_scenario='with_discount',
    cd_to_days_map=CD_TO_DAYS
)

# Scenario 2: No discount
print("Scenario 2: No discount offered...")
no_discount = simulate_with_cd_timing(
    fy2025_df, 
    decile_profile, 
    discount_scenario='no_discount',
    cd_to_days_map=CD_TO_DAYS
)

print("✓ Simulations complete")

# ================================================================
# Summary statistics
# ================================================================
print("\n" + "="*70)
print(f"REVENUE COMPARISON: FY2025 ({FY2025_START.strftime('%d/%m/%Y')} - {FY2025_END.strftime('%d/%m/%Y')})")
print("WITH RETAINED DISCOUNTS")
print("="*70)

def print_scenario_summary(df, scenario_name):
    """Print summary statistics for a scenario"""
    print(f"\n{scenario_name}")
    print("-" * 70)
    
    total_invoices = len(df)
    n_late = df['is_late'].sum()
    n_on_time = total_invoices - n_late
    
    # Invoice statistics
    print(f"Total invoices: {total_invoices:,}")
    print(f"  Paid on time (≤{PAYMENT_TERMS_MONTHS:.2f} months): {n_on_time:,} ({n_on_time/total_invoices*100:.1f}%)")
    print(f"  Paid late (>{PAYMENT_TERMS_MONTHS:.2f} months): {n_late:,} ({n_late/total_invoices*100:.1f}%)")
    
    if n_late > 0:
        print(f"  Avg months overdue (late invoices): {df[df['is_late']]['months_overdue'].mean():.2f}")
        print(f"  Avg days overdue (late invoices): {df[df['is_late']]['days_overdue'].mean():.1f}")
        print(f"  Avg interest per late invoice: ${df[df['is_late']]['interest_charged'].mean():,.2f}")
    
    # cd level distribution (for late payments)
    if n_late > 0:
        print(f"\n  Delinquency level (cd) distribution (late payments only):")
        cd_dist = df[df['is_late']]['cd_level'].value_counts().sort_index()
        for cd, count in cd_dist.items():
            days = CD_TO_DAYS.get(cd, 'N/A')
            print(f"    cd = {cd}: {count:,} ({count/n_late*100:.1f}%) [{days} days]")
        
        # Show days overdue distribution
        print(f"\n  Days overdue distribution (late payments only):")
        days_dist = df[df['is_late']]['days_overdue'].value_counts().sort_index()
        for days, count in days_dist.items():
            print(f"    {days:.0f} days: {count:,} ({count/n_late*100:.1f}%)")
    
    # Invoice amounts (what customers owe - not your revenue)
    print(f"\nTotal Invoice Amounts (Customer Obligations):")
    print(f"  Undiscounted invoice total: ${df['total_undiscounted_price'].sum():,.2f}")
    print(f"  Discounted invoice total: ${df['total_discounted_price'].sum():,.2f}")
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
        'total_undiscounted_price': 'count'
    })
    decile_summary.columns = ['interest', 'retained', 'total_revenue', 'n_late', 'n_invoices']
    decile_summary['pct_late'] = decile_summary['n_late'] / decile_summary['n_invoices'] * 100
    print(decile_summary.to_string())
    
    return {
        'scenario': scenario_name,
        'total_invoices': total_invoices,
        'n_late': n_late,
        'n_on_time': n_on_time,
        'pct_late': n_late/total_invoices*100,
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
summary_with = print_scenario_summary(with_discount, "FY2025 - WITH EARLY PAYMENT DISCOUNT (CD-BASED TIMING)")
summary_no = print_scenario_summary(no_discount, "FY2025 - NO DISCOUNT (CD-BASED TIMING + RETAINED DISCOUNTS)")

# ================================================================
# Comparison
# ================================================================
print("\n" + "="*70)
print("DIRECT COMPARISON - FY2025 (CD-BASED TIMING + RETAINED DISCOUNTS)")
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

print(f"\nLate Payment Rates:")
print(f"  With Discount: {summary_with['pct_late']:.1f}% late")
print(f"  No Discount: {summary_no['pct_late']:.1f}% late")

# ================================================================
# Create comparison DataFrame
# ================================================================
comparison_df = pd.DataFrame([summary_with, summary_no])
output_csv = os.path.join(OUTPUT_DIR, '10_FY2025_cd_timing_comparison_summary.csv')
comparison_df.to_csv(output_csv, index=False)
print(f"\n✓ Saved comparison summary to: {output_csv}")

# ================================================================
# Save detailed simulations
# ================================================================
output_excel = os.path.join(OUTPUT_DIR, '10_FY2025_cd_timing_detailed_simulations.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    with_discount.to_excel(writer, sheet_name='With_Discount', index=False)
    no_discount.to_excel(writer, sheet_name='No_Discount', index=False)
    comparison_df.to_excel(writer, sheet_name='Summary_Comparison', index=False)

print(f"✓ Saved detailed simulations to: {output_excel}")

# ================================================================
# Create cd level analysis
# ================================================================
print("\n" + "="*70)
print("CD LEVEL ANALYSIS (WITH RETAINED DISCOUNTS)")
print("="*70)

cd_analysis = []
for scenario_name, df in [("With Discount", with_discount), ("No Discount", no_discount)]:
    late_payments = df[df['is_late']].copy()
    
    if len(late_payments) > 0:
        cd_summary = late_payments.groupby('cd_level').agg({
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
    cd_analysis_file = os.path.join(OUTPUT_DIR, '10_cd_level_analysis.csv')
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
# Visualization 1: Cumulative Revenue Over Time with Target
# ================================================================
print("\nCreating visualization: Cumulative revenue over time with target...")

fig, ax = plt.subplots(figsize=(16, 8))

# Aggregate by payment month for BOTH scenarios
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

# Plot WITH DISCOUNT (simple line)
ax.plot(with_discount_monthly['payment_date'], with_discount_monthly['cumulative'], 
        marker='o', linewidth=3, label='With Discount (Interest Only)', 
        color='#70AD47', markersize=8)

# Plot NO DISCOUNT as stacked area to show components
ax.fill_between(no_discount_monthly['payment_date'], 0, no_discount_monthly['cumulative_interest'],
                 alpha=0.3, color='#4472C4', label='No Discount - Interest')
ax.fill_between(no_discount_monthly['payment_date'], no_discount_monthly['cumulative_interest'], 
                 no_discount_monthly['cumulative_total'],
                 alpha=0.3, color='#8FAADC', label='No Discount - Retained Discounts')

# Plot NO DISCOUNT total line
ax.plot(no_discount_monthly['payment_date'], no_discount_monthly['cumulative_total'], 
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
    cross_date = no_discount_monthly['payment_date'].iloc[cross_idx]
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

ax.set_title(f'FY2025 Cumulative Revenue Over Time (cd-based timing)\nInterest + Retained Discounts vs Target\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Payment Month', fontsize=14)
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
            xy=(with_discount_monthly['payment_date'].iloc[-1], final_with),
            xytext=(10, -30), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#70AD47',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#70AD47', linewidth=2))

ax.annotate(f'No Discount Total\n${final_no:,.0f}\n(Int: ${final_interest:,.0f}\n+Ret: ${final_retained:,.0f})', 
            xy=(no_discount_monthly['payment_date'].iloc[-1], final_no),
            xytext=(10, 10), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#4472C4',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#4472C4', linewidth=2))

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, '10_cumulative_revenue_with_target_and_components.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_path}")
plt.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE - FY2025 WITH CD-BASED TIMING + RETAINED DISCOUNTS")
print("="*70)
print(f"\nAll results saved to folder: {OUTPUT_DIR}/")
print("Files created:")
print(f"  1. FY2025_cd_timing_comparison_summary.csv - Summary comparison table")
print(f"  2. FY2025_cd_timing_detailed_simulations.xlsx - Full simulation data")
print(f"  3. cd_level_analysis.csv - Revenue breakdown by cd level")
print(f"  4. cumulative_revenue_with_target_and_components.png - Visualization")
print("\nSimulation approach:")
print(f"  - {n_deciles} deciles based on invoice amount")
print(f"  - Decile-specific late payment probabilities")
print(f"  - cd level sampled from P(cd | late, decile)")
print(f"  - Payment timing determined by cd level")
print(f"  - NO DISCOUNT revenue = Interest + Retained Discounts")
print("="*70)