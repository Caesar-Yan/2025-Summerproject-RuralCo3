'''
Docstring for 10.7_Calculation_with_calibrated_baseline

This script makes estimates of revenue generated under discount and no-discount scenarios,
using CALIBRATED baseline late payment rates (from 10.6) with seasonal adjustments.
The calibrated baseline is derived by working backwards from historical revenue target.

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
print("SIMULATION WITH CALIBRATED BASELINE LATE RATES")
print("="*70)
print("CD LEVEL TO PAYMENT TIMING MAPPING")
for cd, days in sorted(CD_TO_DAYS.items()):
    months = days / 30
    print(f"  cd = {cd}: {days} days ({months:.1f} months) overdue")

np.random.seed(RANDOM_SEED)

# ================================================================
# Load calibrated baseline from 10.6
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
    print(f"  (Original snapshot was ~2%, calibrated is {calibrated_baseline*100:.2f}%)")
    
except FileNotFoundError:
    print("ERROR: Calibrated baseline not found!")
    print("Please run 10.6_Reverse_calibration_from_target.py first")
    exit(1)

# ================================================================
# Load reconstructed seasonal late payment rates
# ================================================================
print("\n" + "="*70)
print("LOADING SEASONAL LATE PAYMENT RATES")
print("="*70)

seasonal_rates = pd.read_csv(visualisations_dir / "09.6_reconstructed_late_payment_rates.csv")
seasonal_rates['invoice_period'] = pd.to_datetime(seasonal_rates['invoice_period'])

print(f"✓ Loaded seasonal late payment rates for {len(seasonal_rates)} months")
print(f"Date range: {seasonal_rates['invoice_period'].min()} to {seasonal_rates['invoice_period'].max()}")

# ================================================================
# Calculate seasonal adjustment factors (November baseline)
# ================================================================
print("\n" + "="*70)
print("CALCULATING SEASONAL ADJUSTMENT FACTORS")
print("="*70)

# Get November 2025 reconstructed rate
november_2025_row = seasonal_rates[
    (seasonal_rates['invoice_period'].dt.year == 2025) & 
    (seasonal_rates['invoice_period'].dt.month == 11)
]

if len(november_2025_row) == 0:
    print("ERROR: November 2025 not found in seasonal rates")
    exit(1)

november_reconstructed_rate = november_2025_row['reconstructed_late_rate_pct'].values[0] / 100

print(f"November 2025 reconstructed rate (from spending): {november_reconstructed_rate*100:.2f}%")
print(f"This provides RELATIVE seasonal adjustments")

# Calculate seasonal adjustment factors
seasonal_adjustment_factors = {}

for _, row in seasonal_rates.iterrows():
    year_month = (row['invoice_period'].year, row['invoice_period'].month)
    month_rate = row['reconstructed_late_rate_pct'] / 100
    
    # Adjustment factor = Month Rate / November Rate
    adjustment_factor = month_rate / november_reconstructed_rate
    
    seasonal_adjustment_factors[year_month] = adjustment_factor

print(f"\nSeasonal Adjustment Factors (relative to November 2025):")
print(f"{'Month':<12} {'Reconstructed Rate':<20} {'Adjustment Factor':<20} {'Effect'}")
print("-" * 80)
for year_month, factor in sorted(seasonal_adjustment_factors.items()):
    month_rate = factor * november_reconstructed_rate
    effect = "Increases decile rates" if factor > 1 else "Decreases decile rates" if factor < 1 else "No change"
    print(f"{year_month[0]}-{year_month[1]:02d}      {month_rate*100:>8.2f}%              {factor:>8.3f}x          {effect}")

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

# Filter out negative undiscounted prices
initial_count = len(combined_df)
combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()
filtered_count = initial_count - len(combined_df)
if filtered_count > 0:
    print(f"⚠ Filtered out {filtered_count:,} invoices with negative undiscounted prices")
print(f"Remaining invoices: {len(combined_df):,}")

# Parse and filter dates for FY2025
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
    exit()

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
    print(f"  Using CALIBRATED base rates × seasonal adjustment factors")
    
    # Check if cd_given_late exists
    sample_decile = decile_profile['deciles']['decile_0']
    if 'cd_given_late' in sample_decile['delinquency_distribution']:
        print(f"  ✓ Profile contains cd distribution for late payments")
    else:
        print(f"  ⚠ WARNING: Profile does not contain cd_given_late distribution")
    
    # Display CALIBRATED decile base rates
    print(f"\nCalibrated Decile Base Late Payment Rates (November baseline):")
    old_baseline_snapshot = 0.02
    for i in range(n_deciles):
        decile_key = f'decile_{i}'
        if decile_key in decile_profile['deciles']:
            snapshot_rate = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
            calibrated_rate = snapshot_rate * scaling_factor
            print(f"  Decile {i}: {snapshot_rate*100:.2f}% (snapshot) → {calibrated_rate*100:.2f}% (calibrated)")
    
except FileNotFoundError:
    print("✗ ERROR: Decile payment profile not found!")
    exit()

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
# Simulation function with CALIBRATED baseline
# ================================================================
def simulate_with_calibrated_baseline(invoices_df, decile_profile_dict, 
                                     seasonal_factors_dict, calibrated_baseline,
                                     scaling_factor, discount_scenario, cd_to_days_map):
    """
    Simulate invoice payments using CALIBRATED baseline with seasonal adjustments
    
    KEY APPROACH:
    - Decile snapshot rates are SCALED by the calibration factor
    - Calibrated Decile Base Rate = Snapshot Rate × Scaling Factor
    - Then apply seasonal adjustment: Adjusted Rate = Calibrated Base × Seasonal Factor
    
    Parameters:
    - invoices_df: DataFrame with invoice data
    - decile_profile_dict: Decile payment profile dictionary
    - seasonal_factors_dict: Dictionary mapping (year, month) to adjustment factor
    - calibrated_baseline: Calibrated November baseline rate
    - scaling_factor: Factor to scale snapshot rates to calibrated rates
    - discount_scenario: 'with_discount' or 'no_discount'
    - cd_to_days_map: Dictionary mapping cd level to days overdue
    
    Returns:
    - DataFrame with simulated payment details
    """
    simulated = invoices_df.copy()
    n_invoices = len(simulated)
    
    # Initialize arrays
    is_late_array = np.zeros(n_invoices, dtype=bool)
    days_overdue_array = np.zeros(n_invoices, dtype=float)
    cd_level_array = np.zeros(n_invoices, dtype=int)
    decile_snapshot_rate_array = np.zeros(n_invoices, dtype=float)
    decile_calibrated_rate_array = np.zeros(n_invoices, dtype=float)
    seasonal_factor_array = np.zeros(n_invoices, dtype=float)
    adjusted_late_rate_array = np.zeros(n_invoices, dtype=float)
    
    print(f"\nSimulating {n_invoices:,} invoices with CALIBRATED baseline rates...")
    
    # Track unique months
    months_used = set()
    
    # Simulate payment behavior
    for idx, row in simulated.iterrows():
        decile_num = row['decile']
        invoice_date = row['invoice_period']
        year_month = (invoice_date.year, invoice_date.month)
        
        # ================================================================
        # STEP 1: Get decile's SNAPSHOT rate (from profile)
        # ================================================================
        decile_key = f'decile_{int(decile_num)}'
        
        if decile_key not in decile_profile_dict['deciles']:
            decile_key = 'decile_0'
        
        decile_data = decile_profile_dict['deciles'][decile_key]
        decile_snapshot_rate = decile_data['payment_behavior']['prob_late']
        decile_snapshot_rate_array[idx] = decile_snapshot_rate
        
        # ================================================================
        # STEP 2: SCALE to CALIBRATED base rate
        # ================================================================
        decile_calibrated_rate = decile_snapshot_rate * scaling_factor
        decile_calibrated_rate_array[idx] = decile_calibrated_rate
        
        # ================================================================
        # STEP 3: Get SEASONAL adjustment factor for this month
        # ================================================================
        if year_month in seasonal_factors_dict:
            seasonal_factor = seasonal_factors_dict[year_month]
            months_used.add(year_month)
        else:
            seasonal_factor = 1.0
            print(f"⚠ Warning: No seasonal factor for {year_month}, using 1.0")
        
        seasonal_factor_array[idx] = seasonal_factor
        
        # ================================================================
        # STEP 4: Calculate FINAL adjusted late rate
        # ================================================================
        # Adjusted Rate = Calibrated Base Rate × Seasonal Factor
        adjusted_late_rate = decile_calibrated_rate * seasonal_factor
        
        # Cap at 100%
        adjusted_late_rate = min(adjusted_late_rate, 1.0)
        
        adjusted_late_rate_array[idx] = adjusted_late_rate
        
        # Determine if payment is late
        is_late = np.random.random() < adjusted_late_rate
        is_late_array[idx] = is_late
        
        if is_late:
            # ================================================================
            # STEP 5: If late, get cd distribution from decile profile
            # ================================================================
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
                
                # ================================================================
                # STEP 6: Use cd level to determine days overdue
                # ================================================================
                if cd_level in cd_to_days_map:
                    days_overdue = cd_to_days_map[cd_level]
                else:
                    days_overdue = 90
                
                days_overdue_array[idx] = days_overdue
            else:
                cd_level_array[idx] = 0
                days_overdue_array[idx] = 60
        else:
            # On-time payment
            days_overdue_array[idx] = 0
            cd_level_array[idx] = 0
    
    print(f"  ✓ Applied calibrated rates with seasonal adjustments from {len(months_used)} unique months")
    
    # Add simulation results to dataframe
    simulated['is_late'] = is_late_array
    simulated['days_overdue'] = days_overdue_array
    simulated['months_overdue'] = days_overdue_array / 30
    simulated['cd_level'] = cd_level_array
    simulated['decile_snapshot_rate'] = decile_snapshot_rate_array
    simulated['decile_calibrated_rate'] = decile_calibrated_rate_array
    simulated['seasonal_adjustment_factor'] = seasonal_factor_array
    simulated['adjusted_late_rate'] = adjusted_late_rate_array
    
    # Calculate dates
    simulated['due_date'] = simulated['invoice_period']
    simulated['payment_date'] = simulated['due_date'] + pd.to_timedelta(simulated['days_overdue'], unit='D')
    
    # Calculate amounts
    if discount_scenario == 'with_discount':
        simulated['principal_amount'] = simulated['total_discounted_price']
        simulated['paid_on_time'] = ~simulated['is_late']
        simulated['discount_applied'] = simulated['discount_amount']
        simulated['retained_discounts'] = 0
        
    else:  # no_discount
        simulated['principal_amount'] = simulated['total_undiscounted_price']
        simulated['paid_on_time'] = ~simulated['is_late']
        simulated['discount_applied'] = 0
        simulated['retained_discounts'] = np.where(
            simulated['is_late'],
            simulated['discount_amount'],
            0
        )
    
    # Calculate interest
    daily_rate = ANNUAL_INTEREST_RATE / 365
    simulated['interest_charged'] = (
        simulated['principal_amount'] * 
        daily_rate * 
        simulated['days_overdue']
    )
    
    # Calculate total revenue
    simulated['credit_card_revenue'] = simulated['interest_charged'] + simulated['retained_discounts']
    
    # Track total amounts
    simulated['total_invoice_amount_discounted'] = simulated['total_discounted_price']
    simulated['total_invoice_amount_undiscounted'] = simulated['total_undiscounted_price']
    
    return simulated

# ================================================================
# Run both scenarios with calibrated baseline
# ================================================================
print("\n" + "="*70)
print("RUNNING SIMULATIONS WITH CALIBRATED BASELINE")
print("="*70)

# Scenario 1: With discount
print("\nScenario 1: With early payment discount + calibrated baseline...")
with_discount = simulate_with_calibrated_baseline(
    fy2025_df, 
    decile_profile, 
    seasonal_adjustment_factors,
    calibrated_baseline,
    scaling_factor,
    discount_scenario='with_discount',
    cd_to_days_map=CD_TO_DAYS
)

# Scenario 2: No discount
print("Scenario 2: No discount + calibrated baseline...")
no_discount = simulate_with_calibrated_baseline(
    fy2025_df, 
    decile_profile, 
    seasonal_adjustment_factors,
    calibrated_baseline,
    scaling_factor,
    discount_scenario='no_discount',
    cd_to_days_map=CD_TO_DAYS
)

print("✓ Simulations complete")

# ================================================================
# Save decile seasonal adjustments summary
# ================================================================
print("\n" + "="*70)
print("CREATING DECILE SEASONAL ADJUSTMENTS SUMMARY")
print("="*70)

decile_seasonal_summary = []

for decile_num in range(n_deciles):
    decile_key = f'decile_{decile_num}'
    if decile_key in decile_profile['deciles']:
        snapshot_rate = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
        calibrated_base_rate = snapshot_rate * scaling_factor
        
        for year_month, factor in sorted(seasonal_adjustment_factors.items()):
            adjusted_rate = calibrated_base_rate * factor
            
            decile_seasonal_summary.append({
                'decile': decile_num,
                'year': year_month[0],
                'month': year_month[1],
                'snapshot_rate_pct': snapshot_rate * 100,
                'calibrated_base_rate_pct': calibrated_base_rate * 100,
                'seasonal_factor': factor,
                'adjusted_late_rate_pct': adjusted_rate * 100
            })

decile_seasonal_df = pd.DataFrame(decile_seasonal_summary)
decile_seasonal_file = os.path.join(OUTPUT_DIR, '10.7_decile_seasonal_adjustments.csv')
decile_seasonal_df.to_csv(decile_seasonal_file, index=False)
print(f"✓ Saved decile seasonal adjustments to: {decile_seasonal_file}")

print("\nSample adjustments (Decile 0):")
print(decile_seasonal_df[decile_seasonal_df['decile'] == 0].head(12).to_string(index=False))

# ================================================================
# Summary statistics
# ================================================================
print("\n" + "="*70)
print(f"REVENUE COMPARISON: FY2025 ({FY2025_START.strftime('%d/%m/%Y')} - {FY2025_END.strftime('%d/%m/%Y')})")
print("WITH CALIBRATED BASELINE RATES")
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
    
    # Calibration summary
    print(f"\nCalibration Summary:")
    print(f"  Avg snapshot rate (deciles): {df['decile_snapshot_rate'].mean()*100:.2f}%")
    print(f"  Avg calibrated base rate: {df['decile_calibrated_rate'].mean()*100:.2f}%")
    print(f"  Avg seasonal factor applied: {df['seasonal_adjustment_factor'].mean():.3f}")
    print(f"  Avg adjusted late rate: {df['adjusted_late_rate'].mean()*100:.2f}%")
    print(f"  Actual late rate (simulated): {n_late/total_invoices*100:.2f}%")
    
    # cd level distribution
    if n_late > 0:
        print(f"\n  Delinquency level (cd) distribution (late payments only):")
        cd_dist = df[df['is_late']]['cd_level'].value_counts().sort_index()
        for cd, count in cd_dist.items():
            days = CD_TO_DAYS.get(cd, 'N/A')
            print(f"    cd = {cd}: {count:,} ({count/n_late*100:.1f}%) [{days} days]")
    
    # Invoice amounts
    print(f"\nTotal Invoice Amounts (Customer Obligations):")
    print(f"  Undiscounted invoice total: ${df['total_undiscounted_price'].sum():,.2f}")
    print(f"  Discounted invoice total: ${df['total_discounted_price'].sum():,.2f}")
    print(f"  Discount amount: ${df['discount_amount'].sum():,.2f}")
    
    # Credit Card Revenue
    interest_total = df['interest_charged'].sum()
    retained_total = df['retained_discounts'].sum()
    revenue_total = df['credit_card_revenue'].sum()
    
    print(f"\nCredit Card Company Revenue:")
    print(f"  Interest revenue: ${interest_total:,.2f}")
    print(f"  Retained discounts: ${retained_total:,.2f}")
    print(f"  TOTAL REVENUE: ${revenue_total:,.2f}")
    
    # Monthly breakdown
    print(f"\nRevenue by Month (with calibrated seasonal adjustments):")
    monthly_revenue = df.groupby(df['invoice_period'].dt.to_period('M')).agg({
        'interest_charged': 'sum',
        'retained_discounts': 'sum',
        'credit_card_revenue': 'sum',
        'is_late': ['sum', 'count'],
        'seasonal_adjustment_factor': 'first',
        'adjusted_late_rate': 'mean'
    })
    monthly_revenue.columns = ['interest', 'retained', 'total_revenue', 'n_late', 'n_invoices', 'seasonal_factor', 'avg_adj_rate']
    monthly_revenue['actual_late_pct'] = monthly_revenue['n_late'] / monthly_revenue['n_invoices'] * 100
    print(monthly_revenue.to_string())
    
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
        'total_revenue': revenue_total,
        'avg_seasonal_factor': df['seasonal_adjustment_factor'].mean(),
        'avg_snapshot_rate': df['decile_snapshot_rate'].mean(),
        'avg_calibrated_base_rate': df['decile_calibrated_rate'].mean(),
        'avg_adjusted_rate': df['adjusted_late_rate'].mean()
    }

# Print summaries
summary_with = print_scenario_summary(with_discount, "FY2025 - WITH EARLY PAYMENT DISCOUNT (CALIBRATED)")
summary_no = print_scenario_summary(no_discount, "FY2025 - NO DISCOUNT (CALIBRATED + RETAINED DISCOUNTS)")

# ================================================================
# Comparison
# ================================================================
print("\n" + "="*70)
print("DIRECT COMPARISON - FY2025 (CALIBRATED BASELINE)")
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

print(f"\nCalibration Impact:")
print(f"  Calibrated November baseline: {calibrated_baseline*100:.2f}%")
print(f"  Scaling factor from snapshot: {scaling_factor:.2f}x")
print(f"  Target revenue: $1,043,000")
print(f"  Actual revenue (no discount): ${summary_no['total_revenue']:,.2f}")
print(f"  Gap: ${summary_no['total_revenue'] - 1_043_000:,.2f}")

# ================================================================
# Save results
# ================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

comparison_df = pd.DataFrame([summary_with, summary_no])
output_csv = os.path.join(OUTPUT_DIR, '10.7_FY2025_calibrated_comparison_summary.csv')
comparison_df.to_csv(output_csv, index=False)
print(f"✓ Saved comparison summary to: {output_csv}")

output_excel = os.path.join(OUTPUT_DIR, '10.7_FY2025_calibrated_detailed_simulations.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    with_discount.to_excel(writer, sheet_name='With_Discount', index=False)
    no_discount.to_excel(writer, sheet_name='No_Discount', index=False)
    comparison_df.to_excel(writer, sheet_name='Summary_Comparison', index=False)
    decile_seasonal_df.to_excel(writer, sheet_name='Decile_Seasonal_Adjustments', index=False)

print(f"✓ Saved detailed simulations to: {output_excel}")

# ================================================================
# cd level analysis
# ================================================================
print("\n" + "="*70)
print("CD LEVEL ANALYSIS (CALIBRATED)")
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
    cd_analysis_file = os.path.join(OUTPUT_DIR, '10.7_cd_level_analysis_calibrated.csv')
    cd_analysis_df.to_csv(cd_analysis_file, index=False)
    print(f"✓ Saved cd level analysis to: {cd_analysis_file}")
    
    print("\nCD Level Revenue Summary:")
    print(cd_analysis_df.to_string())

# ================================================================
# Visualizations
# ================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

TARGET_REVENUE = 1_043_000

# Visualization 1: Cumulative Revenue
print("\nCreating visualization: Cumulative revenue over time...")

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

# Plot WITH DISCOUNT
ax.plot(with_discount_monthly['payment_date'], with_discount_monthly['cumulative'], 
        marker='o', linewidth=3, label='With Discount (Interest Only)', 
        color='#70AD47', markersize=8)

# Plot NO DISCOUNT as stacked area
ax.fill_between(no_discount_monthly['payment_date'], 0, no_discount_monthly['cumulative_interest'],
                 alpha=0.3, color='#4472C4', label='No Discount - Interest')
ax.fill_between(no_discount_monthly['payment_date'], no_discount_monthly['cumulative_interest'], 
                 no_discount_monthly['cumulative_total'],
                 alpha=0.3, color='#8FAADC', label='No Discount - Retained Discounts')

# Plot NO DISCOUNT total line
ax.plot(no_discount_monthly['payment_date'], no_discount_monthly['cumulative_total'], 
        marker='s', linewidth=3, label='No Discount (Total)', 
        color='#4472C4', markersize=8)

# Add TARGET LINE
ax.axhline(y=TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target Revenue (${TARGET_REVENUE:,.0f})', alpha=0.8)

# Find when NO DISCOUNT crosses target
cross_idx = None
for i, val in enumerate(no_discount_monthly['cumulative_total']):
    if val >= TARGET_REVENUE:
        cross_idx = i
        break

if cross_idx is not None:
    cross_date = no_discount_monthly['payment_date'].iloc[cross_idx]
    cross_value = no_discount_monthly['cumulative_total'].iloc[cross_idx]
    
    ax.axvline(x=cross_date, color='red', linestyle=':', linewidth=2, alpha=0.5)
    
    ax.annotate(f'Target reached:\n{cross_date.strftime("%Y-%m")}\n(${cross_value:,.0f})',
               xy=(cross_date, TARGET_REVENUE),
               xytext=(20, 30), textcoords='offset points',
               fontsize=11, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='red', linewidth=2),
               arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

ax.set_title(f'FY2025 Cumulative Revenue (Calibrated Baseline)\nInterest + Retained Discounts vs Target\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
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
viz_path = os.path.join(OUTPUT_DIR, '10.7_cumulative_revenue_calibrated.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_path}")
plt.close()

# Visualization 2: Monthly Late Rates
print("Creating visualization: Monthly late rates comparison...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Get monthly summary
monthly_summary = with_discount.groupby(with_discount['invoice_period'].dt.to_period('M')).agg({
    'seasonal_adjustment_factor': 'first',
    'adjusted_late_rate': 'mean',
    'decile_calibrated_rate': 'mean',
    'is_late': ['sum', 'count']
}).reset_index()
monthly_summary.columns = ['invoice_period', 'seasonal_factor', 'avg_adjusted_rate', 'avg_calibrated_base', 'n_late', 'n_invoices']
monthly_summary['invoice_period'] = monthly_summary['invoice_period'].dt.to_timestamp()
monthly_summary['actual_late_rate'] = monthly_summary['n_late'] / monthly_summary['n_invoices']

# Top panel: Late rates over time
ax1.plot(monthly_summary['invoice_period'], monthly_summary['avg_calibrated_base'] * 100,
         marker='D', linewidth=2.5, markersize=8, label='Calibrated Base Rate',
         color='green', alpha=0.8, linestyle=':')

ax1.plot(monthly_summary['invoice_period'], monthly_summary['avg_adjusted_rate'] * 100,
         marker='o', linewidth=2.5, markersize=8, label='Adjusted Rate (Expected)',
         color='orange', alpha=0.8)

ax1.plot(monthly_summary['invoice_period'], monthly_summary['actual_late_rate'] * 100,
         marker='s', linewidth=2.5, markersize=8, label='Actual Late Rate (Simulated)',
         color='steelblue', alpha=0.8, linestyle='--')

ax1.axhline(y=calibrated_baseline*100, color='red', linestyle=':', linewidth=2,
           label=f'November Calibrated Baseline ({calibrated_baseline*100:.1f}%)', alpha=0.6)

ax1.set_title('Calibrated vs Actual Late Payment Rates\nFY2025', 
             fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Late Payment Rate (%)', fontsize=12)
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.tick_params(axis='x', rotation=45)

# Bottom panel: Seasonal adjustment factors
ax2.plot(monthly_summary['invoice_period'], monthly_summary['seasonal_factor'],
        marker='D', linewidth=2.5, markersize=8, color='purple', alpha=0.8)

ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
           label='No adjustment (1.0)')

ax2.set_title('Seasonal Adjustment Factors Applied\nFY2025', 
             fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Adjustment Factor', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, '10.7_monthly_late_rates_comparison.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_path}")
plt.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE - FY2025 WITH CALIBRATED BASELINE")
print("="*70)
print(f"\nAll results saved to folder: {OUTPUT_DIR}/")
print("Files created:")
print(f"  1. 10.7_FY2025_calibrated_comparison_summary.csv")
print(f"  2. 10.7_FY2025_calibrated_detailed_simulations.xlsx")
print(f"  3. 10.7_cd_level_analysis_calibrated.csv")
print(f"  4. 10.7_cumulative_revenue_calibrated.png")
print(f"  5. 10.7_monthly_late_rates_comparison.png")
print(f"  6. 10.7_decile_seasonal_adjustments.csv")
print("\nSimulation approach:")
print(f"  - November 2025 calibrated baseline: {calibrated_baseline*100:.2f}%")
print(f"  - Snapshot rates scaled by: {scaling_factor:.2f}x")
print(f"  - Each decile has calibrated base rate")
print(f"  - Seasonal adjustment factors applied to calibrated base")
print(f"  - Adjusted Rate = Calibrated Base Rate × Seasonal Factor")
print(f"  - cd level distributions from decile payment profile")
print(f"  - Payment timing determined by cd level")
print(f"  - NO DISCOUNT revenue = Interest + Retained Discounts")
print(f"\nRevenue Target Comparison:")
print(f"  Target: $1,043,000")
print(f"  Simulated (No Discount): ${summary_no['total_revenue']:,.2f}")
print(f"  Difference: ${summary_no['total_revenue'] - 1_043_000:,.2f}")
print("="*70)