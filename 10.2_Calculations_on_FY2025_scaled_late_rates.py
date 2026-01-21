'''
Docstring for 10.2_Calculations_on_FY2025_scaled_late_rates

this script runs the estimation for revenue with the scaled payment profiles(uniform rate, and uniform scaling multiplier).

inputs:
- ats_grouped_transformed_with_discounts.csv
- invoice_grouped_transformed_with_discounts.csv

outputs:
- 10.2_calibrated_methods_comparison.csv
- 10.2_FY2025_[METHOD]_detailed.xlsx
- 10.2_1_revenue_comparison.png
- 10.2_2_late_rates_by_decile.png
- 10.2_3_cumulative_revenue_over_time.png
- 10.2_4_revenue_by_decile.png

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import pickle
import os
from pathlib import Path

# ================================================================
# CONFIGURATION
# ================================================================
ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30  # 20 days = 0.67 months

# New Zealand FY2025 definition
FY2025_START = pd.Timestamp("2024-07-01")
FY2025_END = pd.Timestamp("2025-06-30")

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
data_cleaning_dir = base_dir / "data_cleaning"
visualisations_dir = base_dir / "visualisations"

OUTPUT_DIR = visualisations_dir

# CD level to payment timing mapping
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
print("APPLYING CALIBRATED PROFILES TO FY2025 DATA")
print("="*70)

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

# Filter out negative prices
combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()
print(f"After filtering negatives: {len(combined_df):,}")

# ================================================================
# Parse and filter dates
# ================================================================
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

combined_df['invoice_period'] = parse_invoice_period(combined_df['invoice_period'])
combined_df = combined_df[combined_df['invoice_period'].notna()].copy()

# Filter to FY2025
fy2025_df = combined_df[
    (combined_df['invoice_period'] >= FY2025_START) & 
    (combined_df['invoice_period'] <= FY2025_END)
].copy()

print(f"\nFY2025 ({FY2025_START.strftime('%d/%m/%Y')} - {FY2025_END.strftime('%d/%m/%Y')}): {len(fy2025_df):,} invoices")

if len(fy2025_df) == 0:
    print("\n⚠ WARNING: No invoices found in FY2025!")
    exit()

# ================================================================
# Load calibrated profiles
# ================================================================
print("\n" + "="*70)
print("LOADING CALIBRATED PROFILES")
print("="*70)

profiles = {}

# Try to load MULTIPLIER profile
try:
    multiplier_path = profile_dir / 'decile_payment_profile_CALIBRATED_MULTIPLIER.pkl'
    with open(multiplier_path, 'rb') as f:
        profiles['MULTIPLIER'] = pickle.load(f)
    print(f"✓ Loaded MULTIPLIER profile")
    print(f"  Multiplier: {profiles['MULTIPLIER']['metadata']['multiplier']:.4f}x")
except FileNotFoundError:
    print(f"⚠ MULTIPLIER profile not found at: {multiplier_path}")
    print(f"  Run FY2025_calibrate_MULTIPLIER.py first")

# Try to load UNIFORM profile
try:
    uniform_path = profile_dir / 'decile_payment_profile_CALIBRATED_UNIFORM.pkl'
    with open(uniform_path, 'rb') as f:
        profiles['UNIFORM'] = pickle.load(f)
    print(f"✓ Loaded UNIFORM profile")
    print(f"  Uniform late rate: {profiles['UNIFORM']['metadata']['uniform_late_rate']*100:.2f}%")
except FileNotFoundError:
    print(f"⚠ UNIFORM profile not found at: {uniform_path}")
    print(f"  Run FY2025_calibrate_UNIFORM_RATE.py first")

if not profiles:
    print("\n✗ ERROR: No calibrated profiles found!")
    print("Please run at least one of the calibration scripts first.")
    exit()

# ================================================================
# Map invoices to deciles
# ================================================================
print("\n" + "="*70)
print("MAPPING INVOICES TO DECILES")
print("="*70)

# Get n_deciles from first available profile
n_deciles = list(profiles.values())[0]['metadata']['n_deciles']

fy2025_df = fy2025_df.sort_values('total_undiscounted_price').reset_index(drop=True)

fy2025_df['decile'] = pd.qcut(
    fy2025_df['total_undiscounted_price'], 
    q=n_deciles, 
    labels=False,
    duplicates='drop'
)

print(f"✓ Mapped {len(fy2025_df):,} invoices to {n_deciles} deciles")

# ================================================================
# Simulation functions
# ================================================================
def simulate_payment_behavior(invoices_df, decile_profile_dict, cd_to_days_map):
    """Simulate which invoices are late and their payment timing"""
    simulated = invoices_df.copy()
    n_invoices = len(simulated)
    
    is_late_array = np.zeros(n_invoices, dtype=bool)
    days_overdue_array = np.zeros(n_invoices, dtype=float)
    cd_level_array = np.zeros(n_invoices, dtype=int)
    
    for idx, row in simulated.iterrows():
        decile_num = row['decile']
        decile_key = f'decile_{int(decile_num)}'
        
        if decile_key not in decile_profile_dict['deciles']:
            decile_key = 'decile_0'
        
        decile_data = decile_profile_dict['deciles'][decile_key]
        
        # Determine if late
        prob_late = decile_data['payment_behavior']['prob_late']
        is_late = np.random.random() < prob_late
        is_late_array[idx] = is_late
        
        if is_late:
            # Sample cd level
            cd_given_late = decile_data['delinquency_distribution']['cd_given_late']
            
            if cd_given_late:
                cd_levels = list(cd_given_late.keys())
                cd_probs = list(cd_given_late.values())
                cd_probs = np.array(cd_probs) / np.array(cd_probs).sum()
                
                cd_level = np.random.choice(cd_levels, p=cd_probs)
                cd_level_array[idx] = cd_level
                
                if cd_level in cd_to_days_map:
                    days_overdue = cd_to_days_map[cd_level]
                else:
                    days_overdue = 90
                
                days_overdue_array[idx] = days_overdue
            else:
                cd_level_array[idx] = 0
                days_overdue_array[idx] = 60
        else:
            days_overdue_array[idx] = 0
            cd_level_array[idx] = 0
    
    simulated['is_late'] = is_late_array
    simulated['days_overdue'] = days_overdue_array
    simulated['months_overdue'] = days_overdue_array / 30
    simulated['cd_level'] = cd_level_array
    
    simulated['due_date'] = simulated['invoice_period']
    simulated['payment_date'] = simulated['due_date'] + pd.to_timedelta(simulated['days_overdue'], unit='D')
    
    return simulated

def apply_discount_scenario(simulated_invoices, discount_scenario):
    """Apply discount scenario to invoices with pre-determined payment behavior"""
    result = simulated_invoices.copy()
    
    if discount_scenario == 'with_discount':
        result['principal_amount'] = result['total_discounted_price']
        result['paid_on_time'] = ~result['is_late']
        result['discount_applied'] = result['discount_amount']
    else:
        result['principal_amount'] = result['total_undiscounted_price']
        result['paid_on_time'] = False
        result['discount_applied'] = 0
    
    daily_rate = ANNUAL_INTEREST_RATE / 365
    result['interest_charged'] = (
        result['principal_amount'] * 
        daily_rate * 
        result['days_overdue']
    )
    
    result['credit_card_revenue'] = result['interest_charged']
    result['total_invoice_amount_discounted'] = result['total_discounted_price']
    result['total_invoice_amount_undiscounted'] = result['total_undiscounted_price']
    
    return result

# ================================================================
# Run simulations for each calibrated profile
# ================================================================
print("\n" + "="*70)
print("RUNNING SIMULATIONS WITH CALIBRATED PROFILES")
print("="*70)

results = {}

for profile_name, profile_dict in profiles.items():
    print(f"\n{profile_name} Profile:")
    print("-" * 70)
    
    # Reset random seed for consistency
    np.random.seed(RANDOM_SEED)
    
    # Simulate payment behavior ONCE
    simulated_base = simulate_payment_behavior(fy2025_df, profile_dict, CD_TO_DAYS)
    
    # Apply both discount scenarios
    with_discount = apply_discount_scenario(simulated_base, 'with_discount')
    no_discount = apply_discount_scenario(simulated_base, 'no_discount')
    
    # Calculate metrics
    n_late = simulated_base['is_late'].sum()
    late_rate = n_late / len(simulated_base) * 100
    
    revenue_with = with_discount['credit_card_revenue'].sum()
    revenue_no = no_discount['credit_card_revenue'].sum()
    
    print(f"  Late invoices: {n_late:,} ({late_rate:.1f}%)")
    print(f"  Revenue (WITH DISCOUNT): ${revenue_with:,.2f}")
    print(f"  Revenue (NO DISCOUNT): ${revenue_no:,.2f}")
    print(f"  Revenue difference: ${revenue_no - revenue_with:,.2f}")
    
    # Store results
    results[profile_name] = {
        'profile': profile_dict,
        'simulated_base': simulated_base,
        'with_discount': with_discount,
        'no_discount': no_discount,
        'late_rate': late_rate,
        'revenue_with': revenue_with,
        'revenue_no': revenue_no
    }

# ================================================================
# Summary comparison
# ================================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

comparison_data = []
for profile_name, result_dict in results.items():
    comparison_data.append({
        'Method': profile_name,
        'Late_Rate_Pct': result_dict['late_rate'],
        'Revenue_With_Discount': result_dict['revenue_with'],
        'Revenue_No_Discount': result_dict['revenue_no'],
        'Revenue_Difference': result_dict['revenue_no'] - result_dict['revenue_with']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

# ================================================================
# Save results
# ================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save comparison summary
comparison_csv = os.path.join(OUTPUT_DIR, '10.2_calibrated_methods_comparison.csv')
comparison_df.to_csv(comparison_csv, index=False)
print(f"✓ Saved comparison: {comparison_csv}")

# Save detailed results for each method
for profile_name, result_dict in results.items():
    excel_path = os.path.join(OUTPUT_DIR, f'10.2_FY2025_{profile_name}_detailed.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        result_dict['with_discount'].to_excel(writer, sheet_name='With_Discount', index=False)
        result_dict['no_discount'].to_excel(writer, sheet_name='No_Discount', index=False)
    print(f"✓ Saved {profile_name} details: {excel_path}")

# ================================================================
# Create comprehensive visualizations
# ================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# ================================================================
# Visualization 1: Revenue Comparison Bar Chart
# ================================================================
print("\nCreating visualization 1: Revenue comparison...")

fig1, ax1 = plt.subplots(figsize=(14, 8))

methods = list(results.keys())
x = np.arange(len(methods))
width = 0.35

revenues_with = [results[m]['revenue_with'] for m in methods]
revenues_no = [results[m]['revenue_no'] for m in methods]

bars1 = ax1.bar(x - width/2, revenues_with, width, label='With Discount', 
                color='#70AD47', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, revenues_no, width, label='No Discount', 
                color='#4472C4', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Calibration Method', fontsize=14, fontweight='bold')
ax1.set_ylabel('Interest Revenue ($)', fontsize=14, fontweight='bold')
ax1.set_title(f'FY2025 Revenue Estimates by Calibration Method\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=12)
ax1.legend(fontsize=12, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
viz1_path = os.path.join(OUTPUT_DIR, '10.2_1_revenue_comparison.png')
plt.savefig(viz1_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz1_path}")
plt.close()

# ================================================================
# Visualization 2: Late Payment Rates by Decile
# ================================================================
print("Creating visualization 2: Late rates by decile...")

fig2, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6))
if len(results) == 1:
    axes = [axes]

for idx, (profile_name, result_dict) in enumerate(results.items()):
    ax = axes[idx]
    
    profile = result_dict['profile']
    decile_nums = list(range(n_deciles))
    late_rates = [profile['deciles'][f'decile_{i}']['payment_behavior']['prob_late']*100 
                  for i in range(n_deciles)]
    
    bars = ax.bar(decile_nums, late_rates, color='#4472C4', alpha=0.7, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Late Payment Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{profile_name} Method\nLate Rates by Decile', fontsize=14, fontweight='bold')
    ax.set_xticks(decile_nums)
    ax.set_xticklabels([f'D{i}' for i in decile_nums])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
viz2_path = os.path.join(OUTPUT_DIR, '10.2_2_late_rates_by_decile.png')
plt.savefig(viz2_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz2_path}")
plt.close()

# ================================================================
# Visualization 3: Cumulative Interest Revenue Over Time
# ================================================================
print("Creating visualization 3: Cumulative revenue over time...")

fig3, axes = plt.subplots(len(results), 1, figsize=(16, 6 * len(results)))
if len(results) == 1:
    axes = [axes]

for idx, (profile_name, result_dict) in enumerate(results.items()):
    ax = axes[idx]
    
    # Aggregate by payment month
    with_discount_df = result_dict['with_discount']
    no_discount_df = result_dict['no_discount']
    
    monthly_with = with_discount_df.groupby(with_discount_df['payment_date'].dt.to_period('M')).agg({
        'interest_charged': 'sum'
    }).reset_index()
    monthly_with['payment_date'] = monthly_with['payment_date'].dt.to_timestamp()
    monthly_with['cumulative'] = monthly_with['interest_charged'].cumsum()
    
    monthly_no = no_discount_df.groupby(no_discount_df['payment_date'].dt.to_period('M')).agg({
        'interest_charged': 'sum'
    }).reset_index()
    monthly_no['payment_date'] = monthly_no['payment_date'].dt.to_timestamp()
    monthly_no['cumulative'] = monthly_no['interest_charged'].cumsum()
    
    # Plot
    ax.plot(monthly_with['payment_date'], monthly_with['cumulative'], 
            marker='o', linewidth=3, label='With Discount', color='#70AD47', markersize=8)
    ax.plot(monthly_no['payment_date'], monthly_no['cumulative'], 
            marker='s', linewidth=3, label='No Discount', color='#4472C4', markersize=8)
    
    ax.set_title(f'{profile_name} Method - Cumulative Interest Revenue', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Payment Month', fontsize=12)
    ax.set_ylabel('Cumulative Revenue ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # Add final values
    final_with = monthly_with['cumulative'].iloc[-1]
    final_no = monthly_no['cumulative'].iloc[-1]
    
    ax.annotate(f'${final_with:,.0f}', 
                xy=(monthly_with['payment_date'].iloc[-1], final_with),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#70AD47',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#70AD47'))
    ax.annotate(f'${final_no:,.0f}', 
                xy=(monthly_no['payment_date'].iloc[-1], final_no),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#4472C4',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#4472C4'))

plt.tight_layout()
viz3_path = os.path.join(OUTPUT_DIR, '10.2_3_cumulative_revenue_over_time.png')
plt.savefig(viz3_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz3_path}")
plt.close()

# ================================================================
# Visualization 4: Revenue by Decile
# ================================================================
print("Creating visualization 4: Revenue by decile...")

fig4, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6))
if len(results) == 1:
    axes = [axes]

for idx, (profile_name, result_dict) in enumerate(results.items()):
    ax = axes[idx]
    
    with_discount_df = result_dict['with_discount']
    no_discount_df = result_dict['no_discount']
    
    decile_revenue_with = with_discount_df.groupby('decile')['credit_card_revenue'].sum()
    decile_revenue_no = no_discount_df.groupby('decile')['credit_card_revenue'].sum()
    
    x_pos = np.arange(len(decile_revenue_with))
    width = 0.35
    
    ax.bar(x_pos - width/2, decile_revenue_with, width, 
           label='With Discount', color='#70AD47', alpha=0.7, edgecolor='black')
    ax.bar(x_pos + width/2, decile_revenue_no, width, 
           label='No Discount', color='#4472C4', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Decile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Revenue ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'{profile_name} Method\nRevenue by Decile', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'D{i}' for i in decile_revenue_with.index])
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
viz4_path = os.path.join(OUTPUT_DIR, '10.2_4_revenue_by_decile.png')
plt.savefig(viz4_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz4_path}")
plt.close()

# ================================================================
# Final summary
# ================================================================
print("\n" + "="*70)
print("FINAL REVENUE ESTIMATES - FY2025")
print("="*70)

for profile_name, result_dict in results.items():
    print(f"\n{profile_name} Method:")
    print(f"  WITH DISCOUNT:  ${result_dict['revenue_with']:>15,.2f}")
    print(f"  NO DISCOUNT:    ${result_dict['revenue_no']:>15,.2f}")
    print(f"  Difference:     ${result_dict['revenue_no'] - result_dict['revenue_with']:>15,.2f}")
    print(f"  Late Rate:      {result_dict['late_rate']:>15.1f}%")

print("\n" + "="*70)
print("ALL ANALYSIS COMPLETE")
print("="*70)
print(f"\nFiles saved to: {OUTPUT_DIR}/")
print("  1. calibrated_methods_comparison.csv - Summary comparison")
print("  2. FY2025_[METHOD]_detailed.xlsx - Detailed simulations (each method)")
print("  3. 1_revenue_comparison.png - Revenue bar chart")
print("  4. 2_late_rates_by_decile.png - Late rates by decile")
print("  5. 3_cumulative_revenue_over_time.png - Cumulative revenue trends")
print("  6. 4_revenue_by_decile.png - Revenue breakdown by decile")
print("="*70)