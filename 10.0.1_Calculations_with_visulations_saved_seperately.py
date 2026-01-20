'''
Docstring for 10.0.1_Calculations_with_visulations_saved_seperately
same as script 10_, but only looks at interest difference, and applies same selection of late payments to each scenario.
10_ has new selection of late payments for discount and non-discount, so really the other one should be not used.

inputs:
- ats_grouped_transformed_with_discounts.csv 
- invoice_grouped_transformed_with_discounts.csv 
- decile_payment_profile.pkl 

outputs:
- 10.0.1_1_cumulative_interest_revenue.png
- 10.0.1_2_monthly_interest_revenue.png
- 10.0.1_3_monthly_payment_count.png
- 10.0.1_4_revenue_difference.png
- 10.0.1_5_avg_interest_per_payment.png
- 10.0.1_monthly_cumulative_interest.csv
- 10.0.1_FY2025_cd_timing_detailed_simulations.xlsx

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
ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30  # 20 days = 0.67 months

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
data_cleaning_dir = base_dir / "data_cleaning"
visualisations_dir = base_dir / "visualisations"

# New Zealand FY2025 definition (April 1, 2024 - March 31, 2025)
FY2025_START = pd.Timestamp("2024-07-01")
FY2025_END = pd.Timestamp("2025-06-30")

OUTPUT_DIR = visualisations_dir

# ================================================================
# CD LEVEL TO PAYMENT TIMING MAPPING
# ================================================================
CD_TO_DAYS = {
    2: 30, 
    3: 60,   # cd=3: 60 days overdue
    4: 90,   # cd=4: 90 days overdue
    5: 120,  # cd=5: 120 days overdue
    6: 150,  # cd=6: 150 days overdue
    7: 180,  # cd=7: 180 days overdue
    8: 210,  # cd=8: 210 days overdue
    9: 240   # cd=9: 240 days overdue
}

print("\n" + "="*70)
print("CD LEVEL TO PAYMENT TIMING MAPPING")
print("="*70)
for cd, days in sorted(CD_TO_DAYS.items()):
    months = days / 30
    print(f"  cd = {cd}: {days} days ({months:.1f} months) overdue")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# Function to simulate payment behavior (ONCE for all scenarios)
# ================================================================
def simulate_payment_behavior(invoices_df, decile_profile_dict, cd_to_days_map):
    """
    Simulate which invoices are late and their payment timing.
    This is done ONCE and then applied to both discount scenarios.
    """
    simulated = invoices_df.copy()
    n_invoices = len(simulated)
    
    is_late_array = np.zeros(n_invoices, dtype=bool)
    days_overdue_array = np.zeros(n_invoices, dtype=float)
    cd_level_array = np.zeros(n_invoices, dtype=int)
    
    print(f"\nSimulating payment behavior for {n_invoices:,} invoices...")
    
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
                
                # Determine days overdue from cd level
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
    
    print(f"✓ Payment behavior simulated:")
    print(f"  Late invoices: {is_late_array.sum():,} ({is_late_array.sum()/n_invoices*100:.1f}%)")
    print(f"  On-time invoices: {(~is_late_array).sum():,} ({(~is_late_array).sum()/n_invoices*100:.1f}%)")
    
    return simulated

# ================================================================
# Function to apply discount scenario to pre-simulated invoices
# ================================================================
def apply_discount_scenario(simulated_invoices, discount_scenario):
    """
    Apply discount scenario to invoices with pre-determined payment behavior.
    Uses the SAME late/on-time status for both scenarios.
    """
    result = simulated_invoices.copy()
    
    if discount_scenario == 'with_discount':
        result['principal_amount'] = result['total_discounted_price']
        result['paid_on_time'] = ~result['is_late']
        result['discount_applied'] = result['discount_amount']
    else:  # no_discount
        result['principal_amount'] = result['total_undiscounted_price']
        result['paid_on_time'] = False
        result['discount_applied'] = 0
    
    # Calculate interest charges
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
# Run simulations
# ================================================================
print("\n" + "="*70)
print("RUNNING SIMULATIONS FOR FY2025")
print("="*70)

# STEP 1: Simulate payment behavior ONCE (which invoices are late, cd levels, days overdue)
print("\nSTEP 1: Determining payment behavior (same for both scenarios)...")
simulated_base = simulate_payment_behavior(
    fy2025_df, 
    decile_profile, 
    cd_to_days_map=CD_TO_DAYS
)

# STEP 2: Apply discount scenarios to the same set of invoices
print("\nSTEP 2: Applying discount scenarios...")
print("  Scenario 1: With early payment discount...")
with_discount = apply_discount_scenario(simulated_base, 'with_discount')

print("  Scenario 2: No discount offered...")
no_discount = apply_discount_scenario(simulated_base, 'no_discount')

print("\n✓ Simulations complete")
print("\nKey insight: Both scenarios use the SAME invoices with the SAME payment timing.")
print("The only difference is the principal amount used for interest calculation.")

print("\n✓ Simulations complete")
print("\nKey insight: Both scenarios use the SAME invoices with the SAME payment timing.")
print("The only difference is the principal amount used for interest calculation.")

# Verification: Check that both scenarios have identical payment behavior
print("\n" + "="*70)
print("VERIFICATION: Confirming same payment behavior")
print("="*70)
print(f"With Discount - Late invoices: {with_discount['is_late'].sum():,}")
print(f"No Discount - Late invoices: {no_discount['is_late'].sum():,}")
print(f"Same late invoices? {(with_discount['is_late'] == no_discount['is_late']).all()}")
print(f"Same payment dates? {(with_discount['payment_date'] == no_discount['payment_date']).all()}")
print(f"Same cd levels? {(with_discount['cd_level'] == no_discount['cd_level']).all()}")

if (with_discount['is_late'] == no_discount['is_late']).all():
    print("✓ VERIFIED: Both scenarios use identical payment behavior")
else:
    print("⚠ WARNING: Scenarios have different payment behavior!")

# ================================================================
# Summary statistics
# ================================================================
print("\n" + "="*70)
print(f"REVENUE COMPARISON: FY2025")
print("="*70)

def print_scenario_summary(df, scenario_name):
    """Print summary statistics"""
    print(f"\n{scenario_name}")
    print("-" * 70)
    
    total_invoices = len(df)
    n_late = df['is_late'].sum()
    n_on_time = total_invoices - n_late
    
    print(f"Total invoices: {total_invoices:,}")
    print(f"  Paid on time: {n_on_time:,} ({n_on_time/total_invoices*100:.1f}%)")
    print(f"  Paid late: {n_late:,} ({n_late/total_invoices*100:.1f}%)")
    
    if n_late > 0:
        print(f"  Avg months overdue: {df[df['is_late']]['months_overdue'].mean():.2f}")
        print(f"  Avg days overdue: {df[df['is_late']]['days_overdue'].mean():.1f}")
    
    print(f"\nCredit Card Revenue:")
    print(f"  Total interest revenue: ${df['interest_charged'].sum():,.2f}")
    
    return {
        'scenario': scenario_name,
        'total_invoices': total_invoices,
        'n_late': n_late,
        'pct_late': n_late/total_invoices*100,
        'total_revenue': df['credit_card_revenue'].sum()
    }

summary_with = print_scenario_summary(with_discount, "WITH DISCOUNT")
summary_no = print_scenario_summary(no_discount, "NO DISCOUNT")

# ================================================================
# Visualization: Cumulative Interest Revenue
# ================================================================
print("\n" + "="*70)
print("CREATING CUMULATIVE INTEREST REVENUE VISUALIZATIONS")
print("="*70)

def aggregate_by_month(df, scenario_name):
    """Aggregate revenue by payment month"""
    monthly = df.groupby(df['payment_date'].dt.to_period('M')).agg({
        'interest_charged': 'sum',
        'credit_card_revenue': 'sum',
        'is_late': 'sum',
        'principal_amount': 'sum'
    }).reset_index()
    
    monthly['payment_date'] = monthly['payment_date'].dt.to_timestamp()
    monthly['scenario'] = scenario_name
    monthly['n_payments'] = df.groupby(df['payment_date'].dt.to_period('M')).size().values
    monthly['pct_late'] = (monthly['is_late'] / monthly['n_payments']) * 100
    
    return monthly

monthly_with = aggregate_by_month(with_discount, 'With Discount')
monthly_no = aggregate_by_month(no_discount, 'No Discount')

# Calculate cumulative
monthly_with['cumulative_interest'] = monthly_with['interest_charged'].cumsum()
monthly_no['cumulative_interest'] = monthly_no['interest_charged'].cumsum()

# ================================================================
# Plot 1: Cumulative Interest Revenue
# ================================================================
print("\nCreating Plot 1: Cumulative Interest Revenue...")

fig1, ax1 = plt.subplots(figsize=(16, 9))

ax1.plot(monthly_with['payment_date'], monthly_with['cumulative_interest'], 
         marker='o', linewidth=3, label='With Discount', color='#70AD47', markersize=8)
ax1.plot(monthly_no['payment_date'], monthly_no['cumulative_interest'], 
         marker='s', linewidth=3, label='No Discount', color='#4472C4', markersize=8)

ax1.set_title(f'FY2025 Cumulative Interest Revenue (CD-Based Timing)\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
              fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Payment Month', fontsize=13)
ax1.set_ylabel('Cumulative Interest Revenue ($)', fontsize=13)
ax1.legend(loc='upper left', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.tick_params(axis='x', rotation=45)

# Add final values
final_with = monthly_with['cumulative_interest'].iloc[-1]
final_no = monthly_no['cumulative_interest'].iloc[-1]
ax1.annotate(f'Final: ${final_with:,.0f}', 
             xy=(monthly_with['payment_date'].iloc[-1], final_with),
             xytext=(10, -20), textcoords='offset points',
             fontsize=11, fontweight='bold', color='#70AD47',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#70AD47'))
ax1.annotate(f'Final: ${final_no:,.0f}', 
             xy=(monthly_no['payment_date'].iloc[-1], final_no),
             xytext=(10, 10), textcoords='offset points',
             fontsize=11, fontweight='bold', color='#4472C4',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#4472C4'))

plt.tight_layout()
output_1 = os.path.join(OUTPUT_DIR, '10.0.1_1_cumulative_interest_revenue.png')
plt.savefig(output_1, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_1}")
plt.close()

# ================================================================
# Plot 2: Monthly Interest Revenue
# ================================================================
print("Creating Plot 2: Monthly Interest Revenue...")

fig2, ax2 = plt.subplots(figsize=(16, 9))

width = 12
x_with = monthly_with['payment_date'] - pd.Timedelta(days=width/2)
x_no = monthly_no['payment_date'] + pd.Timedelta(days=width/2)

ax2.bar(x_with, monthly_with['interest_charged'], width=width, 
        label='With Discount', color='#70AD47', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.bar(x_no, monthly_no['interest_charged'], width=width, 
        label='No Discount', color='#4472C4', alpha=0.7, edgecolor='black', linewidth=0.5)

ax2.set_title(f'FY2025 Monthly Interest Revenue (Non-Cumulative)\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
              fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Payment Month', fontsize=13)
ax2.set_ylabel('Interest Revenue ($)', fontsize=13)
ax2.legend(loc='upper left', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_2 = os.path.join(OUTPUT_DIR, '10.0.1_2_monthly_interest_revenue.png')
plt.savefig(output_2, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_2}")
plt.close()

# ================================================================
# Plot 3: Monthly Payment Count
# ================================================================
print("Creating Plot 3: Monthly Payment Count...")

fig3, ax3 = plt.subplots(figsize=(16, 9))

ax3.bar(x_with, monthly_with['n_payments'], width=width, 
        label='With Discount', color='#70AD47', alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.bar(x_no, monthly_no['n_payments'], width=width, 
        label='No Discount', color='#4472C4', alpha=0.7, edgecolor='black', linewidth=0.5)

ax3.set_title(f'FY2025 Monthly Payment Count\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
              fontsize=16, fontweight='bold', pad=15)
ax3.set_xlabel('Payment Month', fontsize=13)
ax3.set_ylabel('Number of Payments', fontsize=13)
ax3.legend(loc='upper left', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_3 = os.path.join(OUTPUT_DIR, '10.0.1_3_monthly_payment_count.png')
plt.savefig(output_3, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_3}")
plt.close()

# ================================================================
# Plot 4: Revenue Difference
# ================================================================
print("Creating Plot 4: Revenue Difference...")

fig4, ax4 = plt.subplots(figsize=(16, 9))

merged = pd.merge(monthly_with[['payment_date', 'cumulative_interest']], 
                  monthly_no[['payment_date', 'cumulative_interest']], 
                  on='payment_date', suffixes=('_with', '_no'), how='outer')
merged = merged.fillna(method='ffill')
merged['revenue_difference'] = merged['cumulative_interest_no'] - merged['cumulative_interest_with']

colors = ['#C55A11' if x > 0 else '#70AD47' for x in merged['revenue_difference']]
ax4.bar(merged['payment_date'], merged['revenue_difference'], width=20, 
        color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

ax4.set_title(f'FY2025 Cumulative Revenue Difference (No Discount - With Discount)\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
              fontsize=16, fontweight='bold', pad=15)
ax4.set_xlabel('Payment Month', fontsize=13)
ax4.set_ylabel('Revenue Difference ($)', fontsize=13)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.tick_params(axis='x', rotation=45)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#C55A11', alpha=0.7, label='No Discount earns MORE'),
    Patch(facecolor='#70AD47', alpha=0.7, label='With Discount earns MORE')
]
ax4.legend(handles=legend_elements, loc='upper left', fontsize=12)

plt.tight_layout()
output_4 = os.path.join(OUTPUT_DIR, '10.0.1_4_revenue_difference.png')
plt.savefig(output_4, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_4}")
plt.close()

# ================================================================
# Plot 5: Average Interest per Payment
# ================================================================
print("Creating Plot 5: Average Interest per Payment...")

fig5, ax5 = plt.subplots(figsize=(16, 9))

monthly_with['avg_interest_per_payment'] = monthly_with['interest_charged'] / monthly_with['n_payments']
monthly_no['avg_interest_per_payment'] = monthly_no['interest_charged'] / monthly_no['n_payments']

ax5.plot(monthly_with['payment_date'], monthly_with['avg_interest_per_payment'], 
         marker='o', linewidth=2, label='With Discount', color='#70AD47', markersize=6)
ax5.plot(monthly_no['payment_date'], monthly_no['avg_interest_per_payment'], 
         marker='s', linewidth=2, label='No Discount', color='#4472C4', markersize=6)

ax5.set_title(f'FY2025 Average Interest per Payment\n{FY2025_START.strftime("%d/%m/%Y")} - {FY2025_END.strftime("%d/%m/%Y")}', 
              fontsize=16, fontweight='bold', pad=15)
ax5.set_xlabel('Payment Month', fontsize=13)
ax5.set_ylabel('Avg Interest ($)', fontsize=13)
ax5.legend(loc='upper left', fontsize=12)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax5.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_5 = os.path.join(OUTPUT_DIR, '10.0.1_5_avg_interest_per_payment.png')
plt.savefig(output_5, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_5}")
plt.close()

print("\n✓ All visualizations saved successfully!")

# Save monthly data
monthly_combined = pd.merge(
    monthly_with[['payment_date', 'interest_charged', 'cumulative_interest', 'n_payments']],
    monthly_no[['payment_date', 'interest_charged', 'cumulative_interest', 'n_payments']],
    on='payment_date',
    suffixes=('_with_discount', '_no_discount'),
    how='outer'
).sort_values('payment_date')

monthly_csv = os.path.join(OUTPUT_DIR, '10.0.1_monthly_cumulative_interest.csv')
monthly_combined.to_csv(monthly_csv, index=False)
print(f"✓ Saved monthly cumulative data to: {monthly_csv}")

# Save detailed simulations
output_excel = os.path.join(OUTPUT_DIR, '10.0.1_FY2025_cd_timing_detailed_simulations.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    with_discount.to_excel(writer, sheet_name='With_Discount', index=False)
    no_discount.to_excel(writer, sheet_name='No_Discount', index=False)

print(f"✓ Saved detailed simulations to: {output_excel}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll results saved to folder: {OUTPUT_DIR}/")