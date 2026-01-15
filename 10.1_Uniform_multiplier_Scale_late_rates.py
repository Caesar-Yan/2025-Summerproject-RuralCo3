"""
10.1_Uniform_multiplier_Scale_late_rates.py
================================
OPTION 1: Uniform Multiplier Approach

Calibrate late payment rates by applying a uniform multiplier to all deciles.
This preserves the relative differences between deciles.

Target: $1,043,000 in interest revenue (WITH DISCOUNT scenario)

Approach:
- Multiply all prob_late values by a constant factor
- Use binary search to find the multiplier that hits target
- Cap probabilities at 1.0
- Preserve decile-specific cd distributions

Author: Chris
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import copy

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
print(f"Working directory set to: {os.getcwd()}")

# ================================================================
# CONFIGURATION
# ================================================================
ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30  # 20 days = 0.67 months

# Target revenue (actual observed) - FOR WITH DISCOUNT SCENARIO
TARGET_REVENUE = 1_043_000  # $1.043M

# New Zealand FY2025 definition
FY2025_START = pd.Timestamp("2024-07-01")
FY2025_END = pd.Timestamp("2025-06-30")

OUTPUT_DIR = "FY2025_outputs_calibrated_MULTIPLIER"

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

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

print("\n" + "="*70)
print("CALIBRATION METHOD: UNIFORM MULTIPLIER")
print("="*70)
print(f"Target Revenue (WITH DISCOUNT): ${TARGET_REVENUE:,.2f}")
print("\nApproach: Multiply all decile late rates by a constant factor")
print("This preserves relative differences between deciles")

# ================================================================
# Load invoice data
# ================================================================
print("\n" + "="*70)
print("LOADING INVOICE DATA")
print("="*70)

ats_grouped = pd.read_csv('ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv('invoice_grouped_transformed_with_discounts.csv')

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

print(f"\nFY2025 invoices: {len(fy2025_df):,}")

if len(fy2025_df) == 0:
    print("\n⚠ WARNING: No invoices found in FY2025!")
    exit()

# ================================================================
# Load decile payment profile
# ================================================================
print("\n" + "="*70)
print("LOADING ORIGINAL DECILE PAYMENT PROFILE")
print("="*70)

try:
    try:
        with open('Payment Profile/decile_payment_profile_MODIFIED.pkl', 'rb') as f:
            original_profile = pickle.load(f)
    except FileNotFoundError:
        with open('Payment Profile/decile_payment_profile.pkl', 'rb') as f:
            original_profile = pickle.load(f)
    
    n_deciles = original_profile['metadata']['n_deciles']
    print(f"✓ Loaded original decile payment profile")
    print(f"  Number of deciles: {n_deciles}")
    
    # Show original late rates
    print(f"\nOriginal late payment rates by decile:")
    for i in range(n_deciles):
        decile_key = f'decile_{i}'
        prob_late = original_profile['deciles'][decile_key]['payment_behavior']['prob_late']
        print(f"  Decile {i}: {prob_late*100:.1f}%")
    
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

def create_scaled_profile(original_profile_dict, multiplier):
    """Create a new profile with scaled late payment probabilities"""
    scaled_profile = copy.deepcopy(original_profile_dict)
    
    for decile_key, decile_data in scaled_profile['deciles'].items():
        original_prob_late = original_profile_dict['deciles'][decile_key]['payment_behavior']['prob_late']
        scaled_prob_late = min(1.0, original_prob_late * multiplier)  # Cap at 1.0
        scaled_profile['deciles'][decile_key]['payment_behavior']['prob_late'] = scaled_prob_late
    
    scaled_profile['metadata']['calibration_method'] = 'uniform_multiplier'
    scaled_profile['metadata']['multiplier'] = multiplier
    scaled_profile['metadata']['target_revenue'] = TARGET_REVENUE
    
    return scaled_profile

def simulate_with_multiplier(invoices_df, original_profile_dict, multiplier, cd_to_days_map):
    """Run full simulation with a given multiplier"""
    # Create scaled profile
    scaled_profile = create_scaled_profile(original_profile_dict, multiplier)
    
    # Simulate payment behavior
    simulated_base = simulate_payment_behavior(invoices_df, scaled_profile, cd_to_days_map)
    
    # Apply both scenarios
    with_discount = apply_discount_scenario(simulated_base, 'with_discount')
    no_discount = apply_discount_scenario(simulated_base, 'no_discount')
    
    revenue_with = with_discount['credit_card_revenue'].sum()
    revenue_no = no_discount['credit_card_revenue'].sum()
    late_count = simulated_base['is_late'].sum()
    late_rate = late_count / len(simulated_base) * 100
    
    return revenue_with, revenue_no, late_rate, with_discount, no_discount, scaled_profile

# ================================================================
# Binary search for optimal multiplier
# ================================================================
print("\n" + "="*70)
print("BINARY SEARCH FOR OPTIMAL MULTIPLIER")
print("="*70)

# First, test the original profile to see current revenue
print("\nTesting original profile (multiplier = 1.0)...")
revenue_with_orig, revenue_no_orig, late_rate_orig, _, _, _ = simulate_with_multiplier(
    fy2025_df, original_profile, 1.0, CD_TO_DAYS
)
print(f"  Revenue (with discount): ${revenue_with_orig:,.2f}")
print(f"  Revenue (no discount): ${revenue_no_orig:,.2f}")
print(f"  Overall late rate: {late_rate_orig:.1f}%")
print(f"  Gap to target: ${TARGET_REVENUE - revenue_with_orig:,.2f}")

# Binary search
print("\nStarting binary search...")
tolerance = 1000  # $1,000 tolerance
max_iterations = 30

low = 1.0
high = 20.0  # Start with reasonable upper bound
best_multiplier = 1.0
best_revenue_with = revenue_with_orig
best_revenue_no = revenue_no_orig

iteration_results = []

for iteration in range(max_iterations):
    mid = (low + high) / 2
    
    # Reset random seed for consistent results
    np.random.seed(RANDOM_SEED)
    
    revenue_with, revenue_no, late_rate, _, _, _ = simulate_with_multiplier(
        fy2025_df, original_profile, mid, CD_TO_DAYS
    )
    
    iteration_results.append({
        'iteration': iteration + 1,
        'multiplier': mid,
        'revenue_with_discount': revenue_with,
        'revenue_no_discount': revenue_no,
        'late_rate': late_rate,
        'gap': revenue_with - TARGET_REVENUE
    })
    
    print(f"  Iteration {iteration+1}: multiplier={mid:.4f}, revenue=${revenue_with:,.0f}, gap=${revenue_with - TARGET_REVENUE:+,.0f}")
    
    if abs(revenue_with - TARGET_REVENUE) < tolerance:
        best_multiplier = mid
        best_revenue_with = revenue_with
        best_revenue_no = revenue_no
        print(f"\n✓ CONVERGED! Found optimal multiplier: {best_multiplier:.4f}")
        break
    
    if revenue_with < TARGET_REVENUE:
        low = mid
    else:
        high = mid
    
    best_multiplier = mid
    best_revenue_with = revenue_with
    best_revenue_no = revenue_no

print("\n" + "="*70)
print("CALIBRATION COMPLETE")
print("="*70)
print(f"\nOptimal Multiplier: {best_multiplier:.4f}")
print(f"  Original avg late rate: {late_rate_orig:.1f}%")

# ================================================================
# Generate final results with optimal multiplier
# ================================================================
print("\n" + "="*70)
print("GENERATING FINAL RESULTS")
print("="*70)

np.random.seed(RANDOM_SEED)
revenue_with, revenue_no, late_rate, with_discount_final, no_discount_final, calibrated_profile = simulate_with_multiplier(
    fy2025_df, original_profile, best_multiplier, CD_TO_DAYS
)

print(f"\nCalibrated Results:")
print(f"  WITH DISCOUNT:")
print(f"    Revenue: ${revenue_with:,.2f}")
print(f"    Gap to target: ${revenue_with - TARGET_REVENUE:+,.2f}")
print(f"  NO DISCOUNT:")
print(f"    Revenue: ${revenue_no:,.2f}")
print(f"  Overall late rate: {late_rate:.1f}%")

# Show calibrated late rates by decile
print(f"\nCalibrated late payment rates by decile:")
for i in range(n_deciles):
    decile_key = f'decile_{i}'
    original_prob = original_profile['deciles'][decile_key]['payment_behavior']['prob_late']
    calibrated_prob = calibrated_profile['deciles'][decile_key]['payment_behavior']['prob_late']
    print(f"  Decile {i}: {original_prob*100:.1f}% → {calibrated_prob*100:.1f}% (multiplier: {calibrated_prob/original_prob:.2f}x)")

# ================================================================
# Save results
# ================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save calibrated profile
calibrated_profile_path = os.path.join(OUTPUT_DIR, 'decile_payment_profile_CALIBRATED_MULTIPLIER.pkl')
with open(calibrated_profile_path, 'wb') as f:
    pickle.dump(calibrated_profile, f)
print(f"✓ Saved calibrated profile: {calibrated_profile_path}")

# Save iteration results
iteration_df = pd.DataFrame(iteration_results)
iteration_csv = os.path.join(OUTPUT_DIR, 'calibration_iterations_MULTIPLIER.csv')
iteration_df.to_csv(iteration_csv, index=False)
print(f"✓ Saved iteration history: {iteration_csv}")

# Save detailed simulations
output_excel = os.path.join(OUTPUT_DIR, 'FY2025_calibrated_MULTIPLIER_detailed.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    with_discount_final.to_excel(writer, sheet_name='With_Discount', index=False)
    no_discount_final.to_excel(writer, sheet_name='No_Discount', index=False)
print(f"✓ Saved detailed simulations: {output_excel}")

# Save summary
summary_data = {
    'Method': 'Uniform Multiplier',
    'Optimal_Multiplier': best_multiplier,
    'Target_Revenue': TARGET_REVENUE,
    'Revenue_With_Discount': revenue_with,
    'Revenue_No_Discount': revenue_no,
    'Gap_To_Target': revenue_with - TARGET_REVENUE,
    'Overall_Late_Rate_Pct': late_rate,
    'Original_Late_Rate_Pct': late_rate_orig
}
summary_df = pd.DataFrame([summary_data])
summary_csv = os.path.join(OUTPUT_DIR, 'calibration_summary_MULTIPLIER.csv')
summary_df.to_csv(summary_csv, index=False)
print(f"✓ Saved summary: {summary_csv}")

# ================================================================
# Visualization
# ================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Convergence
ax1 = axes[0, 0]
ax1.plot(iteration_df['iteration'], iteration_df['revenue_with_discount'], 
         marker='o', linewidth=2, label='Simulated Revenue')
ax1.axhline(y=TARGET_REVENUE, color='red', linestyle='--', linewidth=2, label='Target Revenue')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Revenue ($)', fontsize=12)
ax1.set_title('Calibration Convergence', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: Multiplier progression
ax2 = axes[0, 1]
ax2.plot(iteration_df['iteration'], iteration_df['multiplier'], 
         marker='s', linewidth=2, color='green')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Multiplier', fontsize=12)
ax2.set_title('Multiplier Progression', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Late rate comparison
ax3 = axes[1, 0]
decile_nums = list(range(n_deciles))
original_rates = [original_profile['deciles'][f'decile_{i}']['payment_behavior']['prob_late']*100 
                  for i in range(n_deciles)]
calibrated_rates = [calibrated_profile['deciles'][f'decile_{i}']['payment_behavior']['prob_late']*100 
                    for i in range(n_deciles)]

x = np.arange(len(decile_nums))
width = 0.35

ax3.bar(x - width/2, original_rates, width, label='Original', alpha=0.7, color='#4472C4')
ax3.bar(x + width/2, calibrated_rates, width, label='Calibrated', alpha=0.7, color='#70AD47')
ax3.set_xlabel('Decile', fontsize=12)
ax3.set_ylabel('Late Payment Rate (%)', fontsize=12)
ax3.set_title('Late Payment Rates by Decile', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([f'D{i}' for i in decile_nums])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Revenue comparison
ax4 = axes[1, 1]
scenarios = ['With Discount', 'No Discount']
revenues = [revenue_with, revenue_no]
colors = ['#70AD47', '#4472C4']

bars = ax4.bar(scenarios, revenues, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axhline(y=TARGET_REVENUE, color='red', linestyle='--', linewidth=2, label='Target')
ax4.set_ylabel('Revenue ($)', fontsize=12)
ax4.set_title('Calibrated Revenue by Scenario', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:,.0f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, 'calibration_results_MULTIPLIER.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {viz_path}")
plt.close()

print("\n" + "="*70)
print("CALIBRATION COMPLETE - UNIFORM MULTIPLIER METHOD")
print("="*70)
print(f"\nFiles saved to: {OUTPUT_DIR}/")
print(f"  1. Calibrated profile (pickle)")
print(f"  2. Iteration history (CSV)")
print(f"  3. Detailed simulations (Excel)")
print(f"  4. Summary (CSV)")
print(f"  5. Visualization (PNG)")
print("="*70)