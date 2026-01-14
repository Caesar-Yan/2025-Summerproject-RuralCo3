"""
FY2025_simulation_with_decile_profiles.py
==========================================
Simulate FY2025 invoice payments using decile payment profiles.

Key approach:
1. Sort invoices by total_undiscounted_price
2. Map each invoice to appropriate decile
3. Apply decile-specific P(late) and payment timing
4. Calculate interest for both discount scenarios

Author: Chris
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle
import os

# ================================================================
# CONFIGURATION
# ================================================================
ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
RANDOM_SEED = 42
PAYMENT_TERMS_MONTHS = 20 / 30  # 20 days = 0.67 months

# New Zealand FY2025 definition (April 1, 2024 - March 31, 2025)
FY2025_START = pd.Timestamp("2024-04-01")
FY2025_END = pd.Timestamp("2025-03-31")

OUTPUT_DIR = "FY2025_outputs_decile"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# ================================================================
# Load invoice data
# ================================================================
print("="*70)
print("LOADING INVOICE DATA")
print("="*70)

# Load combined invoice data
ats_grouped = pd.read_csv('ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv('invoice_grouped_transformed_with_discounts.csv')

# Combine datasets
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"Total invoices loaded: {len(combined_df):,}")

# # ================================================================
# # Filter out negative undiscounted prices (OPTIONAL - comment out if not needed)
# # ================================================================
# initial_count = len(combined_df)
# combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()
# filtered_count = initial_count - len(combined_df)
# if filtered_count > 0:
#     print(f"⚠ Filtered out {filtered_count:,} invoices with negative undiscounted prices")
# print(f"Remaining invoices: {len(combined_df):,}")
# # ================================================================

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
    with open('Payment Profile/decile_payment_profile.pkl', 'rb') as f:
        decile_profile = pickle.load(f)
    
    n_deciles = decile_profile['metadata']['n_deciles']
    print(f"✓ Loaded decile payment profile")
    print(f"  Number of deciles: {n_deciles}")
    print(f"  Payment terms: {decile_profile['metadata']['payment_terms_months']:.2f} months")
    print(f"  Method: {decile_profile['metadata']['method']}")
    
except FileNotFoundError:
    print("✗ ERROR: Decile payment profile not found!")
    print("  Expected location: Payment Profile/decile_payment_profile.pkl")
    print("  Please run 09.5_Decile_payment_profile.py first")
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
# Simulation function using decile profiles
# ================================================================
def simulate_with_decile_profiles(invoices_df, decile_profile_dict, discount_scenario):
    """
    Simulate invoice payments using decile-specific payment behavior
    
    Parameters:
    - invoices_df: DataFrame with invoice data (must have 'decile' column)
    - decile_profile_dict: Decile payment profile dictionary
    - discount_scenario: 'with_discount' or 'no_discount'
    
    Returns:
    - DataFrame with simulated payment details
    """
    simulated = invoices_df.copy()
    n_invoices = len(simulated)
    
    # Initialize arrays for simulation results
    is_late_array = np.zeros(n_invoices, dtype=bool)
    months_overdue_array = np.zeros(n_invoices, dtype=float)
    cd_level_array = np.zeros(n_invoices, dtype=int)
    
    # Simulate payment behavior for each invoice based on its decile
    for idx, row in simulated.iterrows():
        decile_num = row['decile']
        decile_key = f'decile_{int(decile_num)}'
        
        if decile_key not in decile_profile_dict['deciles']:
            print(f"⚠ Warning: Decile {decile_num} not found in profile, using decile 0")
            decile_key = 'decile_0'
        
        decile_data = decile_profile_dict['deciles'][decile_key]
        
        # 1. Determine if payment is late based on P(late)
        prob_late = decile_data['payment_behavior']['prob_late']
        is_late = np.random.random() < prob_late
        is_late_array[idx] = is_late
        
        if is_late:
            # 2. If late, sample months overdue from distribution
            avg_months_overdue = decile_data['payment_behavior']['avg_months_overdue_given_late']
            
            # Use exponential distribution centered on average
            # (payment times are typically right-skewed)
            if avg_months_overdue > 0:
                months_overdue = np.random.exponential(avg_months_overdue)
            else:
                months_overdue = 0
            
            months_overdue_array[idx] = months_overdue
            
            # 3. Sample cd level from P(cd | late) distribution
            cd_given_late = decile_data['delinquency_distribution']['cd_given_late']
            
            if cd_given_late:
                cd_levels = list(cd_given_late.keys())
                cd_probs = list(cd_given_late.values())
                
                # Normalize probabilities (in case they don't sum to 1)
                cd_probs = np.array(cd_probs)
                cd_probs = cd_probs / cd_probs.sum()
                
                cd_level = np.random.choice(cd_levels, p=cd_probs)
                cd_level_array[idx] = cd_level
            else:
                cd_level_array[idx] = 0
        else:
            # On-time payment
            months_overdue_array[idx] = 0
            cd_level_array[idx] = 0
    
    # Add simulation results to dataframe
    simulated['is_late'] = is_late_array
    simulated['months_overdue'] = months_overdue_array
    simulated['days_overdue'] = months_overdue_array * 30  # Convert to days
    simulated['cd_level'] = cd_level_array
    
    # Calculate dates
    simulated['due_date'] = simulated['invoice_period']
    simulated['payment_date'] = simulated['due_date'] + pd.to_timedelta(simulated['days_overdue'], unit='D')
    
    # ================================================================
    # Calculate amounts based on discount scenario
    # ================================================================
    if discount_scenario == 'with_discount':
        # With discount: everyone gets the discount initially
        simulated['principal_amount'] = simulated['total_discounted_price']
        simulated['paid_on_time'] = ~simulated['is_late']
        simulated['discount_applied'] = simulated['discount_amount']
        
    else:  # no_discount
        # No discount: everyone pays full undiscounted amount
        simulated['principal_amount'] = simulated['total_undiscounted_price']
        simulated['paid_on_time'] = False
        simulated['discount_applied'] = 0
    
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
    # Calculate total revenue (interest only)
    # ================================================================
    simulated['credit_card_revenue'] = simulated['interest_charged']
    
    # Track total amounts
    simulated['total_invoice_amount_discounted'] = simulated['total_discounted_price']
    simulated['total_invoice_amount_undiscounted'] = simulated['total_undiscounted_price']
    
    return simulated

# ================================================================
# Run both scenarios for FY2025
# ================================================================
print("\n" + "="*70)
print("RUNNING SIMULATIONS FOR FY2025 USING DECILE PROFILES")
print("="*70)

# Scenario 1: With early payment discount
print("\nScenario 1: With early payment discount...")
with_discount = simulate_with_decile_profiles(
    fy2025_df, 
    decile_profile, 
    discount_scenario='with_discount'
)

# Scenario 2: No discount
print("Scenario 2: No discount offered...")
no_discount = simulate_with_decile_profiles(
    fy2025_df, 
    decile_profile, 
    discount_scenario='no_discount'
)

print("✓ Simulations complete")

# ================================================================
# Summary statistics
# ================================================================
print("\n" + "="*70)
print(f"REVENUE COMPARISON: FY2025 ({FY2025_START.strftime('%d/%m/%Y')} - {FY2025_END.strftime('%d/%m/%Y')})")
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
            print(f"    cd = {cd}: {count:,} ({count/n_late*100:.1f}%)")
    
    # Invoice amounts (what customers owe - not your revenue)
    print(f"\nTotal Invoice Amounts (Customer Obligations):")
    print(f"  Undiscounted invoice total: ${df['total_undiscounted_price'].sum():,.2f}")
    print(f"  Discounted invoice total: ${df['total_discounted_price'].sum():,.2f}")
    print(f"  Discount amount: ${df['discount_amount'].sum():,.2f}")
    
    # Credit Card Revenue
    print(f"\nCredit Card Company Revenue (Interest Only):")
    print(f"  Interest revenue: ${df['interest_charged'].sum():,.2f}")
    print(f"  Total revenue: ${df['credit_card_revenue'].sum():,.2f}")
    
    # Decile breakdown
    print(f"\nRevenue by Decile:")
    decile_summary = df.groupby('decile').agg({
        'credit_card_revenue': 'sum',
        'is_late': 'sum',
        'total_undiscounted_price': 'count'
    })
    decile_summary.columns = ['revenue', 'n_late', 'n_invoices']
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
        'interest_revenue': df['interest_charged'].sum(),
        'total_revenue': df['credit_card_revenue'].sum()
    }

# Print summaries
summary_with = print_scenario_summary(with_discount, "FY2025 - WITH EARLY PAYMENT DISCOUNT")
summary_no = print_scenario_summary(no_discount, "FY2025 - NO DISCOUNT (INTEREST ON ALL)")

# ================================================================
# Comparison
# ================================================================
print("\n" + "="*70)
print("DIRECT COMPARISON - FY2025")
print("="*70)

revenue_diff = summary_no['total_revenue'] - summary_with['total_revenue']

print(f"\nCredit Card Revenue (Interest Only):")
print(f"  No Discount scenario: ${summary_no['total_revenue']:,.2f}")
print(f"  With Discount scenario: ${summary_with['total_revenue']:,.2f}")
print(f"  Difference: ${revenue_diff:+,.2f}")

if revenue_diff > 0:
    print(f"\n✓ NO DISCOUNT generates ${revenue_diff:,.2f} MORE revenue")
    pct_more = (revenue_diff / summary_with['total_revenue']) * 100
    print(f"  ({pct_more:.1f}% more than with discount)")
else:
    print(f"\n✓ WITH DISCOUNT generates ${abs(revenue_diff):,.2f} MORE revenue")
    pct_more = (abs(revenue_diff) / summary_no['total_revenue']) * 100
    print(f"  ({pct_more:.1f}% more than no discount)")

print(f"\nLate Payment Rates:")
print(f"  With Discount: {summary_with['pct_late']:.1f}% late")
print(f"  No Discount: {summary_no['pct_late']:.1f}% late")

# ================================================================
# Create comparison DataFrame
# ================================================================
comparison_df = pd.DataFrame([summary_with, summary_no])
output_csv = os.path.join(OUTPUT_DIR, 'FY2025_decile_comparison_summary.csv')
comparison_df.to_csv(output_csv, index=False)
print(f"\n✓ Saved comparison summary to: {output_csv}")

# # ================================================================
# # Save detailed simulations
# # ================================================================
# output_excel = os.path.join(OUTPUT_DIR, 'FY2025_decile_detailed_simulations.xlsx')
# with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
#     with_discount.to_excel(writer, sheet_name='With_Discount', index=False)
#     no_discount.to_excel(writer, sheet_name='No_Discount', index=False)
#     comparison_df.to_excel(writer, sheet_name='Summary_Comparison', index=False)

# print(f"✓ Saved detailed simulations to: {output_excel}")

# # ================================================================
# # Visualization: Monthly revenue comparison
# # ================================================================
# print("\n" + "="*70)
# print("CREATING VISUALIZATIONS")
# print("="*70)

# # Monthly aggregation
# def aggregate_by_month(df, scenario_name):
#     """Aggregate revenue by month"""
#     monthly = df.groupby(df['invoice_period'].dt.to_period('M')).agg({
#         'total_undiscounted_price': 'sum',
#         'total_discounted_price': 'sum',
#         'discount_amount': 'sum',
#         'interest_charged': 'sum',
#         'credit_card_revenue': 'sum',
#         'is_late': 'sum'
#     }).reset_index()
    
#     monthly['invoice_period'] = monthly['invoice_period'].dt.to_timestamp()
#     monthly['scenario'] = scenario_name
#     monthly['n_invoices'] = df.groupby(df['invoice_period'].dt.to_period('M')).size().values
#     monthly['pct_late'] = (monthly['is_late'] / monthly['n_invoices']) * 100
    
#     return monthly

# monthly_with = aggregate_by_month(with_discount, 'With Discount')
# monthly_no = aggregate_by_month(no_discount, 'No Discount')

# # Calculate cumulative values
# monthly_with['cumulative_revenue'] = monthly_with['credit_card_revenue'].cumsum()
# monthly_no['cumulative_revenue'] = monthly_no['credit_card_revenue'].cumsum()

# # Create visualization
# fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# # ================================================================
# # Plot 1: Cumulative Revenue Comparison
# # ================================================================
# ax1 = axes[0, 0]

# ax1.plot(monthly_with['invoice_period'], monthly_with['cumulative_revenue'], 
#          marker='o', linewidth=2.5, label='With Discount', color='#70AD47')
# ax1.plot(monthly_no['invoice_period'], monthly_no['cumulative_revenue'], 
#          marker='s', linewidth=2.5, label='No Discount', color='#4472C4')

# ax1.set_title('FY2025 - Cumulative Credit Card Revenue (Using Decile Profiles)', 
#               fontsize=14, fontweight='bold')
# ax1.set_xlabel('Month', fontsize=12)
# ax1.set_ylabel('Cumulative Revenue ($)', fontsize=12)
# ax1.legend(loc='upper left', fontsize=11)
# ax1.grid(True, alpha=0.3)
# ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# # ================================================================
# # Plot 2: Monthly Interest Revenue
# # ================================================================
# ax2 = axes[0, 1]

# width = 15
# x_with = monthly_with['invoice_period'] - pd.Timedelta(days=width/2)
# x_no = monthly_no['invoice_period'] + pd.Timedelta(days=width/2)

# ax2.bar(x_with, monthly_with['interest_charged'], width=width, 
#         label='Interest (With Discount)', color='#70AD47', alpha=0.7)

# ax2.bar(x_no, monthly_no['interest_charged'], width=width, 
#         label='Interest (No Discount)', color='#4472C4', alpha=0.7)

# ax2.set_title('FY2025 - Monthly Interest Revenue', 
#               fontsize=14, fontweight='bold')
# ax2.set_xlabel('Month', fontsize=12)
# ax2.set_ylabel('Monthly Interest Revenue ($)', fontsize=12)
# ax2.legend(loc='upper left', fontsize=9)
# ax2.grid(True, alpha=0.3, axis='y')
# ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# # ================================================================
# # Plot 3: Monthly Late Payment Rate
# # ================================================================
# ax3 = axes[1, 0]

# ax3.plot(monthly_with['invoice_period'], monthly_with['pct_late'], 
#          marker='o', linewidth=2.5, label='With Discount', color='#70AD47')
# ax3.plot(monthly_no['invoice_period'], monthly_no['pct_late'], 
#          marker='s', linewidth=2.5, label='No Discount', color='#4472C4')

# ax3.set_title('FY2025 - Monthly Late Payment Rate', 
#               fontsize=14, fontweight='bold')
# ax3.set_xlabel('Month', fontsize=12)
# ax3.set_ylabel('% Late Payments', fontsize=12)
# ax3.legend(loc='upper left', fontsize=11)
# ax3.grid(True, alpha=0.3)

# # ================================================================
# # Plot 4: Revenue by Decile
# # ================================================================
# ax4 = axes[1, 1]

# decile_revenue_with = with_discount.groupby('decile')['credit_card_revenue'].sum()
# decile_revenue_no = no_discount.groupby('decile')['credit_card_revenue'].sum()

# x_pos = np.arange(len(decile_revenue_with))
# width = 0.35

# ax4.bar(x_pos - width/2, decile_revenue_with, width, 
#         label='With Discount', color='#70AD47', alpha=0.7)
# ax4.bar(x_pos + width/2, decile_revenue_no, width, 
#         label='No Discount', color='#4472C4', alpha=0.7)

# ax4.set_title('FY2025 - Total Revenue by Decile', 
#               fontsize=14, fontweight='bold')
# ax4.set_xlabel('Decile (by Invoice Amount)', fontsize=12)
# ax4.set_ylabel('Total Revenue ($)', fontsize=12)
# ax4.set_xticks(x_pos)
# ax4.set_xticklabels([f'D{i}' for i in decile_revenue_with.index])
# ax4.legend(loc='upper left', fontsize=11)
# ax4.grid(True, alpha=0.3, axis='y')
# ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# # Format x-axes
# for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#     ax.tick_params(axis='x', rotation=45)

# plt.tight_layout()
# output_viz = os.path.join(OUTPUT_DIR, 'FY2025_decile_revenue_analysis.png')
# plt.savefig(output_viz, dpi=300, bbox_inches='tight')
# print(f"✓ Saved visualization to: {output_viz}")

# plt.show()

# print("\n" + "="*70)
# print("ANALYSIS COMPLETE - FY2025 WITH DECILE PROFILES")
# print("="*70)
# print(f"\nAll results saved to folder: {OUTPUT_DIR}/")
# print("Files created:")
# print(f"  1. FY2025_decile_comparison_summary.csv - Summary comparison table")
# print(f"  2. FY2025_decile_detailed_simulations.xlsx - Full simulation data")
# print(f"  3. FY2025_decile_revenue_analysis.png - 4-panel visualization")
# print("\nDecile-based simulation used:")
# print(f"  - {n_deciles} deciles based on invoice amount")
# print(f"  - Decile-specific late payment probabilities")
# print(f"  - Conditional delinquency level (cd) distributions")
# print("="*70)