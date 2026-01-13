import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pickle

# ================================================================
# CONFIGURATION
# ================================================================
ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
LATE_FEE = 10.00  # $10 per late invoice
RANDOM_SEED = 42
START_DATE = pd.Timestamp("2023-12-20")  # Start from 20th December 2023
OUTPUT_DIR = "FY2025_outputs_transformed"  # Updated output directory for transformed data

# Note: invoice_period is already set to 20th of month (the due date)
# Any payment after invoice_period is overdue

# Create output directory if it doesn't exist
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# Load invoice data (TRANSFORMED VERSIONS)
# ================================================================
print("="*70)
print("LOADING TRANSFORMED INVOICE DATA")
print("="*70)

# Load combined invoice data (USING TRANSFORMED FILES)
ats_grouped = pd.read_csv('ats_grouped_with_discounts_transformed.csv')
invoice_grouped = pd.read_csv('invoice_grouped_with_discounts_transformed.csv')

# Combine datasets
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"Total invoices loaded: {len(combined_df):,}")

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

# Filter to data from 20/12/2023 onwards
filtered_df = combined_df[combined_df['invoice_period'] >= START_DATE].copy()

print(f"Invoices from {START_DATE.strftime('%Y-%m-%d')} onwards: {len(filtered_df):,}")

if len(filtered_df) == 0:
    print("\n⚠ WARNING: No invoices found from this date onwards!")
    print("Available date range in data:")
    print(f"  Min: {combined_df['invoice_period'].min()}")
    print(f"  Max: {combined_df['invoice_period'].max()}")
    exit()

# ================================================================
# Load payment timing profiles
# ================================================================
print("\n" + "="*70)
print("LOADING PAYMENT TIMING PROFILES")
print("="*70)

try:
    with open('payment_profiles/payment_profiles.pkl', 'rb') as f:
        payment_profiles = pickle.load(f)
    print(f"✓ Loaded {len(payment_profiles)} payment profiles")
except FileNotFoundError:
    print("⚠ Payment profiles not found. Run create_payment_profiles.py first.")
    print("Using fallback: loading master dataset to create profiles on-the-fly...")
    
    # Fallback: load master dataset
    file_path_master = r"t:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo2\Clean Code\master_dataset_complete.parquet"
    df_master = pd.read_parquet(file_path_master)
    
    # Create simple profile
    payment_profiles = {
        'overall': {
            'mean': df_master['payment_days'].mean(),
            'median': df_master['payment_days'].median(),
            'std': df_master['payment_days'].std(),
            'raw_data': df_master['payment_days'].dropna().values
        }
    }
    print(f"✓ Created fallback profile with {len(payment_profiles['overall']['raw_data']):,} observations")

# ================================================================
# Simulation function
# ================================================================
def simulate_invoice_payments_with_interest(invoices_df, payment_profile, discount_scenario):
    """
    Simulate invoice payments and calculate revenue
    
    Parameters:
    - invoices_df: DataFrame with invoice data
    - payment_profile: Payment timing profile from historical data
    - discount_scenario: 'with_discount' or 'no_discount'
    
    Returns:
    - DataFrame with simulated payment details and revenue calculations
    
    Note: invoice_period is already the due date (20th of month)
    """
    np.random.seed(RANDOM_SEED)
    
    simulated = invoices_df.copy()
    n_invoices = len(simulated)
    
    # Simulate payment timing (days from invoice date)
    raw_payment_days = np.random.choice(
        payment_profile['raw_data'], 
        size=n_invoices, 
        replace=True
    )
    
    # Adjust payment days: subtract 20 and clip to minimum of 0
    # This accounts for invoice_period already being set to the 20th (due date)
    adjusted_payment_days = np.maximum(raw_payment_days - 20, 0)
    simulated['simulated_payment_days'] = adjusted_payment_days
    
    # invoice_period is the due date (20th of month)
    simulated['due_date'] = simulated['invoice_period']
    
    # Calculate actual payment date
    simulated['payment_date'] = simulated['invoice_period'] + pd.to_timedelta(simulated['simulated_payment_days'], unit='D')
    
    # Determine if paid late (anything after due date)
    simulated['days_overdue'] = (simulated['payment_date'] - simulated['due_date']).dt.days
    simulated['days_overdue'] = simulated['days_overdue'].clip(lower=0)  # No negative overdue days
    simulated['is_late'] = simulated['days_overdue'] > 0
    
    # ================================================================
    # Calculate amounts based on discount scenario
    # ================================================================
    if discount_scenario == 'with_discount':
        # With discount: everyone gets the discount, but only on-time payers avoid interest
        # Principal is always the discounted amount
        simulated['principal_amount'] = simulated['total_discounted_price']
        simulated['paid_on_time'] = ~simulated['is_late']
        simulated['discount_applied'] = simulated['discount_amount']
        
    else:  # no_discount
        # No discount: everyone pays full undiscounted amount
        simulated['principal_amount'] = simulated['total_undiscounted_price']
        simulated['paid_on_time'] = False  # No discount given to anyone
        simulated['discount_applied'] = 0
    
    # ================================================================
    # Calculate interest charges (on all overdue days)
    # ================================================================
    # Daily interest rate
    daily_rate = ANNUAL_INTEREST_RATE / 365
    
    # Interest = Principal × Daily Rate × Days Overdue
    # No grace period - interest starts immediately when overdue
    simulated['interest_charged'] = (
        simulated['principal_amount'] * 
        daily_rate * 
        simulated['days_overdue']
    )
    
    # Late fees (only if late)
    simulated['late_fee_charged'] = simulated['is_late'].astype(int) * LATE_FEE
    
    # ================================================================
    # Calculate total revenue components
    # ================================================================
    # For a credit card company, REVENUE = Interest + Late Fees only
    # Invoice amounts are what customers owe (not revenue to the credit card company)
    
    simulated['credit_card_revenue'] = (
        simulated['interest_charged'] + 
        simulated['late_fee_charged']
    )
    
    # Track total amounts for comparison (not revenue)
    simulated['total_invoice_amount_discounted'] = simulated['total_discounted_price']
    simulated['total_invoice_amount_undiscounted'] = simulated['total_undiscounted_price']
    
    return simulated

# ================================================================
# Run both scenarios
# ================================================================
print("\n" + "="*70)
print("RUNNING SIMULATIONS")
print("="*70)

# Use overall profile (or could segment by customer behavior)
profile = payment_profiles['overall']

# Scenario 1: With early payment discount
print("\nScenario 1: With early payment discount...")
with_discount = simulate_invoice_payments_with_interest(
    filtered_df, 
    profile, 
    discount_scenario='with_discount'
)

# Scenario 2: No discount (everyone pays interest on full amount)
print("Scenario 2: No discount offered...")
no_discount = simulate_invoice_payments_with_interest(
    filtered_df, 
    profile, 
    discount_scenario='no_discount'
)

# ================================================================
# Summary statistics
# ================================================================
print("\n" + "="*70)
print(f"REVENUE COMPARISON: FROM {START_DATE.strftime('%d/%m/%Y')} ONWARDS")
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
    print(f"  Paid on time (by due date): {n_on_time:,} ({n_on_time/total_invoices*100:.1f}%)")
    print(f"  Paid late (after due date): {n_late:,} ({n_late/total_invoices*100:.1f}%)")
    
    if n_late > 0:
        print(f"  Avg days overdue (late invoices): {df[df['is_late']]['days_overdue'].mean():.1f}")
        print(f"  Avg interest per late invoice: ${df[df['is_late']]['interest_charged'].mean():,.2f}")
    
    # Invoice amounts (what customers owe - not your revenue)
    print(f"\nTotal Invoice Amounts (Customer Obligations):")
    print(f"  Undiscounted invoice total: ${df['total_undiscounted_price'].sum():,.2f}")
    print(f"  Discounted invoice total: ${df['total_discounted_price'].sum():,.2f}")
    print(f"  Discount amount: ${df['discount_amount'].sum():,.2f}")
    
    # Credit Card Revenue (YOUR revenue - interest + late fees only)
    print(f"\nCredit Card Company Revenue (Interest + Late Fees):")
    print(f"  Interest revenue: ${df['interest_charged'].sum():,.2f}")
    print(f"  Late fee revenue: ${df['late_fee_charged'].sum():,.2f}")
    print(f"  Total revenue: ${df['credit_card_revenue'].sum():,.2f}")
    
    return {
        'scenario': scenario_name,
        'total_invoices': total_invoices,
        'n_late': n_late,
        'n_on_time': n_on_time,
        'pct_late': n_late/total_invoices*100,
        'avg_days_overdue': df[df['is_late']]['days_overdue'].mean() if n_late > 0 else 0,
        'total_undiscounted': df['total_undiscounted_price'].sum(),
        'total_discounted': df['total_discounted_price'].sum(),
        'discount_amount': df['discount_amount'].sum(),
        'interest_revenue': df['interest_charged'].sum(),
        'late_fee_revenue': df['late_fee_charged'].sum(),
        'total_revenue': df['credit_card_revenue'].sum()
    }

# Print summaries
summary_with = print_scenario_summary(with_discount, "WITH EARLY PAYMENT DISCOUNT")
summary_no = print_scenario_summary(no_discount, "NO DISCOUNT (INTEREST ON ALL)")

# ================================================================
# Comparison
# ================================================================
print("\n" + "="*70)
print("DIRECT COMPARISON")
print("="*70)

revenue_diff = summary_no['total_revenue'] - summary_with['total_revenue']

print(f"\nCredit Card Revenue (Interest + Late Fees):")
print(f"  No Discount scenario: ${summary_no['total_revenue']:,.2f}")
print(f"  With Discount scenario: ${summary_with['total_revenue']:,.2f}")
print(f"  Difference: ${revenue_diff:+,.2f}")

if revenue_diff > 0:
    print(f"\n✓ NO DISCOUNT generates ${revenue_diff:,.2f} MORE revenue")
else:
    print(f"\n✓ WITH DISCOUNT generates ${abs(revenue_diff):,.2f} MORE revenue")

print(f"\nRevenue Breakdown:")
print(f"  Interest revenue difference: ${summary_no['interest_revenue'] - summary_with['interest_revenue']:+,.2f}")
print(f"  Late fee revenue difference: ${summary_no['late_fee_revenue'] - summary_with['late_fee_revenue']:+,.2f}")

print(f"\nInvoice Amounts (Customer Obligations - not your revenue):")
print(f"  Total undiscounted invoices: ${summary_no['total_undiscounted']:,.2f}")
print(f"  Total discounted invoices: ${summary_with['total_discounted']:,.2f}")
print(f"  Discount amount: ${summary_with['discount_amount']:,.2f}")

# ================================================================
# Create comparison DataFrame
# ================================================================
comparison_df = pd.DataFrame([summary_with, summary_no])
output_csv = os.path.join(OUTPUT_DIR, 'comparison_summary.csv')
comparison_df.to_csv(output_csv, index=False)
print(f"\n✓ Saved comparison summary to: {output_csv}")

# ================================================================
# Save detailed simulations
# ================================================================
output_excel = os.path.join(OUTPUT_DIR, 'detailed_simulations.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    with_discount.to_excel(writer, sheet_name='With_Discount', index=False)
    no_discount.to_excel(writer, sheet_name='No_Discount', index=False)
    comparison_df.to_excel(writer, sheet_name='Summary_Comparison', index=False)

print(f"✓ Saved detailed simulations to: {output_excel}")

# ================================================================
# Visualization: Monthly revenue comparison
# ================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Monthly aggregation
def aggregate_by_month(df, scenario_name):
    """Aggregate revenue by month"""
    monthly = df.groupby(df['invoice_period'].dt.to_period('M')).agg({
        'total_undiscounted_price': 'sum',
        'total_discounted_price': 'sum',
        'discount_amount': 'sum',
        'interest_charged': 'sum',
        'late_fee_charged': 'sum',
        'credit_card_revenue': 'sum'
    }).reset_index()
    
    monthly['invoice_period'] = monthly['invoice_period'].dt.to_timestamp()
    monthly['scenario'] = scenario_name
    
    return monthly

monthly_with = aggregate_by_month(with_discount, 'With Discount')
monthly_no = aggregate_by_month(no_discount, 'No Discount')

# Calculate cumulative values
monthly_with['cumulative_undiscounted'] = monthly_with['total_undiscounted_price'].cumsum()
monthly_with['cumulative_discounted'] = monthly_with['total_discounted_price'].cumsum()
monthly_with['cumulative_revenue'] = monthly_with['credit_card_revenue'].cumsum()

monthly_no['cumulative_undiscounted'] = monthly_no['total_undiscounted_price'].cumsum()
monthly_no['cumulative_revenue'] = monthly_no['credit_card_revenue'].cumsum()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# ================================================================
# Plot 1: Cumulative Invoice Amounts + Revenue (With Discount)
# ================================================================
ax1 = axes[0, 0]

# Invoice amounts (what customers owe)
ax1.plot(monthly_with['invoice_period'], monthly_with['cumulative_undiscounted'], 
         marker='o', linewidth=2.5, label='Total Invoices (Undiscounted)', 
         color='#4472C4', linestyle='--', alpha=0.7)
ax1.plot(monthly_with['invoice_period'], monthly_with['cumulative_discounted'], 
         marker='s', linewidth=2.5, label='Total Invoices (Discounted)', 
         color='#70AD47', linestyle='--', alpha=0.7)

# Credit card revenue (interest + late fees)
ax1.plot(monthly_with['invoice_period'], monthly_with['cumulative_revenue'], 
         marker='^', linewidth=3, label='Cumulative Revenue (Interest + Late Fees)', 
         color='#FF6B6B', zorder=10)

ax1.set_title('WITH DISCOUNT: Invoice Amounts vs Credit Card Revenue', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Cumulative Amount ($)', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# ================================================================
# Plot 2: Cumulative Invoice Amounts + Revenue (No Discount)
# ================================================================
ax2 = axes[0, 1]

# Invoice amounts
ax2.plot(monthly_no['invoice_period'], monthly_no['cumulative_undiscounted'], 
         marker='o', linewidth=2.5, label='Total Invoices (Undiscounted)', 
         color='#4472C4', linestyle='--', alpha=0.7)

# Credit card revenue (interest + late fees)
ax2.plot(monthly_no['invoice_period'], monthly_no['cumulative_revenue'], 
         marker='^', linewidth=3, label='Cumulative Revenue (Interest + Late Fees)', 
         color='#FF6B6B', zorder=10)

ax2.set_title('NO DISCOUNT: Invoice Amounts vs Credit Card Revenue', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Cumulative Amount ($)', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# ================================================================
# Plot 3: Revenue Comparison (Both Scenarios)
# ================================================================
ax3 = axes[1, 0]

ax3.plot(monthly_with['invoice_period'], monthly_with['cumulative_revenue'], 
         marker='o', linewidth=2.5, label='With Discount', color='#70AD47')
ax3.plot(monthly_no['invoice_period'], monthly_no['cumulative_revenue'], 
         marker='s', linewidth=2.5, label='No Discount', color='#4472C4')

ax3.set_title('Cumulative Credit Card Revenue Comparison', 
              fontsize=14, fontweight='bold')
ax3.set_xlabel('Month', fontsize=12)
ax3.set_ylabel('Cumulative Revenue ($)', fontsize=12)
ax3.legend(loc='upper left', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# ================================================================
# Plot 4: Monthly Revenue Breakdown
# ================================================================
ax4 = axes[1, 1]

# Stack bar chart for revenue components
width = 15  # bar width in days
x_with = monthly_with['invoice_period'] - pd.Timedelta(days=width/2)
x_no = monthly_no['invoice_period'] + pd.Timedelta(days=width/2)

ax4.bar(x_with, monthly_with['interest_charged'], width=width, 
        label='Interest (With Discount)', color='#70AD47', alpha=0.7)
ax4.bar(x_with, monthly_with['late_fee_charged'], width=width, 
        bottom=monthly_with['interest_charged'],
        label='Late Fees (With Discount)', color='#A9D18E', alpha=0.7)

ax4.bar(x_no, monthly_no['interest_charged'], width=width, 
        label='Interest (No Discount)', color='#4472C4', alpha=0.7)
ax4.bar(x_no, monthly_no['late_fee_charged'], width=width, 
        bottom=monthly_no['interest_charged'],
        label='Late Fees (No Discount)', color='#8FAADC', alpha=0.7)

ax4.set_title('Monthly Revenue Components: Interest vs Late Fees', 
              fontsize=14, fontweight='bold')
ax4.set_xlabel('Month', fontsize=12)
ax4.set_ylabel('Monthly Revenue ($)', fontsize=12)
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Format all x-axes
for ax in axes.flat:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = os.path.join(OUTPUT_DIR, 'revenue_analysis_visualization.png')
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_viz}")

plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll results saved to folder: {OUTPUT_DIR}/")
print("Files created:")
print(f"  1. comparison_summary.csv - Summary comparison table")
print(f"  2. detailed_simulations.xlsx - Full simulation data (both scenarios)")
print(f"  3. revenue_analysis_visualization.png - 4-panel visualization")
print("="*70)