import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================================================================
# CONFIGURATION
# ================================================================
BREAKEVEN_TARGET = 1_043_000
TARGET_DATE = pd.Timestamp("2025-03-31")  # Financial Year End 2025
RANDOM_SEED = 42
TOLERANCE = 1000  # How close to breakeven we need to be (in dollars)
MAX_ITERATIONS = 50  # Maximum binary search iterations

# Payment timing parameters
GRACE_PERIOD_DAYS = 20  # No interest/fees if paid within 20 days
NORMAL_PAYMENT_DAYS = 58.21  # Good customers from data
DELINQUENT_PAYMENT_DAYS = 30  # Pay at next invoice period (1 month)
SERIOUSLY_DELINQUENT_PAYMENT_DAYS = 90  # Pay at 3 months

# Interest and fees
ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
DAILY_INTEREST_RATE = ANNUAL_INTEREST_RATE / 365
LATE_FEE_PER_INVOICE = 10.00

# Delinquency split (from actual data)
DELINQUENT_PROPORTION = 123 / 6075  # ~2.02%
SERIOUSLY_DELINQUENT_PROPORTION = 58 / 6075  # ~0.95%
TOTAL_DEFAULT_RATE_ACTUAL = DELINQUENT_PROPORTION + SERIOUSLY_DELINQUENT_PROPORTION  # ~2.97%

# Interest calculation scenario
INTEREST_SCENARIO = 'full_amount'  # or 'discounted_amount'

# ================================================================
# Load and clean data
# ================================================================
print("="*70)
print("FINDING DEFAULT RATE TO REACH BREAKEVEN BY END OF FY2025")
print("WITH PAYMENT-BASED INTEREST AND LATE FEES")
print("="*70)

ats_grouped = pd.read_csv('ats_grouped_with_discounts.csv')
invoice_grouped = pd.read_csv('invoice_grouped_with_discounts.csv')

print(f"\nOriginal ATS data: {len(ats_grouped):,} invoices")
print(f"Original Invoice data: {len(invoice_grouped):,} invoices")

# Filter outliers
OUTLIER_THRESHOLD = 1_000_000
ats_clean = ats_grouped[ats_grouped['discount_amount'] <= OUTLIER_THRESHOLD].copy()
invoice_clean = invoice_grouped[invoice_grouped['discount_amount'] <= OUTLIER_THRESHOLD].copy()

print(f"Cleaned ATS: {len(ats_clean):,} invoices")
print(f"Cleaned Invoice: {len(invoice_clean):,} invoices")

# Combine
ats_clean['customer_type'] = 'ATS'
invoice_clean['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_clean, invoice_clean], ignore_index=True)

# ================================================================
# Robust date parsing
# ================================================================
def parse_invoice_period(series: pd.Series) -> pd.Series:
    """Robustly parse invoice_period."""
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

# Hard filter to remove epoch-era junk
min_valid = pd.Timestamp("2000-01-01")
max_valid = pd.Timestamp("2035-12-31")
combined_df = combined_df[(combined_df['invoice_period'] >= min_valid) & 
                          (combined_df['invoice_period'] <= max_valid)].copy()

combined_df = combined_df.sort_values('invoice_period').reset_index(drop=True)

# Calculate invoice amounts
if 'invoice_amount' not in combined_df.columns:
    print("\nCalculating invoice_amount from discount_amount (assuming 2% discount)...")
    combined_df['invoice_amount'] = combined_df['discount_amount'] / 0.02
else:
    print("\nUsing 'invoice_amount' column from data")

combined_df['discounted_invoice_amount'] = combined_df['invoice_amount'] - combined_df['discount_amount']

# Filter to only include data up to end of FY2025
combined_df_fy2025 = combined_df[combined_df['invoice_period'] <= TARGET_DATE].copy()
print(f"\nInvoices through end of FY2025 (March 31, 2025): {len(combined_df_fy2025):,}")

print(f"\nPayment configuration:")
print(f"  Grace period: {GRACE_PERIOD_DAYS} days")
print(f"  Interest rate: {ANNUAL_INTEREST_RATE*100:.2f}% p.a. ({DAILY_INTEREST_RATE*365*100:.4f}% daily)")
print(f"  Late fee: ${LATE_FEE_PER_INVOICE} per invoice")
print(f"  Interest scenario: {INTEREST_SCENARIO}")

# ================================================================
# Function to calculate interest and late fees
# ================================================================
def calculate_payment_charges(payment_days, invoice_amount, grace_period=GRACE_PERIOD_DAYS):
    """Calculate interest and late fees based on payment timing"""
    if payment_days <= grace_period:
        return {
            'days_past_grace': 0,
            'interest_charged': 0,
            'late_fee': 0,
            'total_charges': 0
        }
    else:
        days_past_grace = payment_days - grace_period
        interest = invoice_amount * DAILY_INTEREST_RATE * days_past_grace
        late_fee = LATE_FEE_PER_INVOICE
        
        return {
            'days_past_grace': days_past_grace,
            'interest_charged': interest,
            'late_fee': late_fee,
            'total_charges': interest + late_fee
        }

# ================================================================
# Function to calculate total revenue for a given default rate
# ================================================================
def calculate_revenue_at_date(df, default_rate, target_date, seed=RANDOM_SEED, 
                              interest_scenario=INTEREST_SCENARIO):
    """
    Calculate cumulative revenue (discount retained + interest + late fees) 
    by a target date for a given default rate.
    
    Returns: (total_revenue, period_summary_df)
    """
    df_scenario = df.copy()
    np.random.seed(seed)
    
    n_invoices = len(df_scenario)
    n_defaults = int(n_invoices * default_rate)
    
    # Split defaults into delinquency types proportionally
    if n_defaults > 0:
        seriously_delinquent_rate = SERIOUSLY_DELINQUENT_PROPORTION / TOTAL_DEFAULT_RATE_ACTUAL
        n_seriously_delinquent = int(n_defaults * seriously_delinquent_rate)
        n_delinquent = n_defaults - n_seriously_delinquent
    else:
        n_seriously_delinquent = 0
        n_delinquent = 0
    
    # Randomly assign delinquency status
    if n_defaults > 0:
        default_indices = np.random.choice(df_scenario.index, size=n_defaults, replace=False)
        
        if n_seriously_delinquent > 0:
            seriously_delinquent_indices = np.random.choice(default_indices, 
                                                           size=n_seriously_delinquent, 
                                                           replace=False)
        else:
            seriously_delinquent_indices = np.array([])
        
        delinquent_indices = np.setdiff1d(default_indices, seriously_delinquent_indices)
    else:
        default_indices = np.array([])
        seriously_delinquent_indices = np.array([])
        delinquent_indices = np.array([])
    
    # Initialize columns
    df_scenario['paid_on_time'] = True
    df_scenario['delinquency_type'] = 'Normal'
    df_scenario['payment_days'] = NORMAL_PAYMENT_DAYS
    
    # Assign payment days based on delinquency type
    if len(delinquent_indices) > 0:
        df_scenario.loc[delinquent_indices, 'paid_on_time'] = False
        df_scenario.loc[delinquent_indices, 'delinquency_type'] = 'Delinquent'
        df_scenario.loc[delinquent_indices, 'payment_days'] = DELINQUENT_PAYMENT_DAYS
    
    if len(seriously_delinquent_indices) > 0:
        df_scenario.loc[seriously_delinquent_indices, 'paid_on_time'] = False
        df_scenario.loc[seriously_delinquent_indices, 'delinquency_type'] = 'Seriously Delinquent'
        df_scenario.loc[seriously_delinquent_indices, 'payment_days'] = SERIOUSLY_DELINQUENT_PAYMENT_DAYS
    
    # Calculate discount retained
    df_scenario['discount_retained'] = df_scenario.apply(
        lambda row: row['discount_amount'] if not row['paid_on_time'] else 0, 
        axis=1
    )
    
    # Determine which invoice amount to use for interest calculation
    if interest_scenario == 'full_amount':
        df_scenario['interest_base_amount'] = df_scenario['invoice_amount']
    else:  # 'discounted_amount'
        df_scenario['interest_base_amount'] = df_scenario['discounted_invoice_amount']
    
    # Calculate interest and late fees
    charges_list = []
    for idx, row in df_scenario.iterrows():
        charges = calculate_payment_charges(
            payment_days=row['payment_days'],
            invoice_amount=row['interest_base_amount'],
            grace_period=GRACE_PERIOD_DAYS
        )
        charges_list.append(charges)
    
    charges_df = pd.DataFrame(charges_list)
    df_scenario['interest_earned'] = charges_df['interest_charged']
    df_scenario['late_fees'] = charges_df['late_fee']
    
    # Total revenue impact
    df_scenario['total_revenue_impact'] = (df_scenario['discount_retained'] + 
                                           df_scenario['interest_earned'] + 
                                           df_scenario['late_fees'])
    
    # Group by period
    period_summary = df_scenario.groupby('invoice_period').agg({
        'discount_amount': 'sum',
        'discount_retained': 'sum',
        'interest_earned': 'sum',
        'late_fees': 'sum',
        'total_revenue_impact': 'sum',
        'paid_on_time': ['sum', 'count']
    }).reset_index()
    
    period_summary.columns = ['invoice_period', 'total_discount_offered', 'discount_retained',
                              'interest_earned', 'late_fees', 'total_revenue_impact',
                              'n_paid_on_time', 'n_total_invoices']
    
    # Calculate cumulative values
    period_summary['cumulative_discount_offered'] = period_summary['total_discount_offered'].cumsum()
    period_summary['cumulative_discount_retained'] = period_summary['discount_retained'].cumsum()
    period_summary['cumulative_interest_earned'] = period_summary['interest_earned'].cumsum()
    period_summary['cumulative_late_fees'] = period_summary['late_fees'].cumsum()
    period_summary['cumulative_total_revenue'] = period_summary['total_revenue_impact'].cumsum()
    
    # Get total revenue by target date
    mask_target = period_summary['invoice_period'] <= target_date
    if mask_target.any():
        revenue_by_target = period_summary[mask_target]['cumulative_total_revenue'].iloc[-1]
    else:
        revenue_by_target = 0
    
    return revenue_by_target, period_summary

# ================================================================
# Binary search to find the required default rate
# ================================================================
print(f"\nTarget: ${BREAKEVEN_TARGET:,} by {TARGET_DATE.strftime('%Y-%m-%d')} (FY2025 End)")
print(f"Tolerance: ±${TOLERANCE:,}")
print("\nStarting binary search...")
print("-"*70)

low_rate = 0.0
high_rate = 1.0
best_rate = None
best_revenue = None
best_period_summary = None
iteration = 0

search_history = []

while iteration < MAX_ITERATIONS:
    iteration += 1
    mid_rate = (low_rate + high_rate) / 2
    
    revenue, period_summary = calculate_revenue_at_date(
        combined_df_fy2025, mid_rate, TARGET_DATE, interest_scenario=INTEREST_SCENARIO
    )
    
    gap = revenue - BREAKEVEN_TARGET
    
    search_history.append({
        'iteration': iteration,
        'default_rate': mid_rate,
        'total_revenue': revenue,
        'gap': gap
    })
    
    print(f"Iteration {iteration:2d}: Rate={mid_rate*100:6.3f}% → Revenue=${revenue:>12,.2f} (Gap: ${gap:>10,.2f})")
    
    # Check if we're within tolerance
    if abs(gap) <= TOLERANCE:
        best_rate = mid_rate
        best_revenue = revenue
        best_period_summary = period_summary
        print(f"\n✓ FOUND! Default rate of {best_rate*100:.3f}% achieves breakeven within tolerance")
        break
    
    # Adjust search bounds
    if revenue < BREAKEVEN_TARGET:
        # Need more revenue (higher default rate)
        low_rate = mid_rate
    else:
        # Too much revenue (lower default rate)
        high_rate = mid_rate
    
    # Store best attempt
    if best_revenue is None or abs(gap) < abs(best_revenue - BREAKEVEN_TARGET):
        best_rate = mid_rate
        best_revenue = revenue
        best_period_summary = period_summary

if iteration >= MAX_ITERATIONS:
    print(f"\n⚠ Reached maximum iterations. Best result:")

# ================================================================
# Display final results
# ================================================================
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Optimal default rate: {best_rate*100:.3f}%")
print(f"Total revenue by end of FY2025: ${best_revenue:,.2f}")
print(f"Target (breakeven): ${BREAKEVEN_TARGET:,.2f}")
print(f"Difference: ${best_revenue - BREAKEVEN_TARGET:+,.2f}")

# Calculate component breakdown at optimal rate
mask_target = best_period_summary['invoice_period'] <= TARGET_DATE
if mask_target.any():
    data_at_target = best_period_summary[mask_target].iloc[-1]
    discount_retained = data_at_target['cumulative_discount_retained']
    interest_earned = data_at_target['cumulative_interest_earned']
    late_fees = data_at_target['cumulative_late_fees']
    
    print(f"\nRevenue breakdown:")
    print(f"  Discount retained: ${discount_retained:,.2f} ({discount_retained/best_revenue*100:.1f}%)")
    print(f"  Interest earned: ${interest_earned:,.2f} ({interest_earned/best_revenue*100:.1f}%)")
    print(f"  Late fees: ${late_fees:,.2f} ({late_fees/best_revenue*100:.1f}%)")

print(f"\nThis means approximately {int(len(combined_df_fy2025) * best_rate):,} out of {len(combined_df_fy2025):,} invoices")
print(f"would need to NOT pay on time for the strategy to break even by end of FY2025.")

# ================================================================
# Calculate and display additional scenarios for comparison
# ================================================================
print("\n" + "="*70)
print("COMPARISON WITH OTHER DEFAULT RATES")
print("="*70)

comparison_rates = [0.05, 0.10, 0.15, 0.20, best_rate, 0.30, 0.40, 0.50]
comparison_rates = sorted(set(comparison_rates))  # Remove duplicates and sort

comparison_results = []
scenario_data = {}

for rate in comparison_rates:
    revenue, period_summary = calculate_revenue_at_date(
        combined_df_fy2025, rate, TARGET_DATE, interest_scenario=INTEREST_SCENARIO
    )
    
    # Get component breakdown
    mask_target = period_summary['invoice_period'] <= TARGET_DATE
    if mask_target.any():
        data_at_target = period_summary[mask_target].iloc[-1]
        discount_retained = data_at_target['cumulative_discount_retained']
        interest_earned = data_at_target['cumulative_interest_earned']
        late_fees = data_at_target['cumulative_late_fees']
    else:
        discount_retained = 0
        interest_earned = 0
        late_fees = 0
    
    comparison_results.append({
        'default_rate': rate,
        'total_revenue_fy2025': revenue,
        'discount_retained': discount_retained,
        'interest_earned': interest_earned,
        'late_fees': late_fees,
        'gap_from_breakeven': revenue - BREAKEVEN_TARGET,
        'reaches_breakeven': revenue >= BREAKEVEN_TARGET,
        'n_defaults': int(len(combined_df_fy2025) * rate)
    })
    
    scenario_data[rate] = period_summary

comparison_df = pd.DataFrame(comparison_results)

for _, row in comparison_df.iterrows():
    status = "✓ BREAKS EVEN" if row['reaches_breakeven'] else "✗ Below target"
    print(f"\n{row['default_rate']*100:5.2f}% default: {status}")
    print(f"  Total revenue: ${row['total_revenue_fy2025']:,.2f}")
    print(f"    - Discount: ${row['discount_retained']:,.2f}")
    print(f"    - Interest: ${row['interest_earned']:,.2f}")
    print(f"    - Late fees: ${row['late_fees']:,.2f}")
    print(f"  Gap: ${row['gap_from_breakeven']:+,.2f}")
    print(f"  Defaults: {row['n_defaults']:,} invoices")

# ================================================================
# Create visualization
# ================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATION")
print("="*70)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))

# Color palette
colors = ['#4472C4', '#70AD47', '#FFC000', '#A64D79', '#FF6B6B', '#845EC2']

# Top plot: Cumulative revenue over time
start_date = pd.Timestamp("2023-10-01")

for i, rate in enumerate([0.10, 0.15, 0.20, best_rate, 0.30, 0.35]):
    if rate in scenario_data:
        period_data = scenario_data[rate]
        color = colors[i % len(colors)]
        linewidth = 3 if rate == best_rate else 2
        alpha = 1.0 if rate == best_rate else 0.7
        
        label = f"{rate*100:.1f}% default" + (" (Breakeven rate)" if rate == best_rate else "")
        
        mark_every = max(1, len(period_data)//15)
        ax1.plot(period_data['invoice_period'], period_data['cumulative_total_revenue'],
                color=color, linewidth=linewidth, alpha=alpha,
                label=label, marker='o', markersize=4, markevery=mark_every)

# Reference line for discount offered
ref_data = scenario_data[list(scenario_data.keys())[0]]
ax1.plot(ref_data['invoice_period'], ref_data['cumulative_discount_offered'],
        color="#222222", linewidth=1.5, alpha=0.5,
        label='Total discount offered (if all pay early)',
        linestyle='--', marker='s', markersize=3, markevery=max(1, len(ref_data)//15))

# Breakeven line
ax1.axhline(y=BREAKEVEN_TARGET, color='#FF0000', linewidth=2,
           linestyle='-', alpha=0.8, label=f'Breakeven target (${BREAKEVEN_TARGET:,})', zorder=10)

# End of FY2025 line
ax1.axvline(x=TARGET_DATE, color='#8B7BA8', linewidth=2,
           linestyle=':', alpha=0.7, label='End of FY2025 (Mar 31, 2025)', zorder=9)

# Add year-end markers
ax1.axvline(x=pd.Timestamp("2023-12-31"), color='#D3D3D3', linewidth=1,
           linestyle=':', alpha=0.5, label='End of CY2023')
ax1.axvline(x=pd.Timestamp("2024-12-31"), color='#D3D3D3', linewidth=1,
           linestyle=':', alpha=0.5, label='End of CY2024')

ax1.set_xlim(left=start_date, right=TARGET_DATE)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.tick_params(axis='x', rotation=45)

ax1.set_xlabel('Invoice Period', fontsize=13, fontweight='bold')
ax1.set_ylabel('$ Revenue Amount', fontsize=13, fontweight='bold')
title_text = f'Total Revenue (Discount + Interest + Late Fees): Finding Breakeven Rate by FY2025'
ax1.set_title(title_text, fontsize=16, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.set_facecolor('#F8F8F8')

# Bottom plot: Default rate vs. total revenue
revenue_amounts = [result['total_revenue_fy2025'] for result in comparison_results]
default_rates_pct = [result['default_rate']*100 for result in comparison_results]

ax2.plot(default_rates_pct, revenue_amounts, 
         color='#4472C4', linewidth=3, marker='o', markersize=8, label='Total Revenue')

# Mark the optimal point
ax2.scatter([best_rate*100], [best_revenue], 
           color='#FF0000', s=200, marker='*', zorder=10,
           label=f'Breakeven rate: {best_rate*100:.2f}%', edgecolors='black', linewidths=2)

# Breakeven line
ax2.axhline(y=BREAKEVEN_TARGET, color='#FF0000', linewidth=2,
           linestyle='--', alpha=0.6, label=f'Breakeven target (${BREAKEVEN_TARGET:,})')

ax2.set_xlabel('Default Rate (%)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Total Revenue by End of FY2025 ($)', fontsize=13, fontweight='bold')
ax2.set_title('Default Rate vs. Total Revenue (Through FY2025)',
             fontsize=16, fontweight='bold', pad=15)
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
ax2.set_facecolor('#F8F8F8')

plt.tight_layout()
plt.savefig('breakeven_rate_fy2025_payment_based.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to: breakeven_rate_fy2025_payment_based.png")

# ================================================================
# Save results
# ================================================================
comparison_df.to_csv('breakeven_rate_comparison_payment_based_fy2025.csv', index=False)
print("✓ Saved comparison results to: breakeven_rate_comparison_payment_based_fy2025.csv")

# Save search history
search_df = pd.DataFrame(search_history)
search_df.to_csv('binary_search_history_payment_based_fy2025.csv', index=False)
print("✓ Saved search history to: binary_search_history_payment_based_fy2025.csv")

# Save detailed period data for optimal rate
best_period_summary.to_csv('optimal_rate_period_detail_payment_based_fy2025.csv', index=False)
print("✓ Saved optimal rate period details to: optimal_rate_period_detail_payment_based_fy2025.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nKEY FINDING: A default rate of {best_rate*100:.3f}% is needed to break even by end of FY2025")
print(f"This includes discount retention, interest charges (>{GRACE_PERIOD_DAYS} days), and late fees.")
print(f"Interest calculated on: {INTEREST_SCENARIO}")
print(f"\nNote: FY2025 = April 1, 2024 through March 31, 2025")