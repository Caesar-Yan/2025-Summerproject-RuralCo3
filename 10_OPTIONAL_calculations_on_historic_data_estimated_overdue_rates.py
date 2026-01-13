import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================================================================
# CONFIGURATION - Delinquency-based scenarios
# ================================================================
RANDOM_SEED = 42

# Delinquency proportions from behavior analysis
DELINQUENT_PROPORTION = 123 / 6075  # ~2.02%
SERIOUSLY_DELINQUENT_PROPORTION = 58 / 6075  # ~0.95%
TOTAL_DEFAULT_RATE = DELINQUENT_PROPORTION + SERIOUSLY_DELINQUENT_PROPORTION  # ~2.97%

# Interest and fees
ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
MONTHLY_INTEREST_RATE = ANNUAL_INTEREST_RATE / 12  # ~1.996% per month
LATE_FEE_PER_INVOICE = 10.00

# Payment delays
DELINQUENT_DELAY_MONTHS = 1
SERIOUSLY_DELINQUENT_DELAY_MONTHS = 3

# Scenarios to compare
INTEREST_SCENARIOS = ['full_amount', 'discounted_amount']

print(f"{'='*70}")
print(f"DELINQUENCY-BASED DISCOUNT RETENTION ANALYSIS")
print(f"{'='*70}")
print(f"\nParameters:")
print(f"  Total default rate: {TOTAL_DEFAULT_RATE*100:.2f}%")
print(f"  - Delinquent: {DELINQUENT_PROPORTION*100:.2f}% (1 month delay)")
print(f"  - Seriously delinquent: {SERIOUSLY_DELINQUENT_PROPORTION*100:.2f}% (3 month delay)")
print(f"  Interest rate: {ANNUAL_INTEREST_RATE*100:.2f}% p.a. ({MONTHLY_INTEREST_RATE*100:.2f}% per month)")
print(f"  Late fee: ${LATE_FEE_PER_INVOICE} per invoice")

# ================================================================
# Load and clean data
# ================================================================
ats_grouped = pd.read_csv('ats_grouped_with_discounts.csv')
invoice_grouped = pd.read_csv('invoice_grouped_with_discounts.csv')

print(f"\nOriginal ATS data: {len(ats_grouped):,} invoices")
print(f"Original Invoice data: {len(invoice_grouped):,} invoices")

OUTLIER_THRESHOLD = 1_000_000
ats_clean = ats_grouped[ats_grouped['discount_amount'] <= OUTLIER_THRESHOLD].copy()
invoice_clean = invoice_grouped[invoice_grouped['discount_amount'] <= OUTLIER_THRESHOLD].copy()

print(f"\nCleaned ATS: {len(ats_clean):,} invoices")
print(f"Cleaned Invoice: {len(invoice_clean):,} invoices")

ats_clean['customer_type'] = 'ATS'
invoice_clean['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_clean, invoice_clean], ignore_index=True)

# ================================================================
# Robust date parsing
# ================================================================
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

min_valid = pd.Timestamp("2023-11-01")
max_valid = pd.Timestamp("2035-12-31")
combined_df = combined_df[(combined_df['invoice_period'] >= min_valid) & 
                          (combined_df['invoice_period'] <= max_valid)].copy()
combined_df = combined_df.sort_values('invoice_period').reset_index(drop=True)

# Calculate invoice amounts (assuming discount_amount is based on discounted invoice amount)
# If you have actual invoice_amount column, use that instead
if 'invoice_amount' not in combined_df.columns:
    # We need to know the discount rate to back-calculate
    # For now, let's assume it's in the data or calculate from discount_amount/discounted_total
    # You may need to adjust this based on your actual data structure
    print("\nWarning: 'invoice_amount' column not found. Please specify how to calculate it.")
    print("For now, assuming discount represents 2% of undiscounted amount...")
    combined_df['invoice_amount'] = combined_df['discount_amount'] / 0.02
else:
    print("\nUsing 'invoice_amount' column from data")

combined_df['discounted_invoice_amount'] = combined_df['invoice_amount'] - combined_df['discount_amount']

# ================================================================
# SIMULATE DELINQUENCY SCENARIOS
# ================================================================
print(f"\n{'='*70}")
print(f"SIMULATING 2 INTEREST CALCULATION SCENARIOS")
print(f"{'='*70}")

scenario_results = {}
scenario_summaries = []

for interest_scenario in INTEREST_SCENARIOS:
    scenario_name = "Full Invoice Amount" if interest_scenario == 'full_amount' else "Discounted Amount Only"
    print(f"\n{'='*50}")
    print(f"Scenario: Interest on {scenario_name}")
    print(f"{'='*50}")
    
    df_scenario = combined_df.copy()
    np.random.seed(RANDOM_SEED)
    
    n_invoices = len(df_scenario)
    n_defaults = int(n_invoices * TOTAL_DEFAULT_RATE)
    
    # Split defaults into delinquency types proportionally
    seriously_delinquent_rate = SERIOUSLY_DELINQUENT_PROPORTION / TOTAL_DEFAULT_RATE
    n_seriously_delinquent = int(n_defaults * seriously_delinquent_rate)
    n_delinquent = n_defaults - n_seriously_delinquent
    
    print(f"  Total invoices: {n_invoices:,}")
    print(f"  Total defaults: {n_defaults:,} ({TOTAL_DEFAULT_RATE*100:.2f}%)")
    print(f"  - Delinquent (1 month): {n_delinquent:,} ({n_delinquent/n_invoices*100:.2f}%)")
    print(f"  - Seriously delinquent (3 months): {n_seriously_delinquent:,} ({n_seriously_delinquent/n_invoices*100:.2f}%)")
    
    # Randomly assign delinquency status
    default_indices = np.random.choice(df_scenario.index, size=n_defaults, replace=False)
    seriously_delinquent_indices = np.random.choice(default_indices, size=n_seriously_delinquent, replace=False)
    delinquent_indices = np.setdiff1d(default_indices, seriously_delinquent_indices)
    
    # Initialize columns
    df_scenario['paid_on_time'] = True
    df_scenario['delinquency_type'] = 'None'
    df_scenario['payment_delay_months'] = 0
    
    # Flag delinquent
    df_scenario.loc[delinquent_indices, 'paid_on_time'] = False
    df_scenario.loc[delinquent_indices, 'delinquency_type'] = 'Delinquent'
    df_scenario.loc[delinquent_indices, 'payment_delay_months'] = DELINQUENT_DELAY_MONTHS
    
    # Flag seriously delinquent
    df_scenario.loc[seriously_delinquent_indices, 'paid_on_time'] = False
    df_scenario.loc[seriously_delinquent_indices, 'delinquency_type'] = 'Seriously Delinquent'
    df_scenario.loc[seriously_delinquent_indices, 'payment_delay_months'] = SERIOUSLY_DELINQUENT_DELAY_MONTHS
    
    # Calculate discount retained (discount not given)
    df_scenario['discount_retained'] = df_scenario.apply(
        lambda row: row['discount_amount'] if not row['paid_on_time'] else 0, 
        axis=1
    )
    
    # Calculate late fees
    df_scenario['late_fees'] = df_scenario.apply(
        lambda row: LATE_FEE_PER_INVOICE if not row['paid_on_time'] else 0,
        axis=1
    )
    
    # Calculate interest earned based on scenario
    if interest_scenario == 'full_amount':
        # Interest on full undiscounted invoice amount
        df_scenario['interest_earned'] = df_scenario.apply(
            lambda row: 0 if row['paid_on_time'] else 
                       row['invoice_amount'] * MONTHLY_INTEREST_RATE * row['payment_delay_months'],
            axis=1
        )
    else:  # 'discounted_amount'
        # Interest on discounted invoice amount only
        df_scenario['interest_earned'] = df_scenario.apply(
            lambda row: 0 if row['paid_on_time'] else 
                       row['discounted_invoice_amount'] * MONTHLY_INTEREST_RATE * row['payment_delay_months'],
            axis=1
        )
    
    # Total revenue from canceling discount policy
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
    
    scenario_results[interest_scenario] = period_summary
    
    # Summary statistics
    total_discount_offered = period_summary['cumulative_discount_offered'].iloc[-1]
    total_discount_retained = period_summary['cumulative_discount_retained'].iloc[-1]
    total_interest_earned = period_summary['cumulative_interest_earned'].iloc[-1]
    total_late_fees = period_summary['cumulative_late_fees'].iloc[-1]
    total_revenue = period_summary['cumulative_total_revenue'].iloc[-1]
    
    scenario_summaries.append({
        'interest_scenario': interest_scenario,
        'scenario_name': scenario_name,
        'n_defaults': n_defaults,
        'n_delinquent': n_delinquent,
        'n_seriously_delinquent': n_seriously_delinquent,
        'total_discount_offered': total_discount_offered,
        'total_discount_retained': total_discount_retained,
        'total_interest_earned': total_interest_earned,
        'total_late_fees': total_late_fees,
        'total_revenue_impact': total_revenue
    })
    
    print(f"\n  Revenue summary:")
    print(f"    Discount retained: ${total_discount_retained:,.2f}")
    print(f"    Interest earned: ${total_interest_earned:,.2f}")
    print(f"    Late fees: ${total_late_fees:,.2f}")
    print(f"    Total revenue impact: ${total_revenue:,.2f}")

summary_df = pd.DataFrame(scenario_summaries)

# ================================================================
# Print Comparison Summary
# ================================================================
print(f"\n{'='*70}")
print(f"SCENARIO COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"Total invoices analyzed: {len(combined_df):,}")
print(f"Total default rate: {TOTAL_DEFAULT_RATE*100:.2f}%")
print(f"Total discount offered (if all pay early): ${summary_df['total_discount_offered'].iloc[0]:,.2f}")

for _, row in summary_df.iterrows():
    print(f"\n{'='*50}")
    print(f"Scenario: Interest on {row['scenario_name']}")
    print(f"{'='*50}")
    print(f"  Delinquent invoices: {row['n_delinquent']:,}")
    print(f"  Seriously delinquent invoices: {row['n_seriously_delinquent']:,}")
    print(f"\n  Revenue breakdown:")
    print(f"    Discount retained: ${row['total_discount_retained']:,.2f} ({row['total_discount_retained']/row['total_revenue_impact']*100:.1f}%)")
    print(f"    Interest earned: ${row['total_interest_earned']:,.2f} ({row['total_interest_earned']/row['total_revenue_impact']*100:.1f}%)")
    print(f"    Late fees: ${row['total_late_fees']:,.2f} ({row['total_late_fees']/row['total_revenue_impact']*100:.1f}%)")
    print(f"    TOTAL REVENUE: ${row['total_revenue_impact']:,.2f}")

# Calculate difference between scenarios
diff_interest = summary_df.iloc[0]['total_interest_earned'] - summary_df.iloc[1]['total_interest_earned']
diff_total = summary_df.iloc[0]['total_revenue_impact'] - summary_df.iloc[1]['total_revenue_impact']
print(f"\n{'='*50}")
print(f"DIFFERENCE BETWEEN SCENARIOS")
print(f"{'='*50}")
print(f"  Additional interest (full vs discounted): ${diff_interest:,.2f}")
print(f"  Additional total revenue (full vs discounted): ${diff_total:,.2f}")

# ================================================================
# Annual Analysis
# ================================================================
print(f"\n{'='*70}")
print(f"ANNUAL REVENUE ANALYSIS")
print(f"{'='*70}")

end_of_2023 = pd.Timestamp("2023-12-31")
end_of_2024 = pd.Timestamp("2024-12-31")
end_of_2025 = pd.Timestamp("2025-12-31")

annual_results = {2023: [], 2024: [], 2025: []}

for interest_scenario in INTEREST_SCENARIOS:
    period_data = scenario_results[interest_scenario]
    scenario_name = "Full Invoice Amount" if interest_scenario == 'full_amount' else "Discounted Amount Only"
    
    for year, end_date in [(2023, end_of_2023), (2024, end_of_2024), (2025, end_of_2025)]:
        mask_year = period_data['invoice_period'] <= end_date
        
        if mask_year.any():
            data_year = period_data[mask_year]
            discount_retained = data_year['cumulative_discount_retained'].iloc[-1]
            interest_earned = data_year['cumulative_interest_earned'].iloc[-1]
            late_fees = data_year['cumulative_late_fees'].iloc[-1]
            total_revenue = data_year['cumulative_total_revenue'].iloc[-1]
            last_period_year = data_year['invoice_period'].iloc[-1]
            
            annual_results[year].append({
                'scenario': scenario_name,
                'last_period': last_period_year.strftime('%Y-%m'),
                'discount_retained': discount_retained,
                'interest_earned': interest_earned,
                'late_fees': late_fees,
                'total_revenue': total_revenue,
                'net_after_implementation': total_revenue - 20_000
            })

# Print annual results
for year in [2023, 2024, 2025]:
    print(f"\n{'-'*70}")
    print(f"END OF {year} RESULTS")
    print(f"{'-'*70}")
    
    if annual_results[year]:
        for result in annual_results[year]:
            print(f"\nScenario: Interest on {result['scenario']}")
            print(f"  Last period: {result['last_period']}")
            print(f"  Discount retained: ${result['discount_retained']:,.2f}")
            print(f"  Interest earned: ${result['interest_earned']:,.2f}")
            print(f"  Late fees: ${result['late_fees']:,.2f}")
            print(f"  Total revenue: ${result['total_revenue']:,.2f}")
            print(f"  Net after $20K implementation: ${result['net_after_implementation']:,.2f}")
    else:
        print(f"\n✗ No data available for {year}")

# ================================================================
# Breakeven and Cost Recovery Analysis
# ================================================================
BREAKEVEN_DISCOUNT_RETAINED = 1_043_000
COST_OF_IMPLEMENTATION = 20_000

print(f"\n{'='*70}")
print(f"$20,000 IMPLEMENTATION COST RECOVERY ANALYSIS")
print(f"{'='*70}")
print(f"Implementation cost: ${COST_OF_IMPLEMENTATION:,}")

for interest_scenario in INTEREST_SCENARIOS:
    period_data = scenario_results[interest_scenario]
    scenario_name = "Full Invoice Amount" if interest_scenario == 'full_amount' else "Discounted Amount Only"
    
    print(f"\n{'='*50}")
    print(f"Scenario: Interest on {scenario_name}")
    print(f"{'='*50}")
    
    # Find when total revenue crosses implementation cost
    recovery_mask = period_data['cumulative_total_revenue'] >= COST_OF_IMPLEMENTATION
    
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        recovery_date = period_data.loc[recovery_idx, 'invoice_period']
        start_date = period_data['invoice_period'].min()
        
        months_to_recovery = (recovery_date.year - start_date.year) * 12 + (recovery_date.month - start_date.month)
        days_to_recovery = (recovery_date - start_date).days
        amount_at_recovery = period_data.loc[recovery_idx, 'cumulative_total_revenue']
        
        print(f"  ✓ Implementation cost RECOVERED")
        print(f"  Recovery date: {recovery_date.strftime('%Y-%m')}")
        print(f"  Time to recover: {months_to_recovery} months ({days_to_recovery} days)")
        print(f"  Total revenue at recovery: ${amount_at_recovery:,.2f}")
    else:
        print(f"  ✗ Implementation cost NOT recovered")
        print(f"  Maximum revenue: ${period_data['cumulative_total_revenue'].max():,.2f}")

print(f"\n{'='*70}")
print(f"BREAKEVEN ANALYSIS (Target: ${BREAKEVEN_DISCOUNT_RETAINED:,})")
print(f"{'='*70}")

for interest_scenario in INTEREST_SCENARIOS:
    period_data = scenario_results[interest_scenario]
    scenario_name = "Full Invoice Amount" if interest_scenario == 'full_amount' else "Discounted Amount Only"
    total_revenue = period_data['cumulative_total_revenue'].iloc[-1]
    
    print(f"\nScenario: Interest on {scenario_name}")
    
    gap = total_revenue - BREAKEVEN_DISCOUNT_RETAINED
    status = '✓ REACHED' if total_revenue >= BREAKEVEN_DISCOUNT_RETAINED else '✗ NOT REACHED'
    print(f"  Status: {status}")
    print(f"  Total revenue: ${total_revenue:,.2f}")
    print(f"  Gap to breakeven: ${gap:+,.2f}")
    
    if total_revenue >= BREAKEVEN_DISCOUNT_RETAINED:
        breakeven_mask = period_data['cumulative_total_revenue'] >= BREAKEVEN_DISCOUNT_RETAINED
        
        if breakeven_mask.any():
            breakeven_idx = breakeven_mask.idxmax()
            breakeven_date = period_data.loc[breakeven_idx, 'invoice_period']
            start_date = period_data['invoice_period'].min()
            
            months_to_breakeven = (breakeven_date.year - start_date.year) * 12 + (breakeven_date.month - start_date.month)
            print(f"  Breakeven reached: {breakeven_date.strftime('%Y-%m')} ({months_to_breakeven} months)")

# ================================================================
# Create Comparison Visualization
# ================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))

# Color scheme
colors = {'full_amount': '#4472C4', 'discounted_amount': '#70AD47'}
reference_period = scenario_results['full_amount']
mark_every = max(1, len(reference_period)//20)

# Plot 1: Total Revenue Comparison
for interest_scenario in INTEREST_SCENARIOS:
    period_data = scenario_results[interest_scenario]
    scenario_name = "Full Invoice Amount" if interest_scenario == 'full_amount' else "Discounted Amount Only"
    
    ax1.plot(period_data['invoice_period'], period_data['cumulative_total_revenue'],
            color=colors[interest_scenario], linewidth=2.5,
            label=f"Total revenue ({scenario_name})",
            marker='o', markersize=4, markevery=mark_every)

# Add reference lines
ax1.plot(reference_period['invoice_period'], reference_period['cumulative_discount_offered'],
        color="#222222", linewidth=1.5, alpha=0.6,
        label='Total discount offered (if all pay early)',
        marker='o', markersize=3, markevery=mark_every, linestyle='--')

ax1.axhline(y=BREAKEVEN_DISCOUNT_RETAINED, color='#FF0000', linewidth=1.5,
           linestyle='-', alpha=0.8, label='Breakeven target ($1,043,000)', zorder=10)

ax1.axhline(y=COST_OF_IMPLEMENTATION, color="#FF9100", linewidth=1.5,
           linestyle='-', alpha=0.8, label='Implementation cost ($20,000)', zorder=10)

# Year end lines
ax1.axvline(x=end_of_2023, color='#8B7BA8', linewidth=1.5, linestyle=':', alpha=0.7, label='End of 2023', zorder=9)
ax1.axvline(x=end_of_2024, color='#8B7BA8', linewidth=1.5, linestyle=':', alpha=0.7, label='End of 2024', zorder=9)
ax1.axvline(x=end_of_2025, color='#8B7BA8', linewidth=1.5, linestyle=':', alpha=0.7, label='End of 2025', zorder=9)

# Styling for plot 1
start_date = pd.Timestamp("2023-10-01")
ax1.set_xlim(left=start_date)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.tick_params(axis='x', rotation=45)
ax1.set_xlabel('Invoice Period (Year-Month)', fontsize=13, fontweight='bold')
ax1.set_ylabel('$ Revenue Amount', fontsize=13, fontweight='bold')
ax1.set_title('Cumulative Revenue Impact: Interest Calculation Comparison',
             fontsize=16, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.set_facecolor('#F8F8F8')

# Plot 2: Revenue Component Breakdown (Stacked Area)
for interest_scenario in INTEREST_SCENARIOS:
    period_data = scenario_results[interest_scenario]
    scenario_name = "Full" if interest_scenario == 'full_amount' else "Discounted"
    
    ax2.plot(period_data['invoice_period'], period_data['cumulative_discount_retained'],
            color=colors[interest_scenario], linewidth=2, alpha=0.8,
            label=f"{scenario_name}: Discount retained", linestyle='-')
    
    ax2.plot(period_data['invoice_period'], period_data['cumulative_interest_earned'],
            color=colors[interest_scenario], linewidth=2, alpha=0.6,
            label=f"{scenario_name}: Interest earned", linestyle='--')
    
    ax2.plot(period_data['invoice_period'], period_data['cumulative_late_fees'],
            color=colors[interest_scenario], linewidth=1.5, alpha=0.5,
            label=f"{scenario_name}: Late fees", linestyle=':')

# Styling for plot 2
ax2.set_xlim(left=start_date)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.tick_params(axis='x', rotation=45)
ax2.set_xlabel('Invoice Period (Year-Month)', fontsize=13, fontweight='bold')
ax2.set_ylabel('$ Amount', fontsize=13, fontweight='bold')
ax2.set_title('Revenue Components Breakdown by Scenario',
             fontsize=16, fontweight='bold', pad=15)
ax2.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.set_facecolor('#F8F8F8')

plt.tight_layout()
plt.savefig('discount_retention_interest_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization to: discount_retention_interest_comparison.png")
plt.show()

# ================================================================
# Save detailed results
# ================================================================
# Enhance summary with additional metrics
summary_df['discount_pct_of_revenue'] = (summary_df['total_discount_retained'] / summary_df['total_revenue_impact'] * 100).round(2)
summary_df['interest_pct_of_revenue'] = (summary_df['total_interest_earned'] / summary_df['total_revenue_impact'] * 100).round(2)
summary_df['late_fees_pct_of_revenue'] = (summary_df['total_late_fees'] / summary_df['total_revenue_impact'] * 100).round(2)
summary_df['breakeven_gap'] = summary_df['total_revenue_impact'] - BREAKEVEN_DISCOUNT_RETAINED
summary_df['reaches_breakeven'] = summary_df['total_revenue_impact'] >= BREAKEVEN_DISCOUNT_RETAINED
summary_df['implementation_cost_recovered'] = summary_df['total_revenue_impact'] >= COST_OF_IMPLEMENTATION

# Add annual results to summary
for year in [2023, 2024, 2025]:
    revenue_col = f'revenue_by_end_{year}'
    net_col = f'net_after_implementation_{year}'
    
    year_revenues = []
    year_nets = []
    
    for interest_scenario in INTEREST_SCENARIOS:
        period_data = scenario_results[interest_scenario]
        end_date = pd.Timestamp(f"{year}-12-31")
        mask_year = period_data['invoice_period'] <= end_date
        
        if mask_year.any():
            data_year = period_data[mask_year]
            revenue = data_year['cumulative_total_revenue'].iloc[-1]
            year_revenues.append(revenue)
            year_nets.append(revenue - COST_OF_IMPLEMENTATION)
        else:
            year_revenues.append(None)
            year_nets.append(None)
    
    summary_df[revenue_col] = year_revenues
    summary_df[net_col] = year_nets

summary_df.to_csv('interest_scenario_comparison_summary.csv', index=False)
print(f"✓ Saved scenario comparison to: interest_scenario_comparison_summary.csv")

# Save detailed period-by-period results
with pd.ExcelWriter('interest_scenario_detailed_results.xlsx', engine='openpyxl') as writer:
    for interest_scenario, period_data in scenario_results.items():
        scenario_name = "Full_Amount" if interest_scenario == 'full_amount' else "Discounted_Amount"
        period_data.to_excel(writer, sheet_name=scenario_name, index=False)
    
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"✓ Saved detailed period results to: interest_scenario_detailed_results.xlsx")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")