import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================================================================
# CONFIGURATION - Set your default rates here
# ================================================================
DEFAULT_RATES = [0.01, 0.05, 0.10]  # Add as many rates as you want to test
RANDOM_SEED = 42  # Set seed for reproducibility (change or remove for different random selections)

# ================================================================
# Load and clean data
# ================================================================
ats_grouped = pd.read_csv('ats_grouped_with_discounts.csv')
invoice_grouped = pd.read_csv('invoice_grouped_with_discounts.csv')

print(f"Original ATS data: {len(ats_grouped):,} invoices")
print(f"Original Invoice data: {len(invoice_grouped):,} invoices")

# Filter outliers
OUTLIER_THRESHOLD = 1_000_000
ats_clean = ats_grouped[ats_grouped['discount_amount'] <= OUTLIER_THRESHOLD].copy()
invoice_clean = invoice_grouped[invoice_grouped['discount_amount'] <= OUTLIER_THRESHOLD].copy()

print(f"\nCleaned ATS: {len(ats_clean):,} invoices")
print(f"Cleaned Invoice: {len(invoice_clean):,} invoices")

# Combine
ats_clean['customer_type'] = 'ATS'
invoice_clean['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_clean, invoice_clean], ignore_index=True)

# ================================================================
# Robust date parsing to avoid "1970" junk
# ================================================================
def parse_invoice_period(series: pd.Series) -> pd.Series:
    """
    Robustly parse invoice_period.
    Handles:
      - 'YYYY-MM' / 'YYYY-MM-DD'
      - 'YYYYMM' (e.g., 202312)
      - avoids numeric-to-epoch misparse that often shows up as 1970
    """
    s = series.copy()

    # Convert to string first (prevents pandas treating ints as nanoseconds since epoch)
    s_str = s.astype(str).str.strip()

    # Treat common null-like strings as missing
    s_str = s_str.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})

    # Case 1: looks like YYYYMM (6 digits)
    mask_yyyymm = s_str.str.fullmatch(r"\d{6}", na=False)
    out = pd.Series(pd.NaT, index=s.index)

    if mask_yyyymm.any():
        out.loc[mask_yyyymm] = pd.to_datetime(s_str.loc[mask_yyyymm], format="%Y%m", errors="coerce")

    # Case 2: everything else -> let pandas parse (YYYY-MM, YYYY-MM-DD, etc.)
    mask_other = ~mask_yyyymm
    if mask_other.any():
        out.loc[mask_other] = pd.to_datetime(s_str.loc[mask_other], errors="coerce")

    return out

combined_df['invoice_period'] = parse_invoice_period(combined_df['invoice_period'])

# Drop invalid dates
combined_df = combined_df[combined_df['invoice_period'].notna()].copy()

# Hard filter to remove obvious epoch-era junk like 1970-xx-xx
min_valid = pd.Timestamp("2023-11-01")
max_valid = pd.Timestamp("2035-12-31")
combined_df = combined_df[(combined_df['invoice_period'] >= min_valid) & (combined_df['invoice_period'] <= max_valid)].copy()

combined_df = combined_df.sort_values('invoice_period').reset_index(drop=True)

# ================================================================
# SIMULATE MULTIPLE DEFAULT RATE SCENARIOS
# ================================================================
print(f"\n{'='*70}")
print(f"SIMULATING {len(DEFAULT_RATES)} DEFAULT RATE SCENARIOS")
print(f"{'='*70}")

# Store results for each scenario
scenario_results = {}
scenario_summaries = []

for default_rate in DEFAULT_RATES:
    print(f"\nProcessing {default_rate*100:.1f}% default rate...")
    
    # Create a copy for this scenario
    df_scenario = combined_df.copy()
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Randomly select invoices that don't pay on time
    n_invoices = len(df_scenario)
    n_defaults = int(n_invoices * default_rate)
    
    default_indices = np.random.choice(df_scenario.index, size=n_defaults, replace=False)
    df_scenario['paid_on_time'] = True
    df_scenario.loc[default_indices, 'paid_on_time'] = False
    
    # Calculate discount retained
    df_scenario['discount_retained'] = df_scenario.apply(
        lambda row: row['discount_amount'] if not row['paid_on_time'] else 0, 
        axis=1
    )
    
    # Group by period
    period_summary = df_scenario.groupby('invoice_period').agg({
        'discount_amount': 'sum',
        'discount_retained': 'sum',
        'paid_on_time': ['sum', 'count']
    }).reset_index()
    
    period_summary.columns = ['invoice_period', 'total_discount_offered', 'discount_retained', 
                              'n_paid_on_time', 'n_total_invoices']
    
    # Calculate cumulative values
    period_summary['cumulative_discount_offered'] = period_summary['total_discount_offered'].cumsum()
    period_summary['cumulative_discount_retained'] = period_summary['discount_retained'].cumsum()
    
    # Store results
    scenario_results[default_rate] = period_summary
    
    # Store summary statistics
    scenario_summaries.append({
        'default_rate': default_rate,
        'n_defaults': n_defaults,
        'total_discount_offered': period_summary['cumulative_discount_offered'].iloc[-1],
        'total_discount_retained': period_summary['cumulative_discount_retained'].iloc[-1]
    })

# Convert summaries to DataFrame
summary_df = pd.DataFrame(scenario_summaries)

# ================================================================
# Print Summary Statistics
# ================================================================
print(f"\n{'='*70}")
print(f"SCENARIO COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"Total invoices analyzed: {len(combined_df):,}")
print(f"Total discount offered (if all pay early): ${summary_df['total_discount_offered'].iloc[0]:,.2f}")

for _, row in summary_df.iterrows():
    rate = row['default_rate']
    retained = row['total_discount_retained']
    offered = row['total_discount_offered']
    print(f"\n{rate*100:.1f}% Default Rate:")
    print(f"  Invoices not paid on time: {row['n_defaults']:,}")
    print(f"  Discount retained: ${retained:,.2f} ({retained/offered*100:.2f}% of offered)")

# ================================================================
# Annual Analysis (2023, 2024, 2025)
# ================================================================
print(f"\n{'='*70}")
print(f"ANNUAL RETENTION ANALYSIS")
print(f"{'='*70}")

# Define end of years
end_of_2023 = pd.Timestamp("2023-12-31")
end_of_2024 = pd.Timestamp("2024-12-31")
end_of_2025 = pd.Timestamp("2025-12-31")

annual_results = {2023: [], 2024: [], 2025: []}

for default_rate in DEFAULT_RATES:
    period_data = scenario_results[default_rate]
    
    # For each year
    for year, end_date in [(2023, end_of_2023), (2024, end_of_2024), (2025, end_of_2025)]:
        mask_year = period_data['invoice_period'] <= end_date
        
        if mask_year.any():
            data_year = period_data[mask_year]
            retained_year = data_year['cumulative_discount_retained'].iloc[-1]
            last_period_year = data_year['invoice_period'].iloc[-1]
            
            annual_results[year].append({
                'default_rate': default_rate,
                'last_period': last_period_year.strftime('%Y-%m'),
                'retained_by_end': retained_year,
                'net_after_implementation': retained_year - 20_000
            })

# Print annual results
for year in [2023, 2024, 2025]:
    print(f"\n{'-'*70}")
    print(f"END OF {year} RESULTS")
    print(f"{'-'*70}")
    
    if annual_results[year]:
        for result in annual_results[year]:
            print(f"\n{result['default_rate']*100:.1f}% Default Rate:")
            print(f"  Last period in {year}: {result['last_period']}")
            print(f"  Discount retained by end of {year}: ${result['retained_by_end']:,.2f}")
            print(f"  Net after implementation cost: ${result['net_after_implementation']:,.2f}")
    else:
        print(f"\n✗ No data available for {year}")

# ================================================================
# Breakeven and Cost Analysis
# ================================================================
BREAKEVEN_DISCOUNT_RETAINED = 1_043_000
COST_OF_IMPLEMENTATION = 20_000

print(f"\n{'='*70}")
print(f"$20,000 IMPLEMENTATION COST RECOVERY ANALYSIS")
print(f"{'='*70}")
print(f"Implementation cost: ${COST_OF_IMPLEMENTATION:,}")

# Calculate time to recover implementation cost
implementation_recovery_data = []

for _, row in summary_df.iterrows():
    rate = row['default_rate']
    period_data = scenario_results[rate]
    
    # Find when cumulative retained crosses implementation cost
    recovery_mask = period_data['cumulative_discount_retained'] >= COST_OF_IMPLEMENTATION
    
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        recovery_date = period_data.loc[recovery_idx, 'invoice_period']
        start_date = period_data['invoice_period'].min()
        
        # Calculate months to recovery
        months_to_recovery = (recovery_date.year - start_date.year) * 12 + (recovery_date.month - start_date.month)
        
        # Calculate days to recovery (more precise)
        days_to_recovery = (recovery_date - start_date).days
        
        amount_at_recovery = period_data.loc[recovery_idx, 'cumulative_discount_retained']
        
        implementation_recovery_data.append({
            'default_rate': rate,
            'recovery_date': recovery_date,
            'months_to_recovery': months_to_recovery,
            'days_to_recovery': days_to_recovery,
            'amount_at_recovery': amount_at_recovery
        })
        
        print(f"\n{rate*100:.1f}% Default Rate:")
        print(f"  Recovery date: {recovery_date.strftime('%Y-%m')}")
        print(f"  Time to recover $20K: {months_to_recovery} months ({days_to_recovery} days)")
        print(f"  Amount retained at recovery: ${amount_at_recovery:,.2f}")
    else:
        print(f"\n{rate*100:.1f}% Default Rate:")
        print(f"  ✗ Implementation cost NOT recovered within data period")
        print(f"  Maximum retained: ${period_data['cumulative_discount_retained'].max():,.2f}")

print(f"\n{'='*70}")
print(f"BREAKEVEN ANALYSIS (Target: ${BREAKEVEN_DISCOUNT_RETAINED:,})")
print(f"{'='*70}")

for _, row in summary_df.iterrows():
    rate = row['default_rate']
    retained = row['total_discount_retained']
    gap = retained - BREAKEVEN_DISCOUNT_RETAINED
    status = '✓ REACHED' if retained >= BREAKEVEN_DISCOUNT_RETAINED else '✗ NOT REACHED'
    print(f"{rate*100:.1f}% default: {status} (${gap:+,.2f})")
    
    # Calculate time to breakeven if reached
    if retained >= BREAKEVEN_DISCOUNT_RETAINED:
        period_data = scenario_results[rate]
        breakeven_mask = period_data['cumulative_discount_retained'] >= BREAKEVEN_DISCOUNT_RETAINED
        
        if breakeven_mask.any():
            breakeven_idx = breakeven_mask.idxmax()
            breakeven_date = period_data.loc[breakeven_idx, 'invoice_period']
            start_date = period_data['invoice_period'].min()
            
            months_to_breakeven = (breakeven_date.year - start_date.year) * 12 + (breakeven_date.month - start_date.month)
            print(f"  Breakeven reached in: {months_to_breakeven} months ({breakeven_date.strftime('%Y-%m')})")

# Calculate required default rate for breakeven
total_offered = summary_df['total_discount_offered'].iloc[0]
required_rate = BREAKEVEN_DISCOUNT_RETAINED / total_offered
print(f"\nDefault rate needed to break even: {required_rate*100:.2f}%")

# Calculate required rate for implementation cost recovery
required_rate_impl = COST_OF_IMPLEMENTATION / total_offered
print(f"Default rate needed to recover implementation cost: {required_rate_impl*100:.2f}%")

# ================================================================
# Create visualization with multiple scenarios
# ================================================================
fig, ax = plt.subplots(figsize=(18, 10))

# Color palette for different scenarios
colors = ['#70AD47', '#A64D79', '#4472C4', '#FFC000', '#FF6B6B', '#845EC2']
markers = ['^', 's', 'o', 'D', 'v', 'P']

# Get a reference period summary for discount offered line
reference_period = scenario_results[DEFAULT_RATES[0]]
mark_every = max(1, len(reference_period)//20)

# Plot cumulative discount offered (reference line - same for all scenarios)
ax.plot(reference_period['invoice_period'], reference_period['cumulative_discount_offered'],
        color="#222222", linewidth=1.5, alpha=0.6,
        label='Total discount offered (if all pay early)',
        marker='o', markersize=3, markevery=mark_every, linestyle='--')

# Plot each default rate scenario with thinner lines
for i, default_rate in enumerate(DEFAULT_RATES):
    period_data = scenario_results[default_rate]
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    
    ax.plot(period_data['invoice_period'], period_data['cumulative_discount_retained'],
            color=color, linewidth=2,
            label=f"Discount retained ({default_rate*100:.1f}% default rate)",
            marker=marker, markersize=4, markevery=mark_every)

# Breakeven line at exactly $1,043,000
ax.axhline(y=BREAKEVEN_DISCOUNT_RETAINED, color='#FF0000', linewidth=1.5,
           linestyle='-', alpha=0.8, label='Current Card interest amount ($1,043,000)', zorder=10)

# Cost of implementation est. at $20K
ax.axhline(y=COST_OF_IMPLEMENTATION, color="#FF9100", linewidth=1.5,
           linestyle='-', alpha=0.8, label='Estimated cost of implementation ($20,000)', zorder=10)

# Add vertical lines for end of years
ax.axvline(x=end_of_2023, color='#8B7BA8', linewidth=1.5,
           linestyle=':', alpha=0.7, label='End of 2023', zorder=9)

ax.axvline(x=end_of_2024, color='#8B7BA8', linewidth=1.5,
           linestyle=':', alpha=0.7, label='End of 2024', zorder=9)

ax.axvline(x=end_of_2025, color='#8B7BA8', linewidth=1.5,
           linestyle=':', alpha=0.7, label='End of 2025', zorder=9)

# Format x-axis density
date_range_days = (reference_period['invoice_period'].max() - reference_period['invoice_period'].min()).days
if date_range_days > 730:
    interval = 3
elif date_range_days > 365:
    interval = 2
else:
    interval = 1


# Set x-axis limits to start at October 2023
start_date = pd.Timestamp("2023-10-01")
ax.set_xlim(left=start_date)

# Then continue with the existing x-axis formatting code:
# Format x-axis density
date_range_days = (reference_period['invoice_period'].max() - reference_period['invoice_period'].min()).days
if date_range_days > 730:
    interval = 3
elif date_range_days > 365:
    interval = 2
else:
    interval = 1

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
plt.xticks(rotation=45, ha='right')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
plt.xticks(rotation=45, ha='right')

# Styling
ax.set_xlabel('Invoice Period (Year-Month)', fontsize=15, fontweight='bold')
ax.set_ylabel('$ Discount Amount', fontsize=15, fontweight='bold')
ax.set_title('Cumulative Discount Retention Analysis: Multiple Default Rate Scenarios',
             fontsize=20, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_facecolor('#F8F8F8')
plt.tight_layout()

plt.savefig('discount_retention_multi_scenario.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved multi-scenario visualization to: discount_retention_multi_scenario.png")
plt.show()

# ================================================================
# Save detailed results
# ================================================================
# Add implementation cost recovery analysis to summary
summary_df['discount_given_away'] = summary_df['total_discount_offered'] - summary_df['total_discount_retained']
summary_df['retention_rate_pct'] = (summary_df['total_discount_retained'] / summary_df['total_discount_offered'] * 100)
summary_df['breakeven_gap'] = summary_df['total_discount_retained'] - BREAKEVEN_DISCOUNT_RETAINED
summary_df['reaches_breakeven'] = summary_df['total_discount_retained'] >= BREAKEVEN_DISCOUNT_RETAINED
summary_df['implementation_cost_recovered'] = summary_df['total_discount_retained'] >= COST_OF_IMPLEMENTATION

# Calculate months to recover for each scenario and annual retention
recovery_months = []
recovery_days = []
breakeven_months = []
retained_by_2023 = []
retained_by_2024 = []
retained_by_2025 = []

for default_rate in DEFAULT_RATES:
    period_data = scenario_results[default_rate]
    start_date = period_data['invoice_period'].min()
    
    # Implementation cost recovery
    recovery_mask = period_data['cumulative_discount_retained'] >= COST_OF_IMPLEMENTATION
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        recovery_date = period_data.loc[recovery_idx, 'invoice_period']
        months = (recovery_date.year - start_date.year) * 12 + (recovery_date.month - start_date.month)
        days = (recovery_date - start_date).days
        recovery_months.append(months)
        recovery_days.append(days)
    else:
        recovery_months.append(None)
        recovery_days.append(None)
    
    # Breakeven
    breakeven_mask = period_data['cumulative_discount_retained'] >= BREAKEVEN_DISCOUNT_RETAINED
    if breakeven_mask.any():
        breakeven_idx = breakeven_mask.idxmax()
        breakeven_date = period_data.loc[breakeven_idx, 'invoice_period']
        months = (breakeven_date.year - start_date.year) * 12 + (breakeven_date.month - start_date.month)
        breakeven_months.append(months)
    else:
        breakeven_months.append(None)
    
    # Amount retained by end of each year
    for year, year_list in [(2023, retained_by_2023), (2024, retained_by_2024), (2025, retained_by_2025)]:
        end_date = pd.Timestamp(f"{year}-12-31")
        mask_year = period_data['invoice_period'] <= end_date
        if mask_year.any():
            data_year = period_data[mask_year]
            year_list.append(data_year['cumulative_discount_retained'].iloc[-1])
        else:
            year_list.append(None)

summary_df['months_to_recover_implementation'] = recovery_months
summary_df['days_to_recover_implementation'] = recovery_days
summary_df['months_to_breakeven'] = breakeven_months
summary_df['retained_by_end_2023'] = retained_by_2023
summary_df['retained_by_end_2024'] = retained_by_2024
summary_df['retained_by_end_2025'] = retained_by_2025
summary_df['net_after_implementation_2023'] = summary_df['retained_by_end_2023'] - COST_OF_IMPLEMENTATION
summary_df['net_after_implementation_2024'] = summary_df['retained_by_end_2024'] - COST_OF_IMPLEMENTATION
summary_df['net_after_implementation_2025'] = summary_df['retained_by_end_2025'] - COST_OF_IMPLEMENTATION

summary_df.to_csv('scenario_comparison_summary.csv', index=False)
print(f"✓ Saved scenario comparison to: scenario_comparison_summary.csv")

# Optionally save detailed period-by-period results for each scenario
with pd.ExcelWriter('scenario_detailed_results.xlsx', engine='openpyxl') as writer:
    for default_rate, period_data in scenario_results.items():
        sheet_name = f"{default_rate*100:.1f}pct_default"
        period_data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Also add summary sheet
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print(f"✓ Saved detailed period results to: scenario_detailed_results.xlsx")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")