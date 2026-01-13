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

# ================================================================
# Load and clean data
# ================================================================
print("="*70)
print("FINDING DEFAULT RATE TO REACH BREAKEVEN BY END OF FY2025")
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

# Filter to only include data up to end of FY2025
combined_df_fy2025 = combined_df[combined_df['invoice_period'] <= TARGET_DATE].copy()
print(f"\nInvoices through end of FY2025 (March 31, 2025): {len(combined_df_fy2025):,}")

# ================================================================
# Function to calculate retention for a given default rate
# ================================================================
def calculate_retention_at_date(df, default_rate, target_date, seed=RANDOM_SEED):
    """
    Calculate cumulative discount retained by a target date for a given default rate.
    
    Returns: (retained_amount, period_summary_df)
    """
    df_scenario = df.copy()
    
    np.random.seed(seed)
    
    n_invoices = len(df_scenario)
    n_defaults = int(n_invoices * default_rate)
    
    default_indices = np.random.choice(df_scenario.index, size=n_defaults, replace=False)
    df_scenario['paid_on_time'] = True
    df_scenario.loc[default_indices, 'paid_on_time'] = False
    
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
    
    period_summary['cumulative_discount_offered'] = period_summary['total_discount_offered'].cumsum()
    period_summary['cumulative_discount_retained'] = period_summary['discount_retained'].cumsum()
    
    # Get amount retained by target date
    mask_target = period_summary['invoice_period'] <= target_date
    if mask_target.any():
        retained_by_target = period_summary[mask_target]['cumulative_discount_retained'].iloc[-1]
    else:
        retained_by_target = 0
    
    return retained_by_target, period_summary

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
best_retained = None
best_period_summary = None
iteration = 0

search_history = []

while iteration < MAX_ITERATIONS:
    iteration += 1
    mid_rate = (low_rate + high_rate) / 2
    
    retained, period_summary = calculate_retention_at_date(
        combined_df_fy2025, mid_rate, TARGET_DATE
    )
    
    gap = retained - BREAKEVEN_TARGET
    
    search_history.append({
        'iteration': iteration,
        'default_rate': mid_rate,
        'retained': retained,
        'gap': gap
    })
    
    print(f"Iteration {iteration:2d}: Rate={mid_rate*100:6.3f}% → Retained=${retained:>12,.2f} (Gap: ${gap:>10,.2f})")
    
    # Check if we're within tolerance
    if abs(gap) <= TOLERANCE:
        best_rate = mid_rate
        best_retained = retained
        best_period_summary = period_summary
        print(f"\n✓ FOUND! Default rate of {best_rate*100:.3f}% achieves breakeven within tolerance")
        break
    
    # Adjust search bounds
    if retained < BREAKEVEN_TARGET:
        # Need more defaults (higher rate)
        low_rate = mid_rate
    else:
        # Too many defaults (lower rate)
        high_rate = mid_rate
    
    # Store best attempt
    if best_retained is None or abs(gap) < abs(best_retained - BREAKEVEN_TARGET):
        best_rate = mid_rate
        best_retained = retained
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
print(f"Discount retained by end of FY2025: ${best_retained:,.2f}")
print(f"Target (breakeven): ${BREAKEVEN_TARGET:,.2f}")
print(f"Difference: ${best_retained - BREAKEVEN_TARGET:+,.2f}")
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
    retained, period_summary = calculate_retention_at_date(
        combined_df_fy2025, rate, TARGET_DATE
    )
    
    comparison_results.append({
        'default_rate': rate,
        'retained_fy2025': retained,
        'gap_from_breakeven': retained - BREAKEVEN_TARGET,
        'reaches_breakeven': retained >= BREAKEVEN_TARGET,
        'n_defaults': int(len(combined_df_fy2025) * rate)
    })
    
    scenario_data[rate] = period_summary

comparison_df = pd.DataFrame(comparison_results)

for _, row in comparison_df.iterrows():
    status = "✓ BREAKS EVEN" if row['reaches_breakeven'] else "✗ Below target"
    print(f"\n{row['default_rate']*100:5.2f}% default: {status}")
    print(f"  Retained: ${row['retained_fy2025']:,.2f}")
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

# Top plot: Cumulative retention over time
start_date = pd.Timestamp("2023-10-01")

for i, rate in enumerate([0.10, 0.15, 0.20, best_rate, 0.30, 0.35]):
    if rate in scenario_data:
        period_data = scenario_data[rate]
        color = colors[i % len(colors)]
        linewidth = 3 if rate == best_rate else 2
        alpha = 1.0 if rate == best_rate else 0.7
        
        label = f"{rate*100:.1f}% default" + (" (Breakeven rate)" if rate == best_rate else "")
        
        mark_every = max(1, len(period_data)//15)
        ax1.plot(period_data['invoice_period'], period_data['cumulative_discount_retained'],
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
ax1.set_ylabel('$ Discount Amount', fontsize=13, fontweight='bold')
ax1.set_title('Cumulative Discount Retention: Finding Breakeven Rate by End of FY2025',
             fontsize=18, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.set_facecolor('#F8F8F8')

# Bottom plot: Default rate vs. retained amount
retention_amounts = [result['retained_fy2025'] for result in comparison_results]
default_rates_pct = [result['default_rate']*100 for result in comparison_results]

ax2.plot(default_rates_pct, retention_amounts, 
         color='#4472C4', linewidth=3, marker='o', markersize=8)

# Mark the optimal point
ax2.scatter([best_rate*100], [best_retained], 
           color='#FF0000', s=200, marker='*', zorder=10,
           label=f'Breakeven rate: {best_rate*100:.2f}%', edgecolors='black', linewidths=2)

# Breakeven line
ax2.axhline(y=BREAKEVEN_TARGET, color='#FF0000', linewidth=2,
           linestyle='--', alpha=0.6, label=f'Breakeven target (${BREAKEVEN_TARGET:,})')

ax2.set_xlabel('Default Rate (%)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Discount Retained by End of FY2025 ($)', fontsize=13, fontweight='bold')
ax2.set_title('Default Rate vs. Discount Retained (Through FY2025)',
             fontsize=18, fontweight='bold', pad=15)
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
ax2.set_facecolor('#F8F8F8')

plt.tight_layout()
plt.savefig('breakeven_rate_fy2025_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to: breakeven_rate_fy2025_analysis.png")

# ================================================================
# Save results
# ================================================================
comparison_df.to_csv('breakeven_rate_comparison_fy2025.csv', index=False)
print("✓ Saved comparison results to: breakeven_rate_comparison_fy2025.csv")

# Save search history
search_df = pd.DataFrame(search_history)
search_df.to_csv('binary_search_history_fy2025.csv', index=False)
print("✓ Saved search history to: binary_search_history_fy2025.csv")

# Save detailed period data for optimal rate
best_period_summary.to_csv('optimal_rate_period_detail_fy2025.csv', index=False)
print("✓ Saved optimal rate period details to: optimal_rate_period_detail_fy2025.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nKEY FINDING: A default rate of {best_rate*100:.3f}% is needed to break even by end of FY2025")
print(f"This assumes ${BREAKEVEN_TARGET:,} in card interest needs to be offset.")
print(f"\nNote: FY2025 = April 1, 2024 through March 31, 2025")