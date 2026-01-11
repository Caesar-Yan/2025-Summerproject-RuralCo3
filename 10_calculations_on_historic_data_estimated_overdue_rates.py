import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================================================================
# CONFIGURATION
# ================================================================
BREAKEVEN_TARGET = 1_043_000
FY2025_START = pd.Timestamp("2024-04-01")  # Start of FY2025
FY2025_END = pd.Timestamp("2025-03-31")    # End of FY2025

# ================================================================
# Load pre-calculated results from payment_days analysis
# ================================================================
print("="*70)
print("REVENUE ANALYSIS FOR FY2025")
print("FROM PAYMENT-BASED INTEREST AND LATE FEES")
print("="*70)

# Load the detailed results from the payment_days analysis
try:
    excel_file = 'payment_days_scenario_detailed_results.xlsx'
    
    # Load both scenarios
    full_amount_data = pd.read_excel(excel_file, sheet_name='Full_Amount')
    discounted_amount_data = pd.read_excel(excel_file, sheet_name='Discounted_Amount')
    summary_data = pd.read_excel(excel_file, sheet_name='Summary')
    
    print(f"\n✓ Loaded period data from: {excel_file}")
    
except FileNotFoundError as e:
    print(f"\n✗ ERROR: Could not find required file: {excel_file}")
    print(f"  Please run the payment_days analysis script first.")
    raise

# ================================================================
# Filter to FY2025 (April 1, 2024 to March 31, 2025)
# ================================================================
full_amount_fy2025 = full_amount_data[
    (full_amount_data['invoice_period'] >= FY2025_START) & 
    (full_amount_data['invoice_period'] <= FY2025_END)
].copy()

discounted_amount_fy2025 = discounted_amount_data[
    (discounted_amount_data['invoice_period'] >= FY2025_START) & 
    (discounted_amount_data['invoice_period'] <= FY2025_END)
].copy()

# Recalculate cumulative values starting from FY2025_START
full_amount_fy2025['cumulative_discount_offered'] = full_amount_fy2025['total_discount_offered'].cumsum()
full_amount_fy2025['cumulative_discount_retained'] = full_amount_fy2025['discount_retained'].cumsum()
full_amount_fy2025['cumulative_interest_earned'] = full_amount_fy2025['interest_earned'].cumsum()
full_amount_fy2025['cumulative_late_fees'] = full_amount_fy2025['late_fees'].cumsum()
full_amount_fy2025['cumulative_total_revenue'] = full_amount_fy2025['total_revenue_impact'].cumsum()

discounted_amount_fy2025['cumulative_discount_offered'] = discounted_amount_fy2025['total_discount_offered'].cumsum()
discounted_amount_fy2025['cumulative_discount_retained'] = discounted_amount_fy2025['discount_retained'].cumsum()
discounted_amount_fy2025['cumulative_interest_earned'] = discounted_amount_fy2025['interest_earned'].cumsum()
discounted_amount_fy2025['cumulative_late_fees'] = discounted_amount_fy2025['late_fees'].cumsum()
discounted_amount_fy2025['cumulative_total_revenue'] = discounted_amount_fy2025['total_revenue_impact'].cumsum()

print(f"\nFiltered to FY2025 ({FY2025_START.strftime('%Y-%m-%d')} to {FY2025_END.strftime('%Y-%m-%d')})")
print(f"  Full amount scenario: {len(full_amount_fy2025)} periods")
print(f"  Discounted amount scenario: {len(discounted_amount_fy2025)} periods")

# ================================================================
# Calculate totals for FY2025
# ================================================================
print(f"\n{'='*70}")
print(f"REVENUE FOR FY2025 (April 1, 2024 - March 31, 2025)")
print(f"{'='*70}")

scenarios = {
    'Full Invoice Amount': full_amount_fy2025,
    'Discounted Amount Only': discounted_amount_fy2025
}

results = []

for scenario_name, period_data in scenarios.items():
    if len(period_data) > 0:
        # Get the final row (cumulative totals at end of FY2025)
        final_row = period_data.iloc[-1]
        
        total_discount_offered = final_row['cumulative_discount_offered']
        discount_retained = final_row['cumulative_discount_retained']
        interest_earned = final_row['cumulative_interest_earned']
        late_fees = final_row['cumulative_late_fees']
        total_revenue = final_row['cumulative_total_revenue']
        last_period = final_row['invoice_period']
        
        print(f"\n{'='*50}")
        print(f"Scenario: Interest on {scenario_name}")
        print(f"{'='*50}")
        print(f"Last period included: {pd.to_datetime(last_period).strftime('%Y-%m')}")
        print(f"\nRevenue breakdown:")
        print(f"  Discount retained: ${discount_retained:,.2f} ({discount_retained/total_revenue*100:.1f}%)")
        print(f"  Interest earned: ${interest_earned:,.2f} ({interest_earned/total_revenue*100:.1f}%)")
        print(f"  Late fees: ${late_fees:,.2f} ({late_fees/total_revenue*100:.1f}%)")
        print(f"  {'─'*45}")
        print(f"  TOTAL REVENUE: ${total_revenue:,.2f}")
        
        print(f"\nBreakeven analysis:")
        print(f"  Breakeven target: ${BREAKEVEN_TARGET:,.2f}")
        print(f"  Gap to breakeven: ${total_revenue - BREAKEVEN_TARGET:+,.2f}")
        
        if total_revenue >= BREAKEVEN_TARGET:
            print(f"  Status: ✓ BREAKEVEN REACHED")
        else:
            print(f"  Status: ✗ Below target")
            shortfall_pct = ((BREAKEVEN_TARGET - total_revenue) / BREAKEVEN_TARGET) * 100
            print(f"  Shortfall: {shortfall_pct:.1f}% below target")
        
        results.append({
            'scenario': scenario_name,
            'last_period': pd.to_datetime(last_period).strftime('%Y-%m'),
            'discount_retained': discount_retained,
            'interest_earned': interest_earned,
            'late_fees': late_fees,
            'total_revenue': total_revenue,
            'gap_to_breakeven': total_revenue - BREAKEVEN_TARGET,
            'reaches_breakeven': total_revenue >= BREAKEVEN_TARGET
        })

# ================================================================
# Comparison between scenarios
# ================================================================
print(f"\n{'='*70}")
print(f"COMPARISON BETWEEN SCENARIOS")
print(f"{'='*70}")

if len(results) == 2:
    diff_interest = results[0]['interest_earned'] - results[1]['interest_earned']
    diff_total = results[0]['total_revenue'] - results[1]['total_revenue']
    
    print(f"\nDifference (Full Amount vs. Discounted Amount):")
    print(f"  Additional interest earned: ${diff_interest:,.2f}")
    print(f"  Additional total revenue: ${diff_total:,.2f}")
    print(f"  Discount retained (same): ${results[0]['discount_retained']:,.2f}")
    print(f"  Late fees (same): ${results[0]['late_fees']:,.2f}")

# ================================================================
# Create visualization
# ================================================================
print(f"\n{'='*70}")
print(f"GENERATING VISUALIZATION")
print(f"{'='*70}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))

# Color scheme
colors = {'Full Invoice Amount': '#4472C4', 'Discounted Amount Only': '#70AD47'}

# Plot 1: Cumulative revenue over time
for scenario_name, period_data in scenarios.items():
    mark_every = max(1, len(period_data)//20)
    
    ax1.plot(pd.to_datetime(period_data['invoice_period']), 
             period_data['cumulative_total_revenue'],
             color=colors[scenario_name], linewidth=2.5,
             label=f"Total revenue ({scenario_name})",
             marker='o', markersize=4, markevery=mark_every)

# Reference line for discount offered
ref_data = full_amount_fy2025
ax1.plot(pd.to_datetime(ref_data['invoice_period']), 
         ref_data['cumulative_discount_offered'],
         color="#222222", linewidth=1.5, alpha=0.6,
         label='Total discount offered (if all pay early)',
         marker='s', markersize=3, markevery=max(1, len(ref_data)//20), 
         linestyle='--')

# Breakeven line
ax1.axhline(y=BREAKEVEN_TARGET, color='#FF0000', linewidth=2,
           linestyle='-', alpha=0.8, label=f'Breakeven target (${BREAKEVEN_TARGET:,})', zorder=10)

# End of FY2025 line
ax1.axvline(x=FY2025_END, color='#8B7BA8', linewidth=2,
           linestyle=':', alpha=0.7, label='End of FY2025 (Mar 31, 2025)', zorder=9)

# Year-end marker for 2024
ax1.axvline(x=pd.Timestamp("2024-12-31"), color='#D3D3D3', linewidth=1,
           linestyle=':', alpha=0.5, label='End of CY2024')

ax1.set_xlim(left=FY2025_START, right=FY2025_END)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax1.tick_params(axis='x', rotation=45)

ax1.set_xlabel('Invoice Period', fontsize=13, fontweight='bold')
ax1.set_ylabel('$ Revenue Amount', fontsize=13, fontweight='bold')
ax1.set_title('Cumulative Revenue FY2025: Payment-Based Interest Comparison',
             fontsize=16, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.set_facecolor('#F8F8F8')

# Plot 2: Revenue component breakdown
for scenario_name, period_data in scenarios.items():
    scenario_short = "Full" if scenario_name == 'Full Invoice Amount' else "Discounted"
    
    ax2.plot(pd.to_datetime(period_data['invoice_period']), 
             period_data['cumulative_discount_retained'],
             color=colors[scenario_name], linewidth=2, alpha=0.8,
             label=f"{scenario_short}: Discount retained", linestyle='-')
    
    ax2.plot(pd.to_datetime(period_data['invoice_period']), 
             period_data['cumulative_interest_earned'],
             color=colors[scenario_name], linewidth=2, alpha=0.6,
             label=f"{scenario_short}: Interest earned", linestyle='--')
    
    ax2.plot(pd.to_datetime(period_data['invoice_period']), 
             period_data['cumulative_late_fees'],
             color=colors[scenario_name], linewidth=1.5, alpha=0.5,
             label=f"{scenario_short}: Late fees", linestyle=':')

ax2.set_xlim(left=FY2025_START, right=FY2025_END)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax2.tick_params(axis='x', rotation=45)

ax2.set_xlabel('Invoice Period', fontsize=13, fontweight='bold')
ax2.set_ylabel('$ Amount', fontsize=13, fontweight='bold')
ax2.set_title('Revenue Components Breakdown by Scenario (FY2025)',
             fontsize=16, fontweight='bold', pad=15)
ax2.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.set_facecolor('#F8F8F8')

plt.tight_layout()
plt.savefig('fy2025_revenue_analysis_payment_based.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to: fy2025_revenue_analysis_payment_based.png")

# ================================================================
# Save results
# ================================================================
results_df = pd.DataFrame(results)
results_df.to_csv('fy2025_revenue_summary.csv', index=False)
print("✓ Saved FY2025 summary to: fy2025_revenue_summary.csv")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")

if len(results) > 0:
    best_scenario = max(results, key=lambda x: x['total_revenue'])
    print(f"\nBest scenario: {best_scenario['scenario']}")
    print(f"  Total revenue for FY2025: ${best_scenario['total_revenue']:,.2f}")
    
    if best_scenario['reaches_breakeven']:
        print(f"  ✓ Reaches breakeven target")
    else:
        print(f"  ✗ Does not reach breakeven (${best_scenario['gap_to_breakeven']:+,.2f})")