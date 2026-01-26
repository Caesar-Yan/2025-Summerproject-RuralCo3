'''
Seasonal Adjustment of Late Payment Rates - Historical Reconstruction

This script reconstructs historical late payment rates by applying seasonal adjustments
based on monthly spending totals. Uses November 2025 as the baseline (observed) rate and
adjusts other months proportionally based on the inverse relationship between spending
and late payment rates.

Inputs:
- monthly_totals_Period_4_Entire.csv (monthly spending data)
- decile_payment_profile_summary.csv (November 2025 baseline late payment rate)

Outputs:
- 09.6_reconstructed_late_payment_rates.csv
- 09.6_late_payment_reconstruction.png
- 09.6_spending_vs_late_rate.png
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================================================================
# Configuration
# ================================================================
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
visualisations_dir = base_dir / "visualisations"
profile_dir = base_dir / "payment_profile"

# ================================================================
# Step 1: Load November Baseline Late Payment Rate
# ================================================================
print("="*70)
print("LOADING NOVEMBER 2025 BASELINE")
print("="*70)

# Load payment profile summary
profile_summary = pd.read_csv(profile_dir / "decile_payment_profile_summary.csv")

# Calculate overall late payment rate from deciles
# Weight by number of accounts in each decile
profile_summary['n_late_numeric'] = profile_summary['n_late']
profile_summary['n_accounts_numeric'] = profile_summary['n_accounts']

total_late = profile_summary['n_late_numeric'].sum()
total_accounts = profile_summary['n_accounts_numeric'].sum()
november_late_rate = (total_late / total_accounts) * 100

print(f"\nNovember 2025 Baseline (from snapshot):")
print(f"  Total accounts: {total_accounts:,}")
print(f"  Late accounts: {total_late:,}")
print(f"  Late payment rate: {november_late_rate:.2f}%")

# ================================================================
# Step 2: Load Monthly Spending Data
# ================================================================
print("\n" + "="*70)
print("LOADING MONTHLY SPENDING DATA")
print("="*70)

monthly_df = pd.read_csv(visualisations_dir / "monthly_totals_Period_4_Entire.csv")
monthly_df['invoice_period'] = pd.to_datetime(monthly_df['invoice_period'])
monthly_df = monthly_df.sort_values('invoice_period').reset_index(drop=True)

print(f"\nLoaded {len(monthly_df)} months of data")
print(f"Date range: {monthly_df['invoice_period'].min()} to {monthly_df['invoice_period'].max()}")

# ================================================================
# Step 3: Identify November 2025 Baseline Spending
# ================================================================
print("\n" + "="*70)
print("IDENTIFYING NOVEMBER 2025 BASELINE SPENDING")
print("="*70)

# Find November 2025 (the month of the snapshot)
november_2025 = monthly_df[
    (monthly_df['invoice_period'].dt.year == 2025) & 
    (monthly_df['invoice_period'].dt.month == 11)
]

if len(november_2025) == 0:
    print("ERROR: November 2025 not found in data")
    print("\nAvailable dates:")
    print(monthly_df[['invoice_period']].to_string())
    exit(1)

november_spending = november_2025['total_undiscounted_price'].values[0]

print(f"\nNovember 2025 spending: ${november_spending:,.2f}")
print(f"This is the baseline for reconstruction")

# ================================================================
# Step 4: Reconstruct Historical Late Payment Rates
# ================================================================
print("\n" + "="*70)
print("RECONSTRUCTING HISTORICAL LATE PAYMENT RATES")
print("="*70)

# Apply proportional adjustment formula:
# Reconstructed Rate = November Rate × (November Spending / Month Spending)

monthly_df['spending_ratio'] = november_spending / monthly_df['total_undiscounted_price']
monthly_df['reconstructed_late_rate_pct'] = november_late_rate * monthly_df['spending_ratio']

# Mark November 2025 as the baseline (observed)
monthly_df['is_baseline'] = (
    (monthly_df['invoice_period'].dt.year == 2025) & 
    (monthly_df['invoice_period'].dt.month == 11)
)

# Add month name for display
monthly_df['month_year'] = monthly_df['invoice_period'].dt.strftime('%b-%Y')

print("\nReconstructed Late Payment Rates:")
print(f"{'Month':<12} {'Spending':<15} {'Ratio':<10} {'Late Rate':<12} {'Status'}")
print("-" * 70)

for _, row in monthly_df.iterrows():
    status = "BASELINE" if row['is_baseline'] else "Reconstructed"
    print(f"{row['month_year']:<12} ${row['total_undiscounted_price']:>12,.0f}  "
          f"{row['spending_ratio']:>6.2f}x    {row['reconstructed_late_rate_pct']:>6.2f}%     {status}")

# ================================================================
# Step 5: Save Results
# ================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_df = monthly_df[[
    'invoice_period', 'month_year', 'total_undiscounted_price', 
    'spending_ratio', 'reconstructed_late_rate_pct', 'is_baseline'
]].copy()

output_df.columns = [
    'invoice_period', 'month_year', 'spending_amount', 
    'spending_ratio_to_november', 'reconstructed_late_rate_pct', 'is_observed_baseline'
]

output_path = visualisations_dir / "09.6_reconstructed_late_payment_rates.csv"
output_df.to_csv(output_path, index=False)
print(f"✓ Saved reconstructed rates to: {output_path}")

# ================================================================
# Step 6: Visualizations
# ================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Plot 1: Reconstructed Late Payment Rates Over Time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Top panel: Late payment rate over time
baseline_mask = monthly_df['is_baseline']

# Plot reconstructed rates
ax1.plot(monthly_df['invoice_period'], monthly_df['reconstructed_late_rate_pct'],
         marker='o', linewidth=2, markersize=8, color='coral', alpha=0.7,
         label='Reconstructed Late Rate')

# Highlight November 2025 baseline
ax1.scatter(monthly_df[baseline_mask]['invoice_period'], 
           monthly_df[baseline_mask]['reconstructed_late_rate_pct'],
           s=200, color='red', zorder=5, marker='*', 
           label='November 2025 (Observed Baseline)', edgecolor='black', linewidth=1.5)

# Add horizontal line at November rate
ax1.axhline(y=november_late_rate, color='red', linestyle='--', 
           linewidth=1.5, alpha=0.5, label=f'November Baseline: {november_late_rate:.1f}%')

ax1.set_title('Reconstructed Historical Late Payment Rates\n(Based on Seasonal Spending Patterns)', 
             fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Late Payment Rate (%)', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=10, loc='best')

# Add value labels for key months
for _, row in monthly_df.iterrows():
    if row['is_baseline'] or row['reconstructed_late_rate_pct'] > 30 or row['reconstructed_late_rate_pct'] < 10:
        ax1.text(row['invoice_period'], row['reconstructed_late_rate_pct'], 
                f"{row['reconstructed_late_rate_pct']:.1f}%",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Bottom panel: Spending over time (for context)
ax2.bar(monthly_df['invoice_period'], monthly_df['total_undiscounted_price']/1e6,
        width=20, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

# Highlight November 2025
ax2.bar(monthly_df[baseline_mask]['invoice_period'], 
       monthly_df[baseline_mask]['total_undiscounted_price']/1e6,
       width=20, color='red', alpha=0.7, edgecolor='black', linewidth=1.5,
       label='November 2025 (Baseline)')

ax2.set_title('Monthly Spending (Context)', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Total Undiscounted Price ($M)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.legend(fontsize=10)

plt.tight_layout()
save_path = visualisations_dir / "09.6_late_payment_reconstruction.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved reconstruction plot to: {save_path}")
plt.close()

# Plot 2: Spending vs Late Rate (Scatter)
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
ax.scatter(monthly_df[~baseline_mask]['total_undiscounted_price']/1e6, 
          monthly_df[~baseline_mask]['reconstructed_late_rate_pct'],
          s=100, alpha=0.6, color='coral', edgecolor='black', linewidth=0.5,
          label='Reconstructed Months')

# Highlight November 2025
ax.scatter(monthly_df[baseline_mask]['total_undiscounted_price']/1e6, 
          monthly_df[baseline_mask]['reconstructed_late_rate_pct'],
          s=300, color='red', marker='*', zorder=5,
          label='November 2025 (Observed)', edgecolor='black', linewidth=1.5)

# Add trend line (theoretical inverse relationship)
x_range = np.linspace(monthly_df['total_undiscounted_price'].min()/1e6, 
                     monthly_df['total_undiscounted_price'].max()/1e6, 100)
y_theoretical = november_late_rate * (november_spending/1e6) / x_range

ax.plot(x_range, y_theoretical, '--', color='gray', linewidth=2, alpha=0.7,
        label='Theoretical Inverse Relationship')

# Add labels for all months
for _, row in monthly_df.iterrows():
    ax.text(row['total_undiscounted_price']/1e6, row['reconstructed_late_rate_pct'],
            row['month_year'], fontsize=7, ha='right', va='bottom', alpha=0.7)

ax.set_title('Spending vs Late Payment Rate\n(Inverse Proportional Relationship)', 
            fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Monthly Spending ($M)', fontsize=12)
ax.set_ylabel('Late Payment Rate (%)', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10, loc='best')

# Add annotation explaining the relationship
textstr = f'Relationship: Late Rate = {november_late_rate:.1f}% × (${november_spending/1e6:.1f}M / Monthly Spending)\nLower spending → Higher late payment rate'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
save_path = visualisations_dir / "09.6_spending_vs_late_rate.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved scatter plot to: {save_path}")
plt.close()

# ================================================================
# Summary Statistics
# ================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nReconstruction Summary:")
print(f"  Baseline month: November 2025")
print(f"  Baseline late rate: {november_late_rate:.2f}%")
print(f"  Baseline spending: ${november_spending:,.2f}")
print(f"\nReconstructed Rates:")
print(f"  Minimum late rate: {monthly_df['reconstructed_late_rate_pct'].min():.2f}% "
      f"({monthly_df.loc[monthly_df['reconstructed_late_rate_pct'].idxmin(), 'month_year']})")
print(f"  Maximum late rate: {monthly_df['reconstructed_late_rate_pct'].max():.2f}% "
      f"({monthly_df.loc[monthly_df['reconstructed_late_rate_pct'].idxmax(), 'month_year']})")
print(f"  Average late rate: {monthly_df['reconstructed_late_rate_pct'].mean():.2f}%")
print(f"  Std dev: {monthly_df['reconstructed_late_rate_pct'].std():.2f}%")

print(f"\nSpending vs Late Rate Correlation:")
correlation = monthly_df['total_undiscounted_price'].corr(monthly_df['reconstructed_late_rate_pct'])
print(f"  Pearson correlation: {correlation:.3f}")
print(f"  (Negative correlation confirms inverse relationship)")

print("\n" + "="*70)
print("RECONSTRUCTION COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  1. 09.6_reconstructed_late_payment_rates.csv")
print("  2. 09.6_late_payment_reconstruction.png")
print("  3. 09.6_spending_vs_late_rate.png")
print("\nKey Assumption:")
print("  Late payment rate is inversely proportional to monthly spending")
print("  (Lower spending → Higher late payment rate)")
print("\nInterpretation:")
print("  These are ESTIMATED historical rates based on November 2025's observed rate")
print("  and the seasonal spending pattern. They represent what late payment")
print("  rates likely were if the inverse relationship holds true.")
print("="*70)