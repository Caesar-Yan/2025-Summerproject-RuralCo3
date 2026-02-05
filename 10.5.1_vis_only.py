"""
10.5.1_vis_only.py

Recreate the 10.5_monthly_revenue_uncertainty_rebundled visualization
using the 10.5_FY2025_seasonal_comparison_summary_rebundled.csv
with font sizes from 15.6.1_cumulative_revenue_with_ci_last_12_months.png

This script creates a simplified version of the monthly revenue uncertainty plot
using summary statistics instead of full simulation data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_PATH = Path(r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3")
ALT_BASE_PATH = Path(r"\\file\Usersc$\cch155\Home\Desktop\2025\data605\2025-Summerproject-RuralCo3")
visualisations_dir = BASE_PATH / "visualisations"
alt_vis_dir = ALT_BASE_PATH

TARGET_REVENUE = 1_043_000

print("\n" + "="*80)
print("RECREATING MONTHLY REVENUE UNCERTAINTY VISUALIZATION")
print("Using 10.5_FY2025_seasonal_comparison_summary_rebundled.csv")
print("With 15.6.1 font sizes")
print("="*80)

# ================================================================
# LOAD DATA
# ================================================================
print("\n📁 Loading summary data...")

# Try to find the CSV file
csv_file = visualisations_dir / '10.5_FY2025_seasonal_comparison_summary_rebundled.csv'
if not csv_file.exists():
    csv_file = alt_vis_dir / '10.5_FY2025_seasonal_comparison_summary_rebundled.csv'

if not csv_file.exists():
    print(f"❌ Error: Could not find 10.5_FY2025_seasonal_comparison_summary_rebundled.csv")
    print(f"   Checked: {visualisations_dir}")
    print(f"   Checked: {alt_vis_dir}")
    exit(1)

summary_df = pd.read_csv(csv_file)
print(f"✓ Loaded summary data: {len(summary_df)} scenarios")
print(f"Columns: {summary_df.columns.tolist()}")

# ================================================================
# PREPARE DATA FOR VISUALIZATION
# ================================================================
print("\n📊 Preparing visualization data...")

# Extract scenarios
with_discount = summary_df[summary_df['scenario'] == 'with_discount'].iloc[0]
no_discount = summary_df[summary_df['scenario'] == 'no_discount'].iloc[0]

# Format revenue values for display
def format_revenue(val):
    """Format revenue values as $XXXK or $X.XXM"""
    if val < 1e6:
        return f'${val/1e3:.0f}K'
    else:
        return f'${val/1e6:.2f}M'

print(f"\nWith Discount Scenario:")
print(f"  Mean Revenue: {format_revenue(with_discount['total_revenue_mean'])}")
print(f"  95% CI: [{format_revenue(with_discount['total_revenue_ci_lower'])}, {format_revenue(with_discount['total_revenue_ci_upper'])}]")

print(f"\nNo Discount Scenario:")
print(f"  Mean Revenue: {format_revenue(no_discount['total_revenue_mean'])}")
print(f"  95% CI: [{format_revenue(no_discount['total_revenue_ci_lower'])}, {format_revenue(no_discount['total_revenue_ci_upper'])}]")

# ================================================================
# CREATE VISUALIZATION
# ================================================================
print("\n🎨 Creating visualization...")

fig, ax = plt.subplots(figsize=(14, 8))

# Create dummy x positions for scenarios (since we don't have monthly data from summary CSV)
scenarios = ['WITH DISCOUNT', 'NO DISCOUNT']
x_pos = [0, 1]
colors = ['#70AD47', '#4472C4']

# Plot bars with error bars representing 95% CI
means = [with_discount['total_revenue_mean'], no_discount['total_revenue_mean']]
ci_lower = [with_discount['total_revenue_ci_lower'], no_discount['total_revenue_ci_lower']]
ci_upper = [with_discount['total_revenue_ci_upper'], no_discount['total_revenue_ci_upper']]

# Calculate error bar sizes
yerr_lower = [means[0] - ci_lower[0], means[1] - ci_lower[1]]
yerr_upper = [ci_upper[0] - means[0], ci_upper[1] - means[1]]
yerr = [yerr_lower, yerr_upper]

# Create bar plot with error bars
bars = ax.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
              yerr=yerr, capsize=10, error_kw={'linewidth': 2, 'ecolor': 'black'})

# Add value labels on bars
for i, (bar, mean_val, ci_low, ci_high) in enumerate(zip(bars, means, ci_lower, ci_upper)):
    height = bar.get_height()
    # Add mean value label
    ax.text(bar.get_x() + bar.get_width()/2., height + yerr_upper[i] + 20000,
            format_revenue(mean_val),
            ha='center', va='bottom', fontsize=12, fontweight='bold', 
            color=colors[i])
    
    # Add CI range label
    ax.text(bar.get_x() + bar.get_width()/2., height/2,
            f'95% CI:\n[{format_revenue(ci_low)},\n {format_revenue(ci_high)}]',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

# Add target line
ax.axhline(TARGET_REVENUE, color='red', linestyle='--', linewidth=2.5, 
           label=f'Target: {format_revenue(TARGET_REVENUE)}', alpha=0.8, zorder=3)

# Add target label
ax.text(0.5, TARGET_REVENUE + 30000, f'Target: {format_revenue(TARGET_REVENUE)}',
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='red',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'))

# Set labels and title with 15.6.1 font sizes
ax.set_title('Revenue Scenarios with 95% Confidence Intervals\n(Monte Carlo Simulation Results)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Scenario', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Revenue ($)', fontsize=14, fontweight='bold')

# Format axes
ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios, fontsize=12, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
ax.tick_params(axis='y', labelsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add legend
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)

# Add summary text box
summary_text = f"""SIMULATION SUMMARY

With Discount:
  Mean: {format_revenue(with_discount['total_revenue_mean'])}
  Std Dev: {format_revenue(with_discount['total_revenue_std'])}

No Discount:  
  Mean: {format_revenue(no_discount['total_revenue_mean'])}
  Std Dev: {format_revenue(no_discount['total_revenue_std'])}

Revenue Difference:
  {format_revenue(no_discount['total_revenue_mean'] - with_discount['total_revenue_mean'])} 
  ({(no_discount['total_revenue_mean'] / with_discount['total_revenue_mean']):.2f}x)"""

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9),
        family='monospace')

# ================================================================
# SAVE VISUALIZATION
# ================================================================
plt.tight_layout()

# Save to both possible output directories
output_file = '10.5.1_revenue_uncertainty_from_summary.png'

# Try to save to original visualisations directory
try:
    output_path = visualisations_dir / output_file
    visualisations_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")
except Exception as e:
    print(f"⚠️ Could not save to {visualisations_dir}: {e}")

# Also save to alternative directory
try:
    alt_output_path = alt_vis_dir / output_file
    plt.savefig(alt_output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {alt_output_path}")
except Exception as e:
    print(f"⚠️ Could not save to {alt_vis_dir}: {e}")

plt.close()

print(f"\n" + "="*80)
print("✅ VISUALIZATION COMPLETE")
print("="*80)
print(f"\nRecreated revenue uncertainty visualization using:")
print(f"  • Data source: 10.5_FY2025_seasonal_comparison_summary_rebundled.csv")
print(f"  • Font sizes: 15.6.1 style (title: 16pt, axes: 14pt)")
print(f"  • Output: {output_file}")
print(f"\nNote: This is a simplified version showing confidence intervals")
print(f"      from summary statistics rather than full monthly uncertainty bands.")
print("="*80)