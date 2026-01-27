'''
15_stratify_forecasted_totals - Extract Decile Distributions from Historical Data

This script analyzes historical invoice data to extract the distribution patterns
of invoices across deciles. These distributions will be used to reconstruct 
synthetic invoice populations from forecasted monthly total_discounted_price amounts.

The key insight: If we know how invoices are distributed across deciles historically,
we can use that pattern to "unpack" monthly forecast totals into synthetic invoices
that preserve the decile structure needed for revenue calculations.

Inputs:
-------
- ats_grouped_transformed_with_discounts.csv
- invoice_grouped_transformed_with_discounts.csv
- decile_payment_profile.pkl

Outputs:
--------
- forecast/15_decile_distribution_by_count.csv (% of invoices in each decile)
- forecast/15_decile_distribution_by_value.csv (% of value in each decile)
- forecast/15_decile_average_invoice_value.csv (avg invoice value per decile)
- forecast/15_monthly_decile_patterns.csv (month-by-month decile patterns)
- forecast/15_decile_distribution_visualization.png
- forecast/15_monthly_variation_heatmap.png

Author: Chris & Team
Date: January 2026
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# PATH CONFIGURATION
# ================================================================
BASE_PATH = Path(r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3")

data_cleaning_dir = BASE_PATH / "data_cleaning"
profile_dir = BASE_PATH / "payment_profile"
visualisations_dir = BASE_PATH / "visualisations"

# Create forecast output directory
FORECAST_DIR = BASE_PATH / "forecast"
FORECAST_DIR.mkdir(exist_ok=True)

print("\n" + "="*80)
print("DECILE DISTRIBUTION EXTRACTION FOR FORECASTING")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"Output Directory: {FORECAST_DIR}")
print("="*80)

# ================================================================
# STEP 1: Load Historical Invoice Data
# ================================================================
print("\n" + "="*80)
print("📁 [Step 1/7] LOADING HISTORICAL INVOICE DATA")
print("="*80)

# Load both ATS and Invoice data
ats_grouped = pd.read_csv(data_cleaning_dir / 'ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv(data_cleaning_dir / 'invoice_grouped_transformed_with_discounts.csv')

ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"  ✓ Loaded {len(ats_grouped):,} ATS invoices")
print(f"  ✓ Loaded {len(invoice_grouped):,} Invoice invoices")
print(f"  Total: {len(combined_df):,} invoices")

# Filter out negatives
combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()
combined_df = combined_df[combined_df['total_discounted_price'] >= 0].copy()

print(f"  After filtering negatives: {len(combined_df):,} invoices")

# ================================================================
# STEP 2: Parse Dates and Filter to Historical Period
# ================================================================
print("\n" + "="*80)
print("📅 [Step 2/7] PARSING DATES AND FILTERING")
print("="*80)

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

print(f"  ✓ Parsed dates: {len(combined_df):,} invoices with valid dates")
print(f"  Date range: {combined_df['invoice_period'].min()} to {combined_df['invoice_period'].max()}")

# Add year-month for grouping
combined_df['year_month'] = combined_df['invoice_period'].dt.to_period('M')

# ================================================================
# STEP 3: Load Decile Profile and Assign Deciles
# ================================================================
print("\n" + "="*80)
print("📊 [Step 3/7] LOADING DECILE PROFILE AND ASSIGNING DECILES")
print("="*80)

with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
    decile_profile = pickle.load(f)

n_deciles = decile_profile['metadata']['n_deciles']
print(f"  ✓ Loaded payment profile with {n_deciles} deciles")

# Assign deciles based on total_discounted_price (matching forecasted metric)
combined_df = combined_df.sort_values('total_discounted_price').reset_index(drop=True)
combined_df['decile'] = pd.qcut(
    combined_df['total_discounted_price'], 
    q=n_deciles, 
    labels=False,
    duplicates='drop'
)

print(f"  ✓ Assigned {len(combined_df):,} invoices to {n_deciles} deciles")

# Verify decile assignment
print(f"\n  Decile Summary:")
decile_summary = combined_df.groupby('decile').agg({
    'total_discounted_price': ['count', 'sum', 'mean', 'min', 'max']
}).round(2)
print(decile_summary.to_string())

# ================================================================
# STEP 4: Calculate Overall Decile Distributions
# ================================================================
print("\n" + "="*80)
print("📈 [Step 4/7] CALCULATING OVERALL DECILE DISTRIBUTIONS")
print("="*80)

# Distribution by COUNT (what % of invoices fall in each decile)
decile_count_dist = combined_df.groupby('decile').size().reset_index(name='n_invoices')
decile_count_dist['pct_of_total_invoices'] = (
    decile_count_dist['n_invoices'] / decile_count_dist['n_invoices'].sum() * 100
)

print("\n  Distribution by Invoice COUNT:")
print(decile_count_dist.to_string(index=False))

# Distribution by VALUE (what % of total value is in each decile)
decile_value_dist = combined_df.groupby('decile').agg({
    'total_discounted_price': 'sum'
}).reset_index()
decile_value_dist.columns = ['decile', 'total_value']
decile_value_dist['pct_of_total_value'] = (
    decile_value_dist['total_value'] / decile_value_dist['total_value'].sum() * 100
)

print("\n  Distribution by VALUE:")
print(decile_value_dist.to_string(index=False))

# Average invoice value per decile
decile_avg_value = combined_df.groupby('decile').agg({
    'total_discounted_price': 'mean'
}).reset_index()
decile_avg_value.columns = ['decile', 'avg_invoice_value']

print("\n  Average Invoice Value by Decile:")
print(decile_avg_value.to_string(index=False))

# ================================================================
# STEP 5: Calculate Monthly Variations in Decile Patterns
# ================================================================
print("\n" + "="*80)
print("📆 [Step 5/7] ANALYZING MONTHLY VARIATIONS")
print("="*80)

# For each month, calculate the distribution
monthly_decile_patterns = []

for year_month in combined_df['year_month'].unique():
    month_data = combined_df[combined_df['year_month'] == year_month]
    
    # Count distribution
    month_count_dist = month_data.groupby('decile').size()
    total_invoices = len(month_data)
    
    # Value distribution
    month_value_dist = month_data.groupby('decile')['total_discounted_price'].sum()
    total_value = month_data['total_discounted_price'].sum()
    
    # For each decile
    for decile in range(n_deciles):
        count = month_count_dist.get(decile, 0)
        value = month_value_dist.get(decile, 0)
        
        monthly_decile_patterns.append({
            'year_month': str(year_month),
            'invoice_period': year_month.to_timestamp(),
            'decile': decile,
            'n_invoices': count,
            'pct_invoices': count / total_invoices * 100 if total_invoices > 0 else 0,
            'total_value': value,
            'pct_value': value / total_value * 100 if total_value > 0 else 0,
            'avg_invoice_value': value / count if count > 0 else 0
        })

monthly_df = pd.DataFrame(monthly_decile_patterns)

print(f"  ✓ Analyzed {len(monthly_df['year_month'].unique())} months")
print(f"  ✓ Created {len(monthly_df)} month-decile combinations")

# Calculate coefficient of variation for each decile across months
print("\n  Stability of Decile Distributions Across Months:")
print("  (Lower CV = more stable pattern)")
stability = monthly_df.groupby('decile')['pct_invoices'].agg(['mean', 'std'])
stability['cv'] = stability['std'] / stability['mean']
print(stability.to_string())

# ================================================================
# STEP 6: Save All Distributions to CSV
# ================================================================
print("\n" + "="*80)
print("💾 [Step 6/7] SAVING DISTRIBUTIONS TO CSV")
print("="*80)

# Save count distribution
count_output = FORECAST_DIR / '15_decile_distribution_by_count.csv'
decile_count_dist.to_csv(count_output, index=False)
print(f"  ✓ Saved: {count_output.name}")

# Save value distribution
value_output = FORECAST_DIR / '15_decile_distribution_by_value.csv'
decile_value_dist.to_csv(value_output, index=False)
print(f"  ✓ Saved: {value_output.name}")

# Save average values
avg_output = FORECAST_DIR / '15_decile_average_invoice_value.csv'
decile_avg_value.to_csv(avg_output, index=False)
print(f"  ✓ Saved: {avg_output.name}")

# Save monthly patterns
monthly_output = FORECAST_DIR / '15_monthly_decile_patterns.csv'
monthly_df.to_csv(monthly_output, index=False)
print(f"  ✓ Saved: {monthly_output.name}")

# Create a combined summary file
summary_df = decile_count_dist.merge(
    decile_value_dist[['decile', 'total_value', 'pct_of_total_value']], 
    on='decile'
).merge(
    decile_avg_value, 
    on='decile'
)

summary_output = FORECAST_DIR / '15_decile_distribution_summary.csv'
summary_df.to_csv(summary_output, index=False)
print(f"  ✓ Saved: {summary_output.name}")

# ================================================================
# STEP 7: Create Visualizations
# ================================================================
print("\n" + "="*80)
print("🎨 [Step 7/7] CREATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Decile Distribution Overview
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Count distribution
ax1 = axes[0, 0]
ax1.bar(decile_count_dist['decile'], decile_count_dist['pct_of_total_invoices'],
        color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_title('Distribution by Invoice Count', fontsize=14, fontweight='bold')
ax1.set_xlabel('Decile', fontsize=12)
ax1.set_ylabel('% of Total Invoices', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')
for i, row in decile_count_dist.iterrows():
    ax1.text(row['decile'], row['pct_of_total_invoices'], 
             f"{row['pct_of_total_invoices']:.1f}%",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Value distribution
ax2 = axes[0, 1]
ax2.bar(decile_value_dist['decile'], decile_value_dist['pct_of_total_value'],
        color='coral', edgecolor='black', alpha=0.7)
ax2.set_title('Distribution by Value', fontsize=14, fontweight='bold')
ax2.set_xlabel('Decile', fontsize=12)
ax2.set_ylabel('% of Total Value', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
for i, row in decile_value_dist.iterrows():
    ax2.text(row['decile'], row['pct_of_total_value'], 
             f"{row['pct_of_total_value']:.1f}%",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Average invoice value
ax3 = axes[1, 0]
ax3.bar(decile_avg_value['decile'], decile_avg_value['avg_invoice_value'],
        color='#70AD47', edgecolor='black', alpha=0.7)
ax3.set_title('Average Invoice Value by Decile', fontsize=14, fontweight='bold')
ax3.set_xlabel('Decile', fontsize=12)
ax3.set_ylabel('Average Invoice Value ($)', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
for i, row in decile_avg_value.iterrows():
    ax3.text(row['decile'], row['avg_invoice_value'], 
             f"${row['avg_invoice_value']:,.0f}",
             ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=45)

# Plot 4: Cumulative distribution
ax4 = axes[1, 1]
cumulative_count = decile_count_dist['pct_of_total_invoices'].cumsum()
cumulative_value = decile_value_dist['pct_of_total_value'].cumsum()
ax4.plot(decile_count_dist['decile'], cumulative_count, 
         marker='o', linewidth=2.5, markersize=8, label='By Count', color='steelblue')
ax4.plot(decile_value_dist['decile'], cumulative_value, 
         marker='s', linewidth=2.5, markersize=8, label='By Value', color='coral')
ax4.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Decile', fontsize=12)
ax4.set_ylabel('Cumulative %', fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='50%')

plt.tight_layout()
viz_output = FORECAST_DIR / '15_decile_distribution_visualization.png'
plt.savefig(viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_output.name}")
plt.close()

# Visualization 2: Monthly Variation Heatmap
print("  Creating monthly variation heatmap...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Pivot for heatmap - % of invoices
pivot_count = monthly_df.pivot(index='year_month', columns='decile', values='pct_invoices')

ax1 = axes[0]
sns.heatmap(pivot_count, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': '% of Invoices'}, ax=ax1, linewidths=0.5)
ax1.set_title('Monthly Variation: % of Invoices by Decile', fontsize=14, fontweight='bold')
ax1.set_xlabel('Decile', fontsize=12)
ax1.set_ylabel('Month', fontsize=12)

# Pivot for heatmap - % of value
pivot_value = monthly_df.pivot(index='year_month', columns='decile', values='pct_value')

ax2 = axes[1]
sns.heatmap(pivot_value, annot=True, fmt='.1f', cmap='YlGnBu', 
            cbar_kws={'label': '% of Value'}, ax=ax2, linewidths=0.5)
ax2.set_title('Monthly Variation: % of Value by Decile', fontsize=14, fontweight='bold')
ax2.set_xlabel('Decile', fontsize=12)
ax2.set_ylabel('Month', fontsize=12)

plt.tight_layout()
heatmap_output = FORECAST_DIR / '15_monthly_variation_heatmap.png'
plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {heatmap_output.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ DECILE DISTRIBUTION EXTRACTION COMPLETE!")
print("="*80)

print(f"\n📊 Key Statistics:")
print(f"  Total invoices analyzed: {len(combined_df):,}")
print(f"  Number of deciles: {n_deciles}")
print(f"  Date range: {combined_df['invoice_period'].min().strftime('%Y-%m')} to {combined_df['invoice_period'].max().strftime('%Y-%m')}")
print(f"  Months analyzed: {len(combined_df['year_month'].unique())}")

print(f"\n💡 Distribution Insights:")
print(f"  Most invoices in decile: {decile_count_dist.loc[decile_count_dist['pct_of_total_invoices'].idxmax(), 'decile']} "
      f"({decile_count_dist['pct_of_total_invoices'].max():.1f}% of invoices)")
print(f"  Most value in decile: {decile_value_dist.loc[decile_value_dist['pct_of_total_value'].idxmax(), 'decile']} "
      f"({decile_value_dist['pct_of_total_value'].max():.1f}% of value)")
print(f"  Highest avg invoice value: Decile {decile_avg_value.loc[decile_avg_value['avg_invoice_value'].idxmax(), 'decile']} "
      f"(${decile_avg_value['avg_invoice_value'].max():,.2f})")
print(f"  Lowest avg invoice value: Decile {decile_avg_value.loc[decile_avg_value['avg_invoice_value'].idxmin(), 'decile']} "
      f"(${decile_avg_value['avg_invoice_value'].min():,.2f})")

print(f"\n📁 Output Files Saved to: {FORECAST_DIR}")
print("  • 15_decile_distribution_by_count.csv")
print("  • 15_decile_distribution_by_value.csv")
print("  • 15_decile_average_invoice_value.csv")
print("  • 15_monthly_decile_patterns.csv")
print("  • 15_decile_distribution_summary.csv")
print("  • 15_decile_distribution_visualization.png")
print("  • 15_monthly_variation_heatmap.png")

print("\n" + "="*80)
print("NEXT STEP: Use these distributions to reconstruct synthetic invoices")
print("           from forecasted monthly totals")
print("="*80)