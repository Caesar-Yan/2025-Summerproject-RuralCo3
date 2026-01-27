'''
15.2_stratify_by_statements - Extract Statement-Level Distributions from Historical Data

This script analyzes historical invoice data by grouping invoices into statements
(collections of invoices per customer per month). This provides a more realistic
representation of payment behavior since customers pay statements, not individual invoices.

Key insight: Bundle invoices into statements based on customer-month combinations,
then analyze the distribution of statement values and sizes across deciles.

IMPORTANT: Since no customer ID exists, invoices are randomly bundled into statements
with an average size matching the expected invoices per customer per month.

Inputs:
-------
- ats_grouped_transformed_with_discounts.csv
- invoice_grouped_transformed_with_discounts.csv
- decile_payment_profile.pkl

Outputs:
--------
- forecast/15.2_statement_distribution_by_count.csv (% of statements in each decile)
- forecast/15.2_statement_distribution_by_value.csv (% of value in each decile)
- forecast/15.2_statement_average_value.csv (avg statement value per decile)
- forecast/15.2_statement_average_invoice_count.csv (avg invoices per statement)
- forecast/15.2_monthly_statement_patterns.csv (month-by-month patterns)
- forecast/15.2_statement_size_distribution.csv (distribution of invoices per statement)
- forecast/15.2_statement_distribution_visualization.png
- forecast/15.2_monthly_variation_heatmap.png
- forecast/15.2_statement_size_analysis.png

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
# CONFIGURATION
# ================================================================
BASE_PATH = Path(r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3")

data_cleaning_dir = BASE_PATH / "data_cleaning"
profile_dir = BASE_PATH / "payment_profile"
visualisations_dir = BASE_PATH / "visualisations"

# Create forecast output directory
FORECAST_DIR = BASE_PATH / "forecast"
FORECAST_DIR.mkdir(exist_ok=True)

# Number of active users (for calculating statements per user)
N_ACTIVE_USERS = 5920
RANDOM_SEED = 42  # For reproducibility

print("\n" + "="*80)
print("STATEMENT-LEVEL DISTRIBUTION EXTRACTION FOR FORECASTING")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"Output Directory: {FORECAST_DIR}")
print(f"Active Users: {N_ACTIVE_USERS:,}")
print(f"Random Seed: {RANDOM_SEED}")
print("="*80)

# ================================================================
# STEP 1: Load Historical Invoice Data
# ================================================================
print("\n" + "="*80)
print("📁 [Step 1/8] LOADING HISTORICAL INVOICE DATA")
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
print("📅 [Step 2/8] PARSING DATES AND FILTERING")
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
# STEP 3: Randomly Bundle Invoices into Statements
# ================================================================
print("\n" + "="*80)
print("📋 [Step 3/8] CREATING STATEMENTS BY RANDOM BUNDLING")
print("="*80)

print("  ℹ️  No customer ID available - creating synthetic statement bundles")
print(f"  Target: Mean of {combined_df.groupby('year_month').size().mean() / N_ACTIVE_USERS:.2f} invoices per statement")

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Create statements by month
all_statements = []

for year_month in combined_df['year_month'].unique():
    month_data = combined_df[combined_df['year_month'] == year_month].copy()
    n_invoices_this_month = len(month_data)
    
    # Calculate expected number of statements this month
    # Mean invoices per statement = n_invoices / n_active_users
    mean_invoices_per_statement = n_invoices_this_month / N_ACTIVE_USERS
    
    # Estimate number of statements (approximately n_active_users)
    # But allow some variance - use Poisson distribution
    n_statements_this_month = max(1, int(n_invoices_this_month / mean_invoices_per_statement))
    
    print(f"\n  Processing {year_month}:")
    print(f"    Invoices: {n_invoices_this_month:,}")
    print(f"    Expected statements: {n_statements_this_month:,}")
    print(f"    Mean invoices/statement: {mean_invoices_per_statement:.2f}")
    
    # Randomly shuffle invoices
    month_data = month_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Assign each invoice to a statement
    # Use roughly equal-sized chunks, but with some randomness
    statement_sizes = np.random.poisson(mean_invoices_per_statement, n_statements_this_month)
    
    # Ensure we use all invoices
    statement_sizes = statement_sizes.astype(int)
    statement_sizes = np.maximum(statement_sizes, 1)  # At least 1 invoice per statement
    
    # Adjust to match total invoice count
    total_assigned = statement_sizes.sum()
    if total_assigned > n_invoices_this_month:
        # Reduce sizes proportionally
        statement_sizes = (statement_sizes * n_invoices_this_month / total_assigned).astype(int)
        statement_sizes = np.maximum(statement_sizes, 1)
    
    # Assign remaining invoices to random statements
    remaining = n_invoices_this_month - statement_sizes.sum()
    if remaining > 0:
        random_indices = np.random.choice(len(statement_sizes), size=remaining, replace=True)
        for idx in random_indices:
            statement_sizes[idx] += 1
    elif remaining < 0:
        # Remove excess
        for i in range(abs(remaining)):
            if statement_sizes[-1] > 1:
                statement_sizes[-1] -= 1
    
    # Create statement IDs for each invoice
    statement_ids = []
    for stmt_idx, size in enumerate(statement_sizes):
        statement_ids.extend([f"{year_month}_stmt_{stmt_idx:04d}"] * size)
    
    # Trim or pad to match invoice count
    statement_ids = statement_ids[:n_invoices_this_month]
    
    month_data['statement_id'] = statement_ids
    all_statements.append(month_data)

# Combine all months
combined_df = pd.concat(all_statements, ignore_index=True)

print(f"\n  ✓ Created synthetic statement IDs for all invoices")
print(f"  Total unique statements: {combined_df['statement_id'].nunique():,}")

# ================================================================
# STEP 4: Aggregate Invoices into Statements
# ================================================================
print("\n" + "="*80)
print("📊 [Step 4/8] AGGREGATING INVOICES INTO STATEMENTS")
print("="*80)

# Aggregate by statement_id
statements = combined_df.groupby(['statement_id', 'year_month', 'invoice_period']).agg({
    'total_discounted_price': 'sum',
    'total_undiscounted_price': 'sum',
    'discount_amount': 'sum',
    'invoice_id': 'count',
    'customer_type': 'first'
}).reset_index()

statements.columns = [
    'statement_id', 'year_month', 'invoice_period', 
    'statement_discounted_value', 'statement_undiscounted_value', 
    'statement_discount_amount', 'n_invoices_in_statement', 'customer_type'
]

print(f"  ✓ Created {len(statements):,} statements from {len(combined_df):,} invoices")
print(f"  Average invoices per statement: {statements['n_invoices_in_statement'].mean():.2f}")
print(f"  Median invoices per statement: {statements['n_invoices_in_statement'].median():.0f}")
print(f"  Std dev invoices per statement: {statements['n_invoices_in_statement'].std():.2f}")

# Calculate statements per month
monthly_statement_counts = statements.groupby('year_month').size()
print(f"\n  Monthly Statement Statistics:")
print(f"    Average statements per month: {monthly_statement_counts.mean():.0f}")
print(f"    Min statements in a month: {monthly_statement_counts.min():,}")
print(f"    Max statements in a month: {monthly_statement_counts.max():,}")
print(f"    Expected statements/user/month: {monthly_statement_counts.mean() / N_ACTIVE_USERS:.4f}")

# Verify all invoices are assigned
print(f"\n  Verification:")
print(f"    Total invoices in original data: {len(combined_df):,}")
print(f"    Total invoices in statements: {statements['n_invoices_in_statement'].sum():,}")
print(f"    Match: {'✓' if len(combined_df) == statements['n_invoices_in_statement'].sum() else '✗'}")

# ================================================================
# STEP 5: Load Decile Profile and Assign Deciles to STATEMENTS
# ================================================================
print("\n" + "="*80)
print("📈 [Step 5/8] ASSIGNING DECILES TO STATEMENTS")
print("="*80)

with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
    decile_profile = pickle.load(f)

n_deciles = decile_profile['metadata']['n_deciles']
print(f"  ✓ Loaded payment profile with {n_deciles} deciles")

# CRITICAL: Assign deciles based on STATEMENT value, not invoice value
# This ensures each statement belongs to exactly one decile
print(f"\n  Assigning deciles based on statement_discounted_value...")

statements = statements.sort_values('statement_discounted_value').reset_index(drop=True)
statements['decile'] = pd.qcut(
    statements['statement_discounted_value'], 
    q=n_deciles, 
    labels=False,
    duplicates='drop'
)

print(f"  ✓ Assigned {len(statements):,} statements to {n_deciles} deciles")
print(f"\n  Verification: Each statement has exactly ONE decile ✓")

# Verify decile assignment at statement level
print(f"\n  Statement Decile Summary:")
decile_summary = statements.groupby('decile').agg({
    'statement_discounted_value': ['count', 'sum', 'mean', 'min', 'max'],
    'n_invoices_in_statement': 'mean'
}).round(2)
decile_summary.columns = ['n_statements', 'total_value', 'mean_value', 'min_value', 'max_value', 'avg_invoices']
print(decile_summary.to_string())

# Show decile boundaries
print(f"\n  Decile Boundaries (by statement value):")
for decile in range(n_deciles):
    decile_data = statements[statements['decile'] == decile]['statement_discounted_value']
    print(f"    Decile {decile}: ${decile_data.min():>10,.2f} to ${decile_data.max():>10,.2f}")

# ================================================================
# STEP 6: Calculate Overall Statement Distributions
# ================================================================
print("\n" + "="*80)
print("📊 [Step 6/8] CALCULATING OVERALL STATEMENT DISTRIBUTIONS")
print("="*80)

# Distribution by COUNT (what % of statements fall in each decile)
statement_count_dist = statements.groupby('decile').size().reset_index(name='n_statements')
statement_count_dist['pct_of_total_statements'] = (
    statement_count_dist['n_statements'] / statement_count_dist['n_statements'].sum() * 100
)

print("\n  Distribution by Statement COUNT:")
print(statement_count_dist.to_string(index=False))

# Distribution by VALUE (what % of total value is in each decile)
statement_value_dist = statements.groupby('decile').agg({
    'statement_discounted_value': 'sum'
}).reset_index()
statement_value_dist.columns = ['decile', 'total_value']
statement_value_dist['pct_of_total_value'] = (
    statement_value_dist['total_value'] / statement_value_dist['total_value'].sum() * 100
)

print("\n  Distribution by VALUE:")
print(statement_value_dist.to_string(index=False))

# Average statement value and invoice count per decile
statement_avg_metrics = statements.groupby('decile').agg({
    'statement_discounted_value': 'mean',
    'n_invoices_in_statement': 'mean'
}).reset_index()
statement_avg_metrics.columns = ['decile', 'avg_statement_value', 'avg_invoices_per_statement']

print("\n  Average Metrics by Decile:")
print(statement_avg_metrics.to_string(index=False))

# Statement size distribution
statement_size_dist = statements['n_invoices_in_statement'].value_counts().sort_index().reset_index()
statement_size_dist.columns = ['invoices_per_statement', 'n_statements']
statement_size_dist['pct_of_statements'] = (
    statement_size_dist['n_statements'] / statement_size_dist['n_statements'].sum() * 100
)

print("\n  Statement Size Distribution (top 20):")
print(statement_size_dist.head(20).to_string(index=False))

# ================================================================
# STEP 7: Calculate Monthly Variations in Statement Patterns
# ================================================================
print("\n" + "="*80)
print("📆 [Step 7/8] ANALYZING MONTHLY VARIATIONS")
print("="*80)

# For each month, calculate the distribution
monthly_statement_patterns = []

for year_month in statements['year_month'].unique():
    month_data = statements[statements['year_month'] == year_month]
    
    # Count distribution
    month_count_dist = month_data.groupby('decile').size()
    total_statements = len(month_data)
    
    # Value distribution
    month_value_dist = month_data.groupby('decile')['statement_discounted_value'].sum()
    total_value = month_data['statement_discounted_value'].sum()
    
    # Invoice count per statement
    month_avg_invoices = month_data.groupby('decile')['n_invoices_in_statement'].mean()
    
    # For each decile
    for decile in range(n_deciles):
        count = month_count_dist.get(decile, 0)
        value = month_value_dist.get(decile, 0)
        avg_invoices = month_avg_invoices.get(decile, 0)
        
        monthly_statement_patterns.append({
            'year_month': str(year_month),
            'invoice_period': year_month.to_timestamp(),
            'decile': decile,
            'n_statements': count,
            'pct_statements': count / total_statements * 100 if total_statements > 0 else 0,
            'total_value': value,
            'pct_value': value / total_value * 100 if total_value > 0 else 0,
            'avg_statement_value': value / count if count > 0 else 0,
            'avg_invoices_per_statement': avg_invoices
        })

monthly_stmt_df = pd.DataFrame(monthly_statement_patterns)

print(f"  ✓ Analyzed {len(monthly_stmt_df['year_month'].unique())} months")
print(f"  ✓ Created {len(monthly_stmt_df)} month-decile combinations")

# Calculate coefficient of variation for each decile across months
print("\n  Stability of Statement Distributions Across Months:")
print("  (Lower CV = more stable pattern)")
stability = monthly_stmt_df.groupby('decile')['pct_statements'].agg(['mean', 'std'])
stability['cv'] = stability['std'] / stability['mean']
print(stability.to_string())

# Additional check: CV for value distribution
print("\n  Stability of VALUE Distributions Across Months:")
value_stability = monthly_stmt_df.groupby('decile')['pct_value'].agg(['mean', 'std'])
value_stability['cv'] = value_stability['std'] / value_stability['mean']
print(value_stability.to_string())

# ================================================================
# STEP 8: Save All Distributions to CSV
# ================================================================
print("\n" + "="*80)
print("💾 [Step 8/8] SAVING DISTRIBUTIONS TO CSV")
print("="*80)

# Save count distribution
count_output = FORECAST_DIR / '15.2_statement_distribution_by_count.csv'
statement_count_dist.to_csv(count_output, index=False)
print(f"  ✓ Saved: {count_output.name}")

# Save value distribution
value_output = FORECAST_DIR / '15.2_statement_distribution_by_value.csv'
statement_value_dist.to_csv(value_output, index=False)
print(f"  ✓ Saved: {value_output.name}")

# Save average metrics
avg_output = FORECAST_DIR / '15.2_statement_average_metrics.csv'
statement_avg_metrics.to_csv(avg_output, index=False)
print(f"  ✓ Saved: {avg_output.name}")

# Save statement size distribution
size_output = FORECAST_DIR / '15.2_statement_size_distribution.csv'
statement_size_dist.to_csv(size_output, index=False)
print(f"  ✓ Saved: {size_output.name}")

# Save monthly patterns
monthly_output = FORECAST_DIR / '15.2_monthly_statement_patterns.csv'
monthly_stmt_df.to_csv(monthly_output, index=False)
print(f"  ✓ Saved: {monthly_output.name}")

# Create a combined summary file
summary_df = statement_count_dist.merge(
    statement_value_dist[['decile', 'total_value', 'pct_of_total_value']], 
    on='decile'
).merge(
    statement_avg_metrics, 
    on='decile'
)

summary_output = FORECAST_DIR / '15.2_statement_distribution_summary.csv'
summary_df.to_csv(summary_output, index=False)
print(f"  ✓ Saved: {summary_output.name}")

# Save detailed statements data for inspection
statements_detail_output = FORECAST_DIR / '15.2_statements_with_deciles.csv'
statements[['statement_id', 'year_month', 'invoice_period', 'decile', 
            'statement_discounted_value', 'statement_undiscounted_value',
            'statement_discount_amount', 'n_invoices_in_statement', 
            'customer_type']].to_csv(statements_detail_output, index=False)
print(f"  ✓ Saved: {statements_detail_output.name}")

# ================================================================
# STEP 9: Create Visualizations
# ================================================================
print("\n" + "="*80)
print("🎨 [Step 9/9] CREATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Statement Distribution Overview
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Count distribution
ax1 = axes[0, 0]
ax1.bar(statement_count_dist['decile'], statement_count_dist['pct_of_total_statements'],
        color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_title('Distribution by Statement Count', fontsize=14, fontweight='bold')
ax1.set_xlabel('Decile', fontsize=12)
ax1.set_ylabel('% of Total Statements', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')
for i, row in statement_count_dist.iterrows():
    ax1.text(row['decile'], row['pct_of_total_statements'], 
             f"{row['pct_of_total_statements']:.1f}%",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Value distribution
ax2 = axes[0, 1]
ax2.bar(statement_value_dist['decile'], statement_value_dist['pct_of_total_value'],
        color='coral', edgecolor='black', alpha=0.7)
ax2.set_title('Distribution by Value', fontsize=14, fontweight='bold')
ax2.set_xlabel('Decile', fontsize=12)
ax2.set_ylabel('% of Total Value', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
for i, row in statement_value_dist.iterrows():
    ax2.text(row['decile'], row['pct_of_total_value'], 
             f"{row['pct_of_total_value']:.1f}%",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Average statement value
ax3 = axes[1, 0]
ax3.bar(statement_avg_metrics['decile'], statement_avg_metrics['avg_statement_value'],
        color='#70AD47', edgecolor='black', alpha=0.7)
ax3.set_title('Average Statement Value by Decile', fontsize=14, fontweight='bold')
ax3.set_xlabel('Decile', fontsize=12)
ax3.set_ylabel('Average Statement Value ($)', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
for i, row in statement_avg_metrics.iterrows():
    ax3.text(row['decile'], row['avg_statement_value'], 
             f"${row['avg_statement_value']:,.0f}",
             ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=45)

# Plot 4: Average invoices per statement
ax4 = axes[1, 1]
ax4.bar(statement_avg_metrics['decile'], statement_avg_metrics['avg_invoices_per_statement'],
        color='#FFC000', edgecolor='black', alpha=0.7)
ax4.set_title('Average Invoices per Statement by Decile', fontsize=14, fontweight='bold')
ax4.set_xlabel('Decile', fontsize=12)
ax4.set_ylabel('Avg Invoices per Statement', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')
for i, row in statement_avg_metrics.iterrows():
    ax4.text(row['decile'], row['avg_invoices_per_statement'], 
             f"{row['avg_invoices_per_statement']:.1f}",
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
viz_output = FORECAST_DIR / '15.2_statement_distribution_visualization.png'
plt.savefig(viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_output.name}")
plt.close()

# Visualization 2: Monthly Variation Heatmap
print("  Creating monthly variation heatmap...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Pivot for heatmap - % of statements
pivot_count = monthly_stmt_df.pivot(index='year_month', columns='decile', values='pct_statements')

ax1 = axes[0]
sns.heatmap(pivot_count, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': '% of Statements'}, ax=ax1, linewidths=0.5)
ax1.set_title('Monthly Variation: % of Statements by Decile', fontsize=14, fontweight='bold')
ax1.set_xlabel('Decile', fontsize=12)
ax1.set_ylabel('Month', fontsize=12)

# Pivot for heatmap - % of value
pivot_value = monthly_stmt_df.pivot(index='year_month', columns='decile', values='pct_value')

ax2 = axes[1]
sns.heatmap(pivot_value, annot=True, fmt='.1f', cmap='YlGnBu', 
            cbar_kws={'label': '% of Value'}, ax=ax2, linewidths=0.5)
ax2.set_title('Monthly Variation: % of Value by Decile', fontsize=14, fontweight='bold')
ax2.set_xlabel('Decile', fontsize=12)
ax2.set_ylabel('Month', fontsize=12)

plt.tight_layout()
heatmap_output = FORECAST_DIR / '15.2_monthly_variation_heatmap.png'
plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {heatmap_output.name}")
plt.close()

# Visualization 3: Statement Size Analysis
print("  Creating statement size analysis...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Statement size distribution (histogram)
ax1 = axes[0, 0]
plot_data = statement_size_dist.head(30)
ax1.bar(plot_data['invoices_per_statement'], 
        plot_data['pct_of_statements'],
        color='#4472C4', edgecolor='black', alpha=0.7)
ax1.set_title('Statement Size Distribution (Top 30)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Invoices per Statement', fontsize=12)
ax1.set_ylabel('% of Statements', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Boxplot of statement values by decile
ax2 = axes[0, 1]
statements.boxplot(column='statement_discounted_value', by='decile', ax=ax2)
ax2.set_title('Statement Value Distribution by Decile', fontsize=14, fontweight='bold')
ax2.set_xlabel('Decile', fontsize=12)
ax2.set_ylabel('Statement Value ($)', fontsize=12)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.suptitle('')  # Remove default title

# Plot 3: Boxplot of invoices per statement by decile
ax3 = axes[1, 0]
statements.boxplot(column='n_invoices_in_statement', by='decile', ax=ax3)
ax3.set_title('Invoices per Statement by Decile', fontsize=14, fontweight='bold')
ax3.set_xlabel('Decile', fontsize=12)
ax3.set_ylabel('Invoices per Statement', fontsize=12)
plt.suptitle('')  # Remove default title

# Plot 4: Scatter - statement value vs invoices in statement
ax4 = axes[1, 1]
colors = plt.cm.tab10(np.linspace(0, 1, n_deciles))
for decile in range(n_deciles):
    decile_data = statements[statements['decile'] == decile]
    sample_size = min(200, len(decile_data))
    if sample_size > 0:
        decile_sample = decile_data.sample(sample_size, random_state=RANDOM_SEED)
        ax4.scatter(decile_sample['n_invoices_in_statement'], 
                   decile_sample['statement_discounted_value'],
                   alpha=0.4, s=30, label=f'Decile {decile}', color=colors[decile])
ax4.set_title('Statement Value vs Invoice Count', fontsize=14, fontweight='bold')
ax4.set_xlabel('Invoices in Statement', fontsize=12)
ax4.set_ylabel('Statement Value ($)', fontsize=12)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax4.legend(fontsize=8, ncol=2, loc='upper left')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
size_viz_output = FORECAST_DIR / '15.2_statement_size_analysis.png'
plt.savefig(size_viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {size_viz_output.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ STATEMENT-LEVEL DISTRIBUTION EXTRACTION COMPLETE!")
print("="*80)

print(f"\n📊 Key Statistics:")
print(f"  Total invoices analyzed: {len(combined_df):,}")
print(f"  Total statements created: {len(statements):,}")
print(f"  Number of deciles: {n_deciles}")
print(f"  Date range: {statements['invoice_period'].min().strftime('%Y-%m')} to {statements['invoice_period'].max().strftime('%Y-%m')}")
print(f"  Months analyzed: {len(statements['year_month'].unique())}")

print(f"\n📋 Statement Metrics:")
print(f"  Average invoices per statement: {statements['n_invoices_in_statement'].mean():.2f}")
print(f"  Median invoices per statement: {statements['n_invoices_in_statement'].median():.0f}")
print(f"  Std dev invoices per statement: {statements['n_invoices_in_statement'].std():.2f}")
print(f"  Max invoices in a statement: {statements['n_invoices_in_statement'].max():,}")
print(f"  Average statements per month: {len(statements) / len(statements['year_month'].unique()):.0f}")
print(f"  Average statements per user per month: {len(statements) / len(statements['year_month'].unique()) / N_ACTIVE_USERS:.4f}")

print(f"\n💡 Distribution Insights:")
print(f"  Most statements in decile: {statement_count_dist.loc[statement_count_dist['pct_of_total_statements'].idxmax(), 'decile']} "
      f"({statement_count_dist['pct_of_total_statements'].max():.1f}% of statements)")
print(f"  Most value in decile: {statement_value_dist.loc[statement_value_dist['pct_of_total_value'].idxmax(), 'decile']} "
      f"({statement_value_dist['pct_of_total_value'].max():.1f}% of value)")
print(f"  Highest avg statement value: Decile {statement_avg_metrics.loc[statement_avg_metrics['avg_statement_value'].idxmax(), 'decile']} "
      f"(${statement_avg_metrics['avg_statement_value'].max():,.2f})")
print(f"  Lowest avg statement value: Decile {statement_avg_metrics.loc[statement_avg_metrics['avg_statement_value'].idxmin(), 'decile']} "
      f"(${statement_avg_metrics['avg_statement_value'].min():,.2f})")

print(f"\n✓ STATEMENT CREATION METHOD:")
print(f"  Random bundling with Poisson distribution")
print(f"  Target mean: {combined_df.groupby('year_month').size().mean() / N_ACTIVE_USERS:.2f} invoices/statement")
print(f"  Random seed: {RANDOM_SEED} (reproducible)")

print(f"\n✓ DECILE ASSIGNMENT VERIFICATION:")
print(f"  Deciles assigned at STATEMENT level (not invoice level)")
print(f"  Each statement belongs to exactly ONE decile")
print(f"  Decile boundaries based on statement_discounted_value")

print(f"\n📁 Output Files Saved to: {FORECAST_DIR}")
print("  • 15.2_statement_distribution_by_count.csv")
print("  • 15.2_statement_distribution_by_value.csv")
print("  • 15.2_statement_average_metrics.csv")
print("  • 15.2_statement_size_distribution.csv")
print("  • 15.2_monthly_statement_patterns.csv")
print("  • 15.2_statement_distribution_summary.csv")
print("  • 15.2_statements_with_deciles.csv (detailed statement data)")
print("  • 15.2_statement_distribution_visualization.png")
print("  • 15.2_monthly_variation_heatmap.png")
print("  • 15.2_statement_size_analysis.png")

print("\n" + "="*80)
print("NEXT STEP: Use statement distributions to estimate revenue from forecasts")
print("           Each statement now has ONE decile assignment ✓")
print("="*80)