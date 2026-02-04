'''
9.8_apply_seasonal_adjustments_to_bundles.py

This script applies seasonal late payment rate adjustments to the bundled statement data.
It uses the reconstructed historical late payment rates from 09.6 and applies them to each
statement based on its invoice_period (month).

The seasonal adjustment modifies the baseline decile late payment probability to account
for seasonal variations in payment behavior (inverse relationship with spending).

Inputs:
- 9.7_ats_grouped_transformed_with_discounts_bundled.csv
- 9.7_invoice_grouped_transformed_with_discounts_bundled.csv
- 09.6_reconstructed_late_payment_rates.csv
- payment_profile/decile_payment_profile.pkl

Outputs:
- 9.8_ats_grouped_transformed_with_discounts_bundled_seasonal.csv
- 9.8_invoice_grouped_transformed_with_discounts_bundled_seasonal.csv
- 9.8_seasonal_adjustment_summary.csv
- 9.8_seasonal_adjustment_visualization.png

Author: Chris & Team
Date: January 2026
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

print("\n" + "="*80)
print("APPLY SEASONAL ADJUSTMENTS TO BUNDLED STATEMENTS")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print("="*80)

# ================================================================
# STEP 1: Load Bundled Invoice Data from 9.7
# ================================================================
print("\n" + "="*80)
print("📂 [Step 1/5] LOADING BUNDLED INVOICE DATA FROM 9.7")
print("="*80)

ats_bundled_path = data_cleaning_dir / '9.7_ats_grouped_transformed_with_discounts_bundled.csv'
invoice_bundled_path = data_cleaning_dir / '9.7_invoice_grouped_transformed_with_discounts_bundled.csv'

if not ats_bundled_path.exists() or not invoice_bundled_path.exists():
    print("❌ ERROR: Bundled invoice files not found!")
    print("Please run 9.7_bundle_invoices_to_statements.py first")
    exit(1)

ats_bundled = pd.read_csv(ats_bundled_path)
invoice_bundled = pd.read_csv(invoice_bundled_path)

print(f"  ✓ Loaded {len(ats_bundled):,} ATS invoices")
print(f"  ✓ Loaded {len(invoice_bundled):,} Invoice invoices")
print(f"  ✓ Total statements (ATS): {ats_bundled['statement_id'].nunique():,}")
print(f"  ✓ Total statements (Invoice): {invoice_bundled['statement_id'].nunique():,}")

# ================================================================
# STEP 2: Load Seasonal Late Payment Rates from 09.6
# ================================================================
print("\n" + "="*80)
print("📊 [Step 2/5] LOADING SEASONAL LATE PAYMENT RATES FROM 09.6")
print("="*80)

seasonal_rates_path = visualisations_dir / "09.6_reconstructed_late_payment_rates.csv"

if not seasonal_rates_path.exists():
    print("❌ ERROR: Seasonal late payment rates file not found!")
    print("Please run 09.6_seasonal_late_payment_reconstruction.py first")
    exit(1)

seasonal_rates = pd.read_csv(seasonal_rates_path)
seasonal_rates['invoice_period'] = pd.to_datetime(seasonal_rates['invoice_period'])

print(f"  ✓ Loaded {len(seasonal_rates)} months of seasonal rates")
print(f"  Date range: {seasonal_rates['invoice_period'].min()} to {seasonal_rates['invoice_period'].max()}")

# Display the seasonal rates
print(f"\n  Seasonal Late Payment Rates:")
print(f"  {'Month':<15} {'Late Rate %':<15} {'Baseline?'}")
print(f"  {'-'*45}")
for _, row in seasonal_rates.iterrows():
    baseline_mark = "✓ BASELINE" if row['is_observed_baseline'] else ""
    print(f"  {row['month_year']:<15} {row['reconstructed_late_rate_pct']:>10.2f}%     {baseline_mark}")

# ================================================================
# STEP 3: Load Payment Profile (for baseline rates)
# ================================================================
print("\n" + "="*80)
print("📋 [Step 3/5] LOADING PAYMENT PROFILE (BASELINE)")
print("="*80)

try:
    with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
        decile_profile = pickle.load(f)
    
    n_deciles = decile_profile['metadata']['n_deciles']
    print(f"  ✓ Loaded decile payment profile")
    print(f"    Number of deciles: {n_deciles}")
    
    # Extract baseline late payment rates by decile
    print(f"\n  Baseline Late Payment Rates by Decile (November 2025):")
    print(f"  {'Decile':<10} {'Late Rate %':<15}")
    print(f"  {'-'*30}")
    
    baseline_rates = {}
    for i in range(n_deciles):
        decile_key = f'decile_{i}'
        if decile_key in decile_profile['deciles']:
            prob_late = decile_profile['deciles'][decile_key]['payment_behavior']['prob_late']
            baseline_rates[i] = prob_late * 100
            print(f"  {i:<10} {prob_late*100:>10.2f}%")
    
except FileNotFoundError:
    print("  ❌ ERROR: Payment profile not found!")
    print("  Please run 09.3_Payment_profile_ML_clustering.py first")
    exit()

# ================================================================
# STEP 4: Apply Seasonal Adjustments
# ================================================================
print("\n" + "="*80)
print("🔧 [Step 4/5] APPLYING SEASONAL ADJUSTMENTS")
print("="*80)

def parse_invoice_period(series: pd.Series) -> pd.Series:
    """Parse invoice_period robustly"""
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

def apply_seasonal_adjustment(df, seasonal_rates_df, baseline_rates_dict, dataset_name):
    """
    Apply seasonal late payment rate adjustments to bundled invoice data
    
    Args:
        df: Bundled invoice DataFrame
        seasonal_rates_df: DataFrame with reconstructed seasonal rates
        baseline_rates_dict: Dictionary of baseline late rates by decile
        dataset_name: Name for logging (e.g., 'ATS' or 'Invoice')
    
    Returns:
        DataFrame with seasonal adjustments applied
    """
    print(f"\n  Processing {dataset_name} dataset...")
    
    # Parse invoice_period
    df = df.copy()
    df['invoice_period'] = parse_invoice_period(df['invoice_period'])
    
    # Filter out invalid dates
    initial_count = len(df)
    df = df[df['invoice_period'].notna()].copy()
    if len(df) < initial_count:
        print(f"    ⚠ Filtered out {initial_count - len(df):,} rows with invalid dates")
    
    # Create year-month for matching
    df['year_month'] = df['invoice_period'].dt.to_period('M')
    seasonal_rates_df['year_month'] = seasonal_rates_df['invoice_period'].dt.to_period('M')
    
    # Merge seasonal rates
    df = df.merge(
        seasonal_rates_df[['year_month', 'reconstructed_late_rate_pct', 'spending_ratio_to_november']],
        on='year_month',
        how='left'
    )
    
    # Check for missing matches
    missing_matches = df['reconstructed_late_rate_pct'].isna().sum()
    if missing_matches > 0:
        print(f"    ⚠ WARNING: {missing_matches:,} invoices could not be matched to seasonal rates")
        print(f"      These will use baseline (November) rates")
        
        # For missing matches, use overall average seasonal rate
        avg_seasonal_rate = seasonal_rates_df['reconstructed_late_rate_pct'].mean()
        df['reconstructed_late_rate_pct'].fillna(avg_seasonal_rate, inplace=True)
        df['spending_ratio_to_november'].fillna(1.0, inplace=True)
    
    # Add baseline decile rates
    df['baseline_decile_late_rate_pct'] = df['decile'].map(baseline_rates_dict)
    
    # Calculate adjustment factor: seasonal_rate / overall_baseline_rate
    # This gives us a multiplier to apply to decile-specific rates
    overall_baseline_rate = np.mean(list(baseline_rates_dict.values()))
    df['seasonal_adjustment_factor'] = df['reconstructed_late_rate_pct'] / overall_baseline_rate
    
    # Apply seasonal adjustment to decile-specific baseline rates
    # This preserves the decile differences while adjusting for seasonality
    df['adjusted_late_rate_pct'] = df['baseline_decile_late_rate_pct'] * df['seasonal_adjustment_factor']
    
    # Convert to probability (0-1)
    df['seasonal_late_prob'] = df['adjusted_late_rate_pct'] / 100
    
    # Ensure probabilities are in valid range [0, 1]
    df['seasonal_late_prob'] = df['seasonal_late_prob'].clip(0, 1)
    df['adjusted_late_rate_pct'] = df['seasonal_late_prob'] * 100
    
    print(f"    ✓ Applied seasonal adjustments to {len(df):,} invoices")
    print(f"    ✓ Adjustment factors range: {df['seasonal_adjustment_factor'].min():.3f} to {df['seasonal_adjustment_factor'].max():.3f}")
    print(f"    ✓ Adjusted late rates range: {df['adjusted_late_rate_pct'].min():.2f}% to {df['adjusted_late_rate_pct'].max():.2f}%")
    
    return df

# Apply adjustments to both datasets
ats_seasonal = apply_seasonal_adjustment(ats_bundled, seasonal_rates, baseline_rates, 'ATS')
invoice_seasonal = apply_seasonal_adjustment(invoice_bundled, seasonal_rates, baseline_rates, 'Invoice')

print(f"\n  ✓ Seasonal adjustments complete")

# ================================================================
# STEP 5: Save Seasonally-Adjusted Data
# ================================================================
print("\n" + "="*80)
print("💾 [Step 5/5] SAVING SEASONALLY-ADJUSTED DATA")
print("="*80)

# Save adjusted bundled data
ats_output = data_cleaning_dir / '9.8_ats_grouped_transformed_with_discounts_bundled_seasonal.csv'
ats_seasonal.to_csv(ats_output, index=False)
print(f"  ✓ Saved: {ats_output.name}")

invoice_output = data_cleaning_dir / '9.8_invoice_grouped_transformed_with_discounts_bundled_seasonal.csv'
invoice_seasonal.to_csv(invoice_output, index=False)
print(f"  ✓ Saved: {invoice_output.name}")

# Create summary of seasonal adjustments
print("\n  Creating seasonal adjustment summary...")

# Combine datasets
all_seasonal = pd.concat([ats_seasonal, invoice_seasonal], ignore_index=True)

# Extract unique statements for summary
statement_summary = all_seasonal.groupby('statement_id').agg({
    'invoice_period': 'first',
    'year_month': 'first',
    'decile': 'first',
    'baseline_decile_late_rate_pct': 'first',
    'reconstructed_late_rate_pct': 'first',
    'seasonal_adjustment_factor': 'first',
    'adjusted_late_rate_pct': 'first',
    'seasonal_late_prob': 'first',
    'spending_ratio_to_november': 'first',
    'customer_type': 'first'
}).reset_index()

# Monthly summary
monthly_summary = all_seasonal.groupby('year_month').agg({
    'statement_id': 'nunique',
    'baseline_decile_late_rate_pct': 'mean',
    'reconstructed_late_rate_pct': 'first',
    'adjusted_late_rate_pct': 'mean',
    'seasonal_adjustment_factor': 'mean',
    'spending_ratio_to_november': 'first'
}).reset_index()

monthly_summary.columns = [
    'year_month', 'n_statements', 'avg_baseline_rate_pct', 
    'seasonal_rate_pct', 'avg_adjusted_rate_pct', 
    'avg_adjustment_factor', 'spending_ratio'
]

summary_output = visualisations_dir / '9.8_seasonal_adjustment_summary.csv'
monthly_summary.to_csv(summary_output, index=False)
print(f"  ✓ Saved: {summary_output.name}")

# ================================================================
# Create Visualization
# ================================================================
print("\n" + "="*80)
print("🎨 CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Seasonal adjustment factors over time
ax1 = axes[0, 0]
monthly_summary_sorted = monthly_summary.sort_values('year_month')
month_labels = [str(ym) for ym in monthly_summary_sorted['year_month']]

ax1.plot(range(len(monthly_summary_sorted)), monthly_summary_sorted['avg_adjustment_factor'],
         marker='o', linewidth=2.5, markersize=8, color='#4472C4', label='Seasonal Adjustment Factor')
ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (No Adjustment)')
ax1.set_title('Seasonal Adjustment Factors Over Time', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Adjustment Factor (vs November)', fontsize=12)
ax1.set_xticks(range(len(month_labels)))
ax1.set_xticklabels(month_labels, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Add value labels
for i, (idx, row) in enumerate(monthly_summary_sorted.iterrows()):
    ax1.text(i, row['avg_adjustment_factor'], f"{row['avg_adjustment_factor']:.2f}",
             ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 2: Baseline vs Adjusted late rates
ax2 = axes[0, 1]
x_pos = np.arange(len(monthly_summary_sorted))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, monthly_summary_sorted['avg_baseline_rate_pct'],
                width, label='Baseline (November)', color='#70AD47', alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x_pos + width/2, monthly_summary_sorted['avg_adjusted_rate_pct'],
                width, label='Seasonally Adjusted', color='#FFC000', alpha=0.7, edgecolor='black')

ax2.set_title('Baseline vs Seasonally Adjusted Late Payment Rates', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Late Payment Rate (%)', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(month_labels, rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Distribution of adjusted late rates by decile
ax3 = axes[1, 0]
decile_summary = statement_summary.groupby('decile').agg({
    'baseline_decile_late_rate_pct': 'mean',
    'adjusted_late_rate_pct': ['min', 'mean', 'max']
}).reset_index()

decile_summary.columns = ['decile', 'baseline', 'adj_min', 'adj_mean', 'adj_max']

x_pos = decile_summary['decile'].values
ax3.plot(x_pos, decile_summary['baseline'], marker='s', linewidth=2.5, 
         markersize=10, color='#70AD47', label='Baseline (No Seasonality)')
ax3.plot(x_pos, decile_summary['adj_mean'], marker='o', linewidth=2.5,
         markersize=10, color='#FFC000', label='Adjusted Mean')

# Add error bars showing seasonal variation
yerr = [decile_summary['adj_mean'] - decile_summary['adj_min'],
        decile_summary['adj_max'] - decile_summary['adj_mean']]
ax3.errorbar(x_pos, decile_summary['adj_mean'], yerr=yerr,
             fmt='none', color='#FFC000', alpha=0.5, capsize=5, linewidth=2,
             label='Seasonal Range')

ax3.set_title('Late Payment Rates by Decile\nBaseline vs Seasonally Adjusted', 
              fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('Decile', fontsize=12)
ax3.set_ylabel('Late Payment Rate (%)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Statements by month
ax4 = axes[1, 1]
monthly_counts = monthly_summary_sorted.copy()

bars = ax4.bar(range(len(monthly_counts)), monthly_counts['n_statements'],
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

ax4.set_title('Number of Statements by Month', fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel('Month', fontsize=12)
ax4.set_ylabel('Number of Statements', fontsize=12)
ax4.set_xticks(range(len(month_labels)))
ax4.set_xticklabels(month_labels, rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, count) in enumerate(zip(bars, monthly_counts['n_statements'])):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{int(count):,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
viz_output = visualisations_dir / '9.8_seasonal_adjustment_visualization.png'
plt.savefig(viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_output.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ SEASONAL ADJUSTMENT COMPLETE!")
print("="*80)

print(f"\n📊 Summary Statistics:")
print(f"  Total invoices processed: {len(all_seasonal):,}")
print(f"    ATS invoices: {len(ats_seasonal):,}")
print(f"    Invoice invoices: {len(invoice_seasonal):,}")

print(f"\n  Total unique statements: {all_seasonal['statement_id'].nunique():,}")

print(f"\n📈 Seasonal Adjustment Range:")
print(f"  Adjustment factors: {all_seasonal['seasonal_adjustment_factor'].min():.3f} to {all_seasonal['seasonal_adjustment_factor'].max():.3f}")
print(f"  Baseline late rates (decile avg): {all_seasonal['baseline_decile_late_rate_pct'].mean():.2f}%")
print(f"  Adjusted late rates (range): {all_seasonal['adjusted_late_rate_pct'].min():.2f}% to {all_seasonal['adjusted_late_rate_pct'].max():.2f}%")
print(f"  Adjusted late rates (mean): {all_seasonal['adjusted_late_rate_pct'].mean():.2f}%")

print(f"\n📁 Output Files:")
print(f"  • 9.8_ats_grouped_transformed_with_discounts_bundled_seasonal.csv")
print(f"  • 9.8_invoice_grouped_transformed_with_discounts_bundled_seasonal.csv")
print(f"  • 9.8_seasonal_adjustment_summary.csv")
print(f"  • 9.8_seasonal_adjustment_visualization.png")

print(f"\n  All files saved to: {data_cleaning_dir}/")

print("\n" + "="*80)
print("KEY METHODOLOGY:")
print("="*80)
print("  1. Loaded bundled statements from 9.7 (with decile assignments)")
print("  2. Loaded seasonal late payment rates from 09.6 (month-specific)")
print("  3. For each statement:")
print("     - Identified its month (invoice_period)")
print("     - Retrieved seasonal late rate for that month")
print("     - Calculated adjustment factor vs November baseline")
print("     - Applied adjustment to decile-specific baseline rate")
print("  4. Result: Each statement has a seasonally-adjusted late probability")
print("     that accounts for BOTH decile and month-of-year effects")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("  1. Modify 10.0.3 to use seasonal late rates instead of static rates")
print("  2. Verify that seasonal adjustments improve revenue forecast accuracy")
print("  3. Compare seasonally-adjusted vs non-seasonal simulations")
print("="*80)

print("\n✓ Ready for seasonal revenue simulation!")

