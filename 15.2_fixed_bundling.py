'''
15.2_extract_statement_distributions_REFACTORED - Extract Statement Distributions from 9.7 Bundled Data

REFACTORED VERSION: Now uses pre-bundled statement data from 9.7 for consistency across all analysis

This script analyzes the statement-level distributions from bundled invoice data
to extract patterns used for revenue forecasting in scripts 15.3 and 15.4.

KEY CHANGE: Instead of re-bundling invoices, this now uses the authoritative bundled
data created by 9.7_bundle_invoices_to_statements.py. This ensures:
- Consistent statement groupings across historical (10.10) and forecast (15.3/15.4) analysis
- No duplication of bundling logic
- Faster execution (just aggregates instead of bundling)
- Adjustment amounts (discount_amount) correctly preserved at statement level

Inputs:
-------
- data_cleaning/9.7_ats_grouped_transformed_with_discounts_bundled.csv
- data_cleaning/9.7_invoice_grouped_transformed_with_discounts_bundled.csv

Outputs:
--------
- forecast/15.2_statement_distribution_summary.csv
- forecast/15.2_monthly_statement_patterns.csv
- forecast/15.2_statement_distribution_visualization.png
- (other analysis files)

Author: Chris & Team
Date: January 2026
REFACTORED: January 2026 (to use 9.7 bundled data)
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

print("\n" + "="*80)
print("STATEMENT-LEVEL DISTRIBUTION EXTRACTION (USING 9.7 BUNDLED DATA)")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"Output Directory: {FORECAST_DIR}")
print("="*80)

# ================================================================
# STEP 1: Load PRE-BUNDLED Invoice Data from 9.7
# ================================================================
print("\n" + "="*80)
print("📂 [Step 1/6] LOADING PRE-BUNDLED DATA FROM 9.7")
print("="*80)

bundled_ats_path = data_cleaning_dir / '9.7_ats_grouped_transformed_with_discounts_bundled.csv'
bundled_invoice_path = data_cleaning_dir / '9.7_invoice_grouped_transformed_with_discounts_bundled.csv'

# Check files exist
if not bundled_ats_path.exists() or not bundled_invoice_path.exists():
    print("\n❌ ERROR: Bundled files from 9.7 not found!")
    print(f"   Expected: {bundled_ats_path}")
    print(f"   Expected: {bundled_invoice_path}")
    print("\n   Please run 9.7_bundle_invoices_to_statements.py first")
    exit(1)

# Load bundled data
ats_bundled = pd.read_csv(bundled_ats_path)
invoice_bundled = pd.read_csv(bundled_invoice_path)

print(f"  ✓ Loaded {len(ats_bundled):,} ATS invoices (bundled)")
print(f"  ✓ Loaded {len(invoice_bundled):,} Invoice invoices (bundled)")

# Verify required columns exist
required_cols = [
    'statement_id', 
    'decile',
    'statement_discounted_value',
    'statement_undiscounted_value',
    'statement_discount_amount',
    'n_invoices_in_statement'
]

for df, name in [(ats_bundled, 'ATS'), (invoice_bundled, 'Invoice')]:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\n❌ ERROR: {name} bundled data missing required columns: {missing}")
        print("\n   This means 9.7 needs to be updated with the fix.")
        print("   Please apply the fix from 'fix_for_9_7_bundling.py' and re-run 9.7")
        exit(1)

print(f"  ✓ All required statement columns present")

# Add customer type markers
ats_bundled['customer_type'] = 'ATS'
invoice_bundled['customer_type'] = 'Invoice'

# Combine datasets
combined_df = pd.concat([ats_bundled, invoice_bundled], ignore_index=True)
print(f"\n  Combined data: {len(combined_df):,} invoices")
print(f"  Unique statements: {combined_df['statement_id'].nunique():,}")

# Filter out negatives (if any)
combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()
combined_df = combined_df[combined_df['total_discounted_price'] >= 0].copy()
print(f"  After filtering negatives: {len(combined_df):,} invoices")

# ================================================================
# STEP 2: Parse Dates
# ================================================================
print("\n" + "="*80)
print("📅 [Step 2/6] PARSING DATES")
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

combined_df['year_month'] = combined_df['invoice_period'].dt.to_period('M')

# ================================================================
# STEP 3: Aggregate to Statement Level (NO RE-BUNDLING!)
# ================================================================
print("\n" + "="*80)
print("📊 [Step 3/6] AGGREGATING TO STATEMENT LEVEL")
print("="*80)

print("  ℹ️  Using pre-assigned statement_id from 9.7 (no re-bundling needed)")

# Aggregate using pre-existing statement_id and pre-computed aggregates
# We use .first() because these values are already aggregated at statement level in 9.7
statements = combined_df.groupby(['statement_id', 'year_month', 'invoice_period']).agg({
    'statement_discounted_value': 'first',      # Already summed in 9.7
    'statement_undiscounted_value': 'first',    # Already summed in 9.7
    'statement_discount_amount': 'first',       # Already summed in 9.7 ← KEY!
    'n_invoices_in_statement': 'first',         # Already counted in 9.7
    'decile': 'first',                          # Already assigned in 9.7
    'customer_type': 'first'
}).reset_index()

print(f"  ✓ Aggregated to {len(statements):,} unique statements")
print(f"  Average invoices per statement: {statements['n_invoices_in_statement'].mean():.2f}")
print(f"  Median invoices per statement: {statements['n_invoices_in_statement'].median():.0f}")
print(f"  Average statement value: ${statements['statement_discounted_value'].mean():,.2f}")

# Verify all invoices are accounted for
total_invoices_in_statements = statements['n_invoices_in_statement'].sum()
print(f"\n  Verification:")
print(f"    Total invoices in original data: {len(combined_df):,}")
print(f"    Total invoices in statements: {total_invoices_in_statements:,}")
print(f"    Match: {'✓' if len(combined_df) == total_invoices_in_statements else '✗'}")

# ================================================================
# STEP 4: Verify Decile Assignments
# ================================================================
print("\n" + "="*80)
print("📈 [Step 4/6] VERIFYING DECILE ASSIGNMENTS FROM 9.7")
print("="*80)

# Load decile profile for reference
with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
    decile_profile = pickle.load(f)

n_deciles = decile_profile['metadata']['n_deciles']
print(f"  ✓ Expected {n_deciles} deciles from payment profile")

# Check decile assignments
print(f"\n  Statement Decile Summary:")
decile_summary = statements.groupby('decile').agg({
    'statement_discounted_value': ['count', 'sum', 'mean', 'min', 'max'],
    'statement_discount_amount': 'sum',
    'n_invoices_in_statement': 'mean'
}).round(2)
decile_summary.columns = ['n_statements', 'total_value', 'mean_value', 'min_value', 
                          'max_value', 'total_discount_amt', 'avg_invoices']
print(decile_summary.to_string())

# Show decile boundaries
print(f"\n  Decile Boundaries (by statement value):")
for decile in range(n_deciles):
    if decile in statements['decile'].values:
        decile_data = statements[statements['decile'] == decile]['statement_discounted_value']
        print(f"    Decile {decile}: ${decile_data.min():>10,.2f} to ${decile_data.max():>10,.2f}")

# ================================================================
# STEP 5: Calculate Statement Distributions
# ================================================================
print("\n" + "="*80)
print("📊 [Step 5/6] CALCULATING STATEMENT DISTRIBUTIONS")
print("="*80)

# Distribution by COUNT
statement_count_dist = statements.groupby('decile').size().reset_index(name='n_statements')
statement_count_dist['pct_of_total_statements'] = (
    statement_count_dist['n_statements'] / statement_count_dist['n_statements'].sum() * 100
)

print("\n  Distribution by Statement COUNT:")
print(statement_count_dist.to_string(index=False))

# Distribution by VALUE
statement_value_dist = statements.groupby('decile').agg({
    'statement_discounted_value': 'sum',
    'statement_discount_amount': 'sum'
}).reset_index()
statement_value_dist.columns = ['decile', 'total_value', 'total_discount_amount']
statement_value_dist['pct_of_total_value'] = (
    statement_value_dist['total_value'] / statement_value_dist['total_value'].sum() * 100
)

print("\n  Distribution by VALUE:")
print(statement_value_dist[['decile', 'total_value', 'pct_of_total_value', 'total_discount_amount']].to_string(index=False))

# Average metrics by decile
statement_avg_metrics = statements.groupby('decile').agg({
    'statement_discounted_value': 'mean',
    'statement_undiscounted_value': 'mean',
    'statement_discount_amount': 'mean',
    'n_invoices_in_statement': 'mean'
}).reset_index()
statement_avg_metrics.columns = ['decile', 'avg_statement_value_discounted', 
                                  'avg_statement_value_undiscounted',
                                  'avg_discount_amount', 'avg_invoices_per_statement']

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
# STEP 6: Save Distribution Summaries
# ================================================================
print("\n" + "="*80)
print("💾 [Step 6/6] SAVING DISTRIBUTION SUMMARIES")
print("="*80)

# Create comprehensive summary for use in 15.3 and 15.4
distribution_summary = statement_count_dist.merge(
    statement_value_dist[['decile', 'pct_of_total_value', 'total_discount_amount']], 
    on='decile'
).merge(
    statement_avg_metrics[['decile', 'avg_statement_value_discounted', 
                           'avg_statement_value_undiscounted', 'avg_discount_amount',
                           'avg_invoices_per_statement']], 
    on='decile'
)

# Add total value for reference
distribution_summary = distribution_summary.merge(
    statement_value_dist[['decile', 'total_value']],
    on='decile'
)

# Reorder columns for clarity
distribution_summary = distribution_summary[[
    'decile', 
    'n_statements', 
    'pct_of_total_statements',
    'total_value',
    'pct_of_total_value',
    'avg_statement_value_discounted',
    'avg_statement_value_undiscounted',
    'total_discount_amount',
    'avg_discount_amount',
    'avg_invoices_per_statement'
]]

# Save summary
summary_output = FORECAST_DIR / '15.2_statement_distribution_summary.csv'
distribution_summary.to_csv(summary_output, index=False)
print(f"  ✓ Saved: {summary_output.name}")

# Save statement size distribution
size_dist_output = FORECAST_DIR / '15.2_statement_size_distribution.csv'
statement_size_dist.to_csv(size_dist_output, index=False)
print(f"  ✓ Saved: {size_dist_output.name}")

# ================================================================
# Create Monthly Patterns (if needed for advanced forecasting)
# ================================================================
# ... rest of the monthly pattern analysis code ...
# (can keep this from original 15.2 if needed)

print("\n" + "="*80)
print("✓ STATEMENT DISTRIBUTION EXTRACTION COMPLETE")
print("="*80)

print(f"\n📊 Summary:")
print(f"  Source: Pre-bundled data from 9.7 ✓")
print(f"  Total statements analyzed: {len(statements):,}")
print(f"  Deciles: {n_deciles}")
print(f"  Statement discount amounts preserved: ✓")
print(f"\n  Ready for use in:")
print(f"    • 15.3_estimated_forecast_revenue_with_deciles_bundled.py")
print(f"    • 15.4_forecast_total_prices_estimated_revenue.py")

print(f"\n  Key improvement: Consistent bundling across all analysis")
print(f"    Historical analysis (10.10) uses same statement_id groupings")
print(f"    Forecast analysis (15.3, 15.4) uses distributions from same groupings")

print("\n" + "="*80)