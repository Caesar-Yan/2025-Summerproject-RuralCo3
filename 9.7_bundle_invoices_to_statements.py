'''
9.7_bundle_invoices_to_statements - Bundle Historical Invoices into Statements

This script takes the historical invoice data and bundles it into statements
using the same random bundling approach as 15.2. This creates statement-level
data that can be used for revenue estimation in Monte Carlo simulations.

The bundling process:
- Groups invoices by month
- Randomly assigns invoices to statements using Poisson distribution
- Target: mean invoices per statement = monthly invoices / 5920 active users
- Each statement gets assigned a decile based on its total value

This prepares the data for statement-based revenue analysis while maintaining
the same bundling logic as the forecasting approach.

Inputs:
-------
- data_cleaning/ats_grouped_transformed_with_discounts.csv
- data_cleaning/invoice_grouped_transformed_with_discounts.csv
- payment_profile/decile_payment_profile.pkl

Outputs:
--------
- data_cleaning/ats_grouped_transformed_with_discounts_bundled.csv
- data_cleaning/invoice_grouped_transformed_with_discounts_bundled.csv
- data_cleaning/9.7_statement_summary_bundled.csv
- data_cleaning/9.7_bundling_statistics_bundled.png

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

# Number of active users (for calculating statements per user)
N_ACTIVE_USERS = 5920
RANDOM_SEED = 42  # For reproducibility

print("\n" + "="*80)
print("BUNDLE HISTORICAL INVOICES INTO STATEMENTS")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"Active Users: {N_ACTIVE_USERS:,}")
print(f"Random Seed: {RANDOM_SEED}")
print("="*80)

# ================================================================
# STEP 1: Load Historical Invoice Data
# ================================================================
print("\n" + "="*80)
print("📁 [Step 1/6] LOADING HISTORICAL INVOICE DATA")
print("="*80)

# Load both ATS and Invoice data
ats_grouped = pd.read_csv(data_cleaning_dir / 'ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv(data_cleaning_dir / 'invoice_grouped_transformed_with_discounts.csv')

print(f"  ✓ Loaded {len(ats_grouped):,} ATS invoices")
print(f"  ✓ Loaded {len(invoice_grouped):,} Invoice invoices")

# Keep them separate for now to maintain data structure
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'

# Filter out negatives
ats_grouped = ats_grouped[ats_grouped['total_undiscounted_price'] >= 0].copy()
ats_grouped = ats_grouped[ats_grouped['total_discounted_price'] >= 0].copy()

invoice_grouped = invoice_grouped[invoice_grouped['total_undiscounted_price'] >= 0].copy()
invoice_grouped = invoice_grouped[invoice_grouped['total_discounted_price'] >= 0].copy()

print(f"  After filtering negatives:")
print(f"    ATS: {len(ats_grouped):,} invoices")
print(f"    Invoice: {len(invoice_grouped):,} invoices")

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

# Parse dates for both datasets
ats_grouped['invoice_period'] = parse_invoice_period(ats_grouped['invoice_period'])
ats_grouped = ats_grouped[ats_grouped['invoice_period'].notna()].copy()
ats_grouped['year_month'] = ats_grouped['invoice_period'].dt.to_period('M')

invoice_grouped['invoice_period'] = parse_invoice_period(invoice_grouped['invoice_period'])
invoice_grouped = invoice_grouped[invoice_grouped['invoice_period'].notna()].copy()
invoice_grouped['year_month'] = invoice_grouped['invoice_period'].dt.to_period('M')

print(f"  ✓ Parsed dates for both datasets")
print(f"  ATS date range: {ats_grouped['invoice_period'].min()} to {ats_grouped['invoice_period'].max()}")
print(f"  Invoice date range: {invoice_grouped['invoice_period'].min()} to {invoice_grouped['invoice_period'].max()}")

# ================================================================
# STEP 3: Bundle Function
# ================================================================
print("\n" + "="*80)
print("📋 [Step 3/6] DEFINING BUNDLING FUNCTION")
print("="*80)

def bundle_invoices_to_statements(df, customer_type_name):
    """
    Bundle invoices into statements using random assignment with Poisson distribution
    
    Args:
        df: DataFrame with invoice data
        customer_type_name: Name for this customer type (for statement IDs)
    
    Returns:
        DataFrame with statement_id added
    """
    np.random.seed(RANDOM_SEED)
    
    all_bundled = []
    
    for year_month in df['year_month'].unique():
        month_data = df[df['year_month'] == year_month].copy()
        n_invoices_this_month = len(month_data)
        
        # Calculate expected number of statements this month
        mean_invoices_per_statement = n_invoices_this_month / N_ACTIVE_USERS
        n_statements_this_month = max(1, int(n_invoices_this_month / mean_invoices_per_statement))
        
        # Randomly shuffle invoices
        month_data = month_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        # Generate statement sizes using Poisson distribution
        statement_sizes = np.random.poisson(mean_invoices_per_statement, n_statements_this_month)
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
            statement_ids.extend([f"{customer_type_name}_{year_month}_stmt_{stmt_idx:04d}"] * size)
        
        # Trim or pad to match invoice count
        statement_ids = statement_ids[:n_invoices_this_month]
        
        month_data['statement_id'] = statement_ids
        all_bundled.append(month_data)
    
    return pd.concat(all_bundled, ignore_index=True)

print("  ✓ Bundling function defined")

# ================================================================
# STEP 4: Bundle Both Datasets
# ================================================================
print("\n" + "="*80)
print("🔨 [Step 4/6] BUNDLING INVOICES INTO STATEMENTS")
print("="*80)

print("\n  Bundling ATS invoices...")
ats_bundled = bundle_invoices_to_statements(ats_grouped, 'ATS')
print(f"    ✓ Created {ats_bundled['statement_id'].nunique():,} statements from {len(ats_bundled):,} ATS invoices")
print(f"    Average invoices per statement: {len(ats_bundled) / ats_bundled['statement_id'].nunique():.2f}")

print("\n  Bundling Invoice invoices...")
invoice_bundled = bundle_invoices_to_statements(invoice_grouped, 'INV')
print(f"    ✓ Created {invoice_bundled['statement_id'].nunique():,} statements from {len(invoice_bundled):,} Invoice invoices")
print(f"    Average invoices per statement: {len(invoice_bundled) / invoice_bundled['statement_id'].nunique():.2f}")

# ================================================================
# STEP 5: Assign Deciles to Statements
# ================================================================
print("\n" + "="*80)
print("📊 [Step 5/6] ASSIGNING DECILES TO STATEMENTS")
print("="*80)

# Load decile profile
with open(profile_dir / 'decile_payment_profile.pkl', 'rb') as f:
    decile_profile = pickle.load(f)

n_deciles = decile_profile['metadata']['n_deciles']
print(f"  ✓ Loaded payment profile with {n_deciles} deciles")

def assign_deciles_to_statements(df):
    """
    Assign deciles to statements based on statement total value
    """
    # Aggregate to statement level
    statements = df.groupby(['statement_id', 'year_month', 'invoice_period']).agg({
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
    
    # Assign deciles based on statement value
    statements = statements.sort_values('statement_discounted_value').reset_index(drop=True)
    statements['decile'] = pd.qcut(
        statements['statement_discounted_value'],
        q=n_deciles,
        labels=False,
        duplicates='drop'
    )
    
    # Map decile back to invoices
    statement_deciles = statements[['statement_id', 'decile']].set_index('statement_id')
    df = df.merge(statement_deciles, left_on='statement_id', right_index=True, how='left')
    
    return df, statements

print("\n  Assigning deciles to ATS statements...")
ats_bundled, ats_statements = assign_deciles_to_statements(ats_bundled)
print(f"    ✓ Assigned deciles to {len(ats_bundled):,} ATS invoices")
print(f"    ✓ Created {len(ats_statements):,} ATS statements")

print("\n  Assigning deciles to Invoice statements...")
invoice_bundled, invoice_statements = assign_deciles_to_statements(invoice_bundled)
print(f"    ✓ Assigned deciles to {len(invoice_bundled):,} Invoice invoices")
print(f"    ✓ Created {len(invoice_statements):,} Invoice statements")

# ================================================================
# STEP 6: Save Bundled Data
# ================================================================
print("\n" + "="*80)
print("💾 [Step 6/6] SAVING BUNDLED DATA")
print("="*80)

# Save bundled invoice data
ats_output = data_cleaning_dir / '9.7_ats_grouped_transformed_with_discounts_bundled.csv'
ats_bundled.to_csv(ats_output, index=False)
print(f"  ✓ Saved: {ats_output.name}")

invoice_output = data_cleaning_dir / '9.7_invoice_grouped_transformed_with_discounts_bundled.csv'
invoice_bundled.to_csv(invoice_output, index=False)
print(f"  ✓ Saved: {invoice_output.name}")

# Create combined statement summary
all_statements = pd.concat([ats_statements, invoice_statements], ignore_index=True)

statement_summary = all_statements.groupby('decile').agg({
    'statement_id': 'count',
    'statement_discounted_value': ['sum', 'mean', 'min', 'max'],
    'n_invoices_in_statement': 'mean'
}).round(2)

statement_summary.columns = ['n_statements', 'total_value', 'mean_value', 'min_value', 'max_value', 'avg_invoices']

summary_output = data_cleaning_dir / '9.7_statement_summary_bundled.csv'
statement_summary.to_csv(summary_output)
print(f"  ✓ Saved: {summary_output.name}")

# ================================================================
# STEP 7: Create Visualization
# ================================================================
print("\n" + "="*80)
print("🎨 [Step 7/7] CREATING VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Statement size distribution
ax1 = axes[0, 0]
statement_sizes = all_statements['n_invoices_in_statement'].value_counts().sort_index()
ax1.bar(statement_sizes.index[:30], statement_sizes.values[:30], 
        color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_title('Statement Size Distribution (Top 30)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Invoices per Statement', fontsize=12)
ax1.set_ylabel('Number of Statements', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Statements per decile
ax2 = axes[0, 1]
decile_counts = all_statements['decile'].value_counts().sort_index()
ax2.bar(decile_counts.index, decile_counts.values, 
        color='coral', edgecolor='black', alpha=0.7)
ax2.set_title('Number of Statements by Decile', fontsize=14, fontweight='bold')
ax2.set_xlabel('Decile', fontsize=12)
ax2.set_ylabel('Number of Statements', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Average statement value by decile
ax3 = axes[1, 0]
avg_values = all_statements.groupby('decile')['statement_discounted_value'].mean()
ax3.bar(avg_values.index, avg_values.values, 
        color='#70AD47', edgecolor='black', alpha=0.7)
ax3.set_title('Average Statement Value by Decile', fontsize=14, fontweight='bold')
ax3.set_xlabel('Decile', fontsize=12)
ax3.set_ylabel('Average Statement Value ($)', fontsize=12)
ax3.grid(True, alpha=0.3, axis='y')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
for i, v in enumerate(avg_values.values):
    ax3.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=45)

# Plot 4: Customer type distribution
ax4 = axes[1, 1]
customer_counts = all_statements['customer_type'].value_counts()
colors_pie = ['#4472C4', '#FFC000']
ax4.pie(customer_counts.values, labels=customer_counts.index, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax4.set_title(f'Statements by Customer Type\nTotal: {len(all_statements):,} statements',
              fontsize=14, fontweight='bold')

plt.tight_layout()
viz_output = data_cleaning_dir / '9.7_bundling_statistics_bundled.png'
plt.savefig(viz_output, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_output.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ INVOICE BUNDLING COMPLETE!")
print("="*80)

print(f"\n📊 Summary Statistics:")
print(f"  Total invoices processed: {len(ats_bundled) + len(invoice_bundled):,}")
print(f"    ATS invoices: {len(ats_bundled):,}")
print(f"    Invoice invoices: {len(invoice_bundled):,}")

print(f"\n  Total statements created: {len(all_statements):,}")
print(f"    ATS statements: {len(ats_statements):,}")
print(f"    Invoice statements: {len(invoice_statements):,}")

print(f"\n📋 Statement Metrics:")
print(f"  Average invoices per statement: {all_statements['n_invoices_in_statement'].mean():.2f}")
print(f"  Median invoices per statement: {all_statements['n_invoices_in_statement'].median():.0f}")
print(f"  Max invoices in a statement: {all_statements['n_invoices_in_statement'].max():,}")
print(f"  Min invoices in a statement: {all_statements['n_invoices_in_statement'].min():,}")

print(f"\n💰 Value Metrics:")
print(f"  Total value (all statements): ${all_statements['statement_discounted_value'].sum():,.2f}")
print(f"  Average statement value: ${all_statements['statement_discounted_value'].mean():,.2f}")
print(f"  Median statement value: ${all_statements['statement_discounted_value'].median():.2f}")

print(f"\n📁 Output Files:")
print(f"  • ats_grouped_transformed_with_discounts_bundled.csv")
print(f"  • invoice_grouped_transformed_with_discounts_bundled.csv")
print(f"  • 9.7_statement_summary_bundled.csv")
print(f"  • 9.7_bundling_statistics_bundled.png")

print(f"\n  All files saved to appropriate directories")

print("\n" + "="*80)
print("NEXT STEP: Use bundled data in Monte Carlo simulation (10.8)")
print("           Replace individual invoice loading with statement-bundled data")
print("="*80)

print(f"\n✓ Verification:")
print(f"  All ATS invoices bundled: {len(ats_bundled) == len(ats_grouped)}")
print(f"  All Invoice invoices bundled: {len(invoice_bundled) == len(invoice_grouped)}")
print(f"  All invoices assigned to statements: {ats_bundled['statement_id'].notna().all() and invoice_bundled['statement_id'].notna().all()}")
print(f"  All statements assigned to deciles: {all_statements['decile'].notna().all()}")