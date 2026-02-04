"""
Script: 17.1_payment_profile_stats_cont.py

Purpose:
    Generate detailed decile-level analysis of payment profiles, including:
    1. Late payment rates by decile
    2. Mean and median delinquency levels (cd) by decile

Inputs:
    - decile_payment_profile_summary.csv (from 09.3)
    - decile_assignments.csv (from 09.3)

Outputs:
    - Console output with decile-level tables
    - 17.1_late_rate_by_decile.csv
    - 17.1_cd_levels_by_decile.csv
    - 17.1_combined_decile_analysis.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories (using same format as other scripts)
base_dir = Path("T:/projects/2025/RuralCo")
ruralco3_dir = base_dir / "Data provided by RuralCo 20251202/RuralCo3"
profile_dir = ruralco3_dir / "payment_profile"

# Create output directory if it doesn't exist
profile_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PAYMENT PROFILE DECILE-LEVEL ANALYSIS")
print("="*80)

# ================================================================
# LOAD PAYMENT PROFILE DATA
# ================================================================
print(f"\n{'='*80}")
print(f"LOADING PAYMENT PROFILE DATA")
print(f"{'='*80}")

# Load decile payment profile summary
decile_summary_file = profile_dir / 'decile_payment_profile_summary.csv'
if decile_summary_file.exists():
    decile_summary_df = pd.read_csv(decile_summary_file)
    print(f"✓ Loaded decile summary: {len(decile_summary_df):,} deciles")
    print(f"Columns: {decile_summary_df.columns.tolist()}")
else:
    print(f"❌ Error: {decile_summary_file} not found")
    exit(1)

# Load decile assignments
decile_assignments_file = profile_dir / 'decile_assignments.csv'
if decile_assignments_file.exists():
    decile_assignments_df = pd.read_csv(decile_assignments_file)
    print(f"✓ Loaded decile assignments: {len(decile_assignments_df):,} accounts")
    print(f"Columns: {decile_assignments_df.columns.tolist()}")
else:
    print(f"❌ Error: {decile_assignments_file} not found")
    exit(1)

# ================================================================
# ANALYZE LATE RATES BY DECILE
# ================================================================
print(f"\n{'='*80}")
print(f"ANALYZING LATE RATES BY DECILE")
print(f"{'='*80}")

# Method 1: From decile summary (if available)
late_rate_by_decile_summary = None
if 'decile' in decile_summary_df.columns:
    print(f"📊 LATE RATES FROM DECILE SUMMARY:")
    
    # Check for the rate column (could be prob_late_pct or calculated from n_late/n_accounts)
    if 'prob_late_pct' in decile_summary_df.columns:
        late_rate_by_decile_summary = decile_summary_df[['decile', 'prob_late_pct', 'n_accounts']].copy()
        late_rate_by_decile_summary = late_rate_by_decile_summary.rename(columns={'prob_late_pct': 'late_rate_pct'})
        
    elif 'n_late' in decile_summary_df.columns and 'n_accounts' in decile_summary_df.columns:
        late_rate_by_decile_summary = decile_summary_df[['decile', 'n_late', 'n_accounts']].copy()
        # Calculate late rate percentage
        late_rate_by_decile_summary['late_rate_pct'] = (
            late_rate_by_decile_summary['n_late'] / late_rate_by_decile_summary['n_accounts'] * 100
        )
    
    if late_rate_by_decile_summary is not None:
        late_rate_by_decile_summary = late_rate_by_decile_summary.sort_values('decile')
        print(late_rate_by_decile_summary.to_string(index=False))

# Method 2: Calculate from individual account assignments
print(f"\n📊 LATE RATES FROM INDIVIDUAL ACCOUNT DATA:")

if 'decile' in decile_assignments_df.columns and 'Late' in decile_assignments_df.columns:
    # Group by decile and calculate late rates
    decile_stats = decile_assignments_df.groupby('decile').agg({
        'Late': ['count', 'sum', 'mean']
    }).round(4)
    
    # Flatten column names
    decile_stats.columns = ['total_accounts', 'late_accounts', 'late_rate']
    decile_stats['late_rate_pct'] = decile_stats['late_rate'] * 100
    decile_stats = decile_stats.reset_index()
    
    print(decile_stats[['decile', 'total_accounts', 'late_accounts', 'late_rate_pct']].to_string(index=False))
    
    # Store for later use
    late_rate_by_decile_detailed = decile_stats.copy()
else:
    print(f"❌ Cannot calculate from assignments - missing decile or Late columns")
    late_rate_by_decile_detailed = None

# ================================================================
# ANALYZE CD LEVELS BY DECILE
# ================================================================
print(f"\n{'='*80}")
print(f"ANALYZING DELINQUENCY LEVELS (CD) BY DECILE")
print(f"{'='*80}")

cd_by_decile = None
if 'decile' in decile_assignments_df.columns and 'cd' in decile_assignments_df.columns:
    print(f"📊 CD LEVELS BY DECILE:")
    
    # Convert cd to numeric
    decile_assignments_df['cd_numeric'] = pd.to_numeric(decile_assignments_df['cd'], errors='coerce')
    
    # Group by decile and calculate cd statistics
    cd_by_decile = decile_assignments_df.groupby('decile')['cd_numeric'].agg([
        'count',
        'mean',
        'median', 
        'std',
        'min',
        'max'
    ]).round(2)
    
    # Reset index to make decile a column
    cd_by_decile = cd_by_decile.reset_index()
    cd_by_decile.columns = ['decile', 'accounts_with_cd', 'mean_cd', 'median_cd', 'std_cd', 'min_cd', 'max_cd']
    
    print(cd_by_decile[['decile', 'accounts_with_cd', 'mean_cd', 'median_cd']].to_string(index=False))
    
    print(f"\n📊 DETAILED CD STATISTICS BY DECILE:")
    print(cd_by_decile.to_string(index=False))

else:
    print(f"❌ Cannot analyze CD levels - missing decile or cd columns")

# ================================================================
# ANALYZE CD LEVELS FOR LATE ACCOUNTS BY DECILE
# ================================================================
print(f"\n{'='*80}")
print(f"ANALYZING CD LEVELS FOR LATE ACCOUNTS BY DECILE")
print(f"{'='*80}")

cd_late_by_decile = None
if ('decile' in decile_assignments_df.columns and 
    'cd' in decile_assignments_df.columns and 
    'Late' in decile_assignments_df.columns):
    
    print(f"📊 CD LEVELS FOR LATE ACCOUNTS BY DECILE:")
    
    # Filter to only late accounts
    late_accounts_only = decile_assignments_df[decile_assignments_df['Late'] == 1].copy()
    
    if len(late_accounts_only) > 0:
        # Group by decile and calculate cd statistics for late accounts only
        cd_late_by_decile = late_accounts_only.groupby('decile')['cd_numeric'].agg([
            'count',
            'mean',
            'median',
            'std',
            'min',
            'max'
        ]).round(2)
        
        cd_late_by_decile = cd_late_by_decile.reset_index()
        cd_late_by_decile.columns = ['decile', 'late_accounts_with_cd', 'mean_cd_late', 'median_cd_late', 
                                     'std_cd_late', 'min_cd_late', 'max_cd_late']
        
        print(cd_late_by_decile[['decile', 'late_accounts_with_cd', 'mean_cd_late', 'median_cd_late']].to_string(index=False))
    else:
        print(f"❌ No late accounts found")

# ================================================================
# CREATE COMBINED ANALYSIS TABLE
# ================================================================
print(f"\n{'='*80}")
print(f"CREATING COMBINED DECILE ANALYSIS")
print(f"{'='*80}")

# Start with the detailed late rate analysis
if late_rate_by_decile_detailed is not None:
    combined_analysis = late_rate_by_decile_detailed[['decile', 'total_accounts', 'late_accounts', 'late_rate_pct']].copy()
    
    # Merge with overall CD statistics
    if cd_by_decile is not None:
        combined_analysis = combined_analysis.merge(
            cd_by_decile[['decile', 'mean_cd', 'median_cd']], 
            on='decile', how='left'
        )
    
    # Merge with late-only CD statistics  
    if cd_late_by_decile is not None:
        combined_analysis = combined_analysis.merge(
            cd_late_by_decile[['decile', 'mean_cd_late', 'median_cd_late']],
            on='decile', how='left'
        )
    
    print(f"📊 COMBINED DECILE ANALYSIS:")
    print(combined_analysis.to_string(index=False))
    
    # Calculate some additional insights
    print(f"\n🎯 KEY INSIGHTS BY DECILE:")
    for _, row in combined_analysis.iterrows():
        decile = int(row['decile'])
        late_rate = row['late_rate_pct']
        total_accounts = int(row['total_accounts'])
        
        print(f"  Decile {decile}:")
        print(f"    • {total_accounts:,} accounts, {late_rate:.1f}% late rate")
        
        if pd.notna(row.get('mean_cd')):
            print(f"    • Mean CD (all accounts): {row['mean_cd']:.2f}")
        if pd.notna(row.get('mean_cd_late')):
            print(f"    • Mean CD (late accounts only): {row['mean_cd_late']:.2f}")
        print()

else:
    combined_analysis = None
    print(f"❌ Cannot create combined analysis - missing base data")

# ================================================================
# SAVE RESULTS
# ================================================================
print(f"\n{'='*80}")
print(f"SAVING RESULTS")
print(f"{'='*80}")

# Save late rate by decile
if late_rate_by_decile_detailed is not None:
    late_rate_output = profile_dir / '17.1_late_rate_by_decile.csv'
    late_rate_by_decile_detailed.to_csv(late_rate_output, index=False)
    print(f"✓ Late rate by decile saved to: {late_rate_output}")

# Save CD levels by decile
if cd_by_decile is not None:
    cd_output = profile_dir / '17.1_cd_levels_by_decile.csv'
    cd_by_decile.to_csv(cd_output, index=False)
    print(f"✓ CD levels by decile saved to: {cd_output}")

# Save combined analysis
if combined_analysis is not None:
    combined_output = profile_dir / '17.1_combined_decile_analysis.csv'
    combined_analysis.to_csv(combined_output, index=False)
    print(f"✓ Combined decile analysis saved to: {combined_output}")

# Save late-only CD analysis if available
if cd_late_by_decile is not None:
    cd_late_output = profile_dir / '17.1_cd_levels_late_accounts_by_decile.csv'
    cd_late_by_decile.to_csv(cd_late_output, index=False)
    print(f"✓ CD levels for late accounts by decile saved to: {cd_late_output}")

print(f"\n{'='*80}")
print(f"DECILE-LEVEL ANALYSIS COMPLETE!")
print(f"{'='*80}")

# ================================================================
# SUMMARY STATISTICS
# ================================================================
if combined_analysis is not None:
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"  • Number of deciles analyzed: {len(combined_analysis)}")
    print(f"  • Total accounts across all deciles: {combined_analysis['total_accounts'].sum():,}")
    print(f"  • Total late accounts: {combined_analysis['late_accounts'].sum():,}")
    print(f"  • Overall late rate: {(combined_analysis['late_accounts'].sum() / combined_analysis['total_accounts'].sum() * 100):.2f}%")
    
    if 'mean_cd' in combined_analysis.columns:
        print(f"  • Decile with highest late rate: {combined_analysis.loc[combined_analysis['late_rate_pct'].idxmax(), 'decile']:.0f} ({combined_analysis['late_rate_pct'].max():.1f}%)")
        print(f"  • Decile with lowest late rate: {combined_analysis.loc[combined_analysis['late_rate_pct'].idxmin(), 'decile']:.0f} ({combined_analysis['late_rate_pct'].min():.1f}%)")
        
        if pd.notna(combined_analysis['mean_cd']).any():
            valid_cd = combined_analysis.dropna(subset=['mean_cd'])
            if len(valid_cd) > 0:
                print(f"  • Decile with highest mean CD: {valid_cd.loc[valid_cd['mean_cd'].idxmax(), 'decile']:.0f} ({valid_cd['mean_cd'].max():.2f})")
                print(f"  • Decile with lowest mean CD: {valid_cd.loc[valid_cd['mean_cd'].idxmin(), 'decile']:.0f} ({valid_cd['mean_cd'].min():.2f})")
