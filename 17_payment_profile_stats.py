"""
Script: 17_payment_profile_stats.py

Purpose:
    Analyze payment profile statistics from scripts 09.3 and 09.3.1 to determine
    the average overall late payment rate.

Inputs:
    - decile_payment_profile_summary.csv (from 09.3)
    - decile_assignments.csv (from 09.3)
    - 09.3.1_summary.csv (from 09.3.1)
    - master_dataset_complete.csv (original data)

Outputs:
    - Console output with late payment statistics
    - 17_payment_profile_summary_stats.csv (summary statistics)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories (using same format as other scripts)
base_dir = Path("T:/projects/2025/RuralCo")
ruralco3_dir = base_dir / "Data provided by RuralCo 20251202/RuralCo3"
profile_dir = ruralco3_dir / "payment_profile"
data_cleaning_dir = ruralco3_dir / "data_cleaning"

# Create output directory if it doesn't exist
profile_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PAYMENT PROFILE STATISTICS ANALYSIS")
print("="*70)

# ================================================================
# LOAD PAYMENT PROFILE DATA FROM 09.3
# ================================================================
print(f"\n{'='*70}")
print(f"LOADING PAYMENT PROFILE DATA FROM 09.3")
print(f"{'='*70}")

# Load decile payment profile summary
decile_summary_file = profile_dir / 'decile_payment_profile_summary.csv'
if decile_summary_file.exists():
    decile_summary_df = pd.read_csv(decile_summary_file)
    print(f"✓ Loaded decile summary: {len(decile_summary_df):,} deciles")
    print(f"Columns: {decile_summary_df.columns.tolist()}")
else:
    print(f"❌ Error: {decile_summary_file} not found")
    decile_summary_df = None

# Load decile assignments
decile_assignments_file = profile_dir / 'decile_assignments.csv'
if decile_assignments_file.exists():
    decile_assignments_df = pd.read_csv(decile_assignments_file)
    print(f"✓ Loaded decile assignments: {len(decile_assignments_df):,} accounts")
    print(f"Columns: {decile_assignments_df.columns.tolist()}")
else:
    print(f"❌ Error: {decile_assignments_file} not found")
    decile_assignments_df = None

# ================================================================
# LOAD DATA FROM 09.3.1
# ================================================================
print(f"\n{'='*70}")
print(f"LOADING LATE PAYMENT ANALYSIS FROM 09.3.1")
print(f"{'='*70}")

summary_091_file = profile_dir / '09.3.1_summary.csv'
if summary_091_file.exists():
    summary_091_df = pd.read_csv(summary_091_file)
    print(f"✓ Loaded 09.3.1 summary: {len(summary_091_df):,} rows")
    print(f"Columns: {summary_091_df.columns.tolist()}")
else:
    print(f"❌ Warning: {summary_091_file} not found")
    summary_091_df = None

# ================================================================
# LOAD MASTER DATASET FOR VERIFICATION
# ================================================================
print(f"\n{'='*70}")
print(f"LOADING MASTER DATASET FOR VERIFICATION")
print(f"{'='*70}")

master_file = profile_dir / 'master_dataset_complete.csv'
if master_file.exists():
    master_df = pd.read_csv(master_file)
    print(f"✓ Loaded master dataset: {len(master_df):,} accounts")
else:
    print(f"❌ Warning: {master_file} not found")
    master_df = None

# ================================================================
# CALCULATE AVERAGE OVERALL LATE PAYMENT RATE
# ================================================================
print(f"\n{'='*70}")
print(f"CALCULATING OVERALL LATE PAYMENT RATES")
print(f"{'='*70}")

# Method 1: From decile summary (weighted by number of accounts per decile)
if decile_summary_df is not None:
    # Check if we have the required columns
    if 'prob_late_pct' in decile_summary_df.columns and 'n_accounts' in decile_summary_df.columns:
        # Ensure numeric types and handle missing values
        decile_summary_df['prob_late_pct'] = pd.to_numeric(decile_summary_df['prob_late_pct'], errors='coerce').fillna(0)
        decile_summary_df['n_accounts'] = pd.to_numeric(decile_summary_df['n_accounts'], errors='coerce').fillna(0)
        
        total_accounts_deciles = int(decile_summary_df['n_accounts'].sum())
        
        if total_accounts_deciles > 0:
            weighted_late_rate_deciles = (decile_summary_df['prob_late_pct'] * decile_summary_df['n_accounts']).sum() / total_accounts_deciles
            print(f"📊 METHOD 1 - From Decile Summary:")
            print(f"  • Total accounts in deciles: {total_accounts_deciles:,}")
            print(f"  • Weighted average late payment rate: {weighted_late_rate_deciles:.2f}%")
        else:
            print(f"❌ METHOD 1 - No accounts found in decile summary")
            weighted_late_rate_deciles = None
            total_accounts_deciles = 0
            
    elif 'n_late' in decile_summary_df.columns and 'n_accounts' in decile_summary_df.columns:
        # Ensure numeric types and handle missing values
        decile_summary_df['n_late'] = pd.to_numeric(decile_summary_df['n_late'], errors='coerce').fillna(0)
        decile_summary_df['n_accounts'] = pd.to_numeric(decile_summary_df['n_accounts'], errors='coerce').fillna(0)
        
        total_accounts_deciles = int(decile_summary_df['n_accounts'].sum())
        total_late_accounts = int(decile_summary_df['n_late'].sum())
        
        if total_accounts_deciles > 0:
            late_rate_deciles = (total_late_accounts / total_accounts_deciles) * 100
            print(f"📊 METHOD 1 - From Decile Summary:")
            print(f"  • Total accounts in deciles: {total_accounts_deciles:,}")
            print(f"  • Total late accounts: {total_late_accounts:,}")
            print(f"  • Overall late payment rate: {late_rate_deciles:.2f}%")
            weighted_late_rate_deciles = late_rate_deciles
        else:
            print(f"❌ METHOD 1 - No accounts found in decile summary")
            weighted_late_rate_deciles = None
            total_accounts_deciles = 0
    else:
        print(f"❌ METHOD 1 - Missing required columns in decile summary")
        print(f"Available columns: {decile_summary_df.columns.tolist()}")
        weighted_late_rate_deciles = None
        total_accounts_deciles = 0
else:
    weighted_late_rate_deciles = None
    total_accounts_deciles = 0

# Method 2: From individual account assignments
if decile_assignments_df is not None and 'Late' in decile_assignments_df.columns:
    total_accounts_assignments = len(decile_assignments_df)
    late_accounts_assignments = (decile_assignments_df['Late'] == 1).sum()
    late_rate_assignments = (late_accounts_assignments / total_accounts_assignments) * 100
    print(f"\n📊 METHOD 2 - From Individual Account Assignments:")
    print(f"  • Total accounts: {total_accounts_assignments:,}")
    print(f"  • Late accounts (Late=1): {late_accounts_assignments:,}")
    print(f"  • Late payment rate: {late_rate_assignments:.2f}%")
else:
    late_rate_assignments = None
    total_accounts_assignments = 0
    late_accounts_assignments = 0

# Method 3: From master dataset (verification)
if master_df is not None:
    if 'Late' in master_df.columns:
        total_accounts_master = len(master_df)
        late_accounts_master = (master_df['Late'] == 1).sum()
        late_rate_master = (late_accounts_master / total_accounts_master) * 100
        print(f"\n📊 METHOD 3 - From Master Dataset (Verification):")
        print(f"  • Total accounts: {total_accounts_master:,}")
        print(f"  • Late accounts (Late=1): {late_accounts_master:,}")
        print(f"  • Late payment rate: {late_rate_master:.2f}%")
    else:
        late_rate_master = None
        total_accounts_master = 0
        late_accounts_master = 0
else:
    late_rate_master = None
    total_accounts_master = 0
    late_accounts_master = 0

# Method 4: From 09.3.1 summary (if available)
if summary_091_df is not None:
    print(f"\n📊 METHOD 4 - From 09.3.1 Summary:")
    print(summary_091_df.to_string(index=False))
    
    # Try to extract late rate from summary
    if 'late_rate_pct' in summary_091_df.columns:
        late_rate_091 = summary_091_df['late_rate_pct'].iloc[0] if len(summary_091_df) > 0 else None
        print(f"  • Late payment rate from 09.3.1: {late_rate_091:.2f}%")
    else:
        late_rate_091 = None
else:
    late_rate_091 = None

# ================================================================
# ANALYZE DELINQUENCY LEVELS FOR LATE ACCOUNTS
# ================================================================
print(f"\n{'='*70}")
print(f"ANALYZING DELINQUENCY LEVELS FOR LATE ACCOUNTS")
print(f"{'='*70}")

# Initialize variables
mean_cd_late = None
median_cd_late = None
late_accounts_with_cd = 0
total_late_for_cd_analysis = 0

# Try to get cd data from decile assignments first (most processed)
if decile_assignments_df is not None and 'Late' in decile_assignments_df.columns and 'cd' in decile_assignments_df.columns:
    print(f"📊 DELINQUENCY ANALYSIS - From Account Assignments:")
    
    # Filter to late accounts only
    late_accounts_df = decile_assignments_df[decile_assignments_df['Late'] == 1].copy()
    total_late_for_cd_analysis = len(late_accounts_df)
    
    if total_late_for_cd_analysis > 0:
        # Convert cd to numeric and handle missing values
        late_accounts_df['cd'] = pd.to_numeric(late_accounts_df['cd'], errors='coerce')
        late_accounts_with_cd_data = late_accounts_df['cd'].notna()
        late_accounts_with_cd = late_accounts_with_cd_data.sum()
        
        if late_accounts_with_cd > 0:
            cd_values = late_accounts_df.loc[late_accounts_with_cd_data, 'cd']
            mean_cd_late = cd_values.mean()
            median_cd_late = cd_values.median()
            
            print(f"  • Total late accounts: {total_late_for_cd_analysis:,}")
            print(f"  • Late accounts with cd data: {late_accounts_with_cd:,}")
            print(f"  • Mean delinquency level (cd): {mean_cd_late:.2f}")
            print(f"  • Median delinquency level (cd): {median_cd_late:.2f}")
            print(f"  • CD range for late accounts: {cd_values.min():.2f} - {cd_values.max():.2f}")
            print(f"  • CD standard deviation: {cd_values.std():.2f}")
        else:
            print(f"  • No cd data available for late accounts")
    else:
        print(f"  • No late accounts found")

# If not available from assignments, try master dataset
elif master_df is not None and 'Late' in master_df.columns and 'cd' in master_df.columns:
    print(f"📊 DELINQUENCY ANALYSIS - From Master Dataset:")
    
    # Filter to late accounts only
    late_accounts_master = master_df[master_df['Late'] == 1].copy()
    total_late_for_cd_analysis = len(late_accounts_master)
    
    if total_late_for_cd_analysis > 0:
        # Convert cd to numeric and handle missing values
        late_accounts_master['cd'] = pd.to_numeric(late_accounts_master['cd'], errors='coerce')
        late_accounts_with_cd_data = late_accounts_master['cd'].notna()
        late_accounts_with_cd = late_accounts_with_cd_data.sum()
        
        if late_accounts_with_cd > 0:
            cd_values = late_accounts_master.loc[late_accounts_with_cd_data, 'cd']
            mean_cd_late = cd_values.mean()
            median_cd_late = cd_values.median()
            
            print(f"  • Total late accounts: {total_late_for_cd_analysis:,}")
            print(f"  • Late accounts with cd data: {late_accounts_with_cd:,}")
            print(f"  • Mean delinquency level (cd): {mean_cd_late:.2f}")
            print(f"  • Median delinquency level (cd): {median_cd_late:.2f}")
            print(f"  • CD range for late accounts: {cd_values.min():.2f} - {cd_values.max():.2f}")
            print(f"  • CD standard deviation: {cd_values.std():.2f}")
        else:
            print(f"  • No cd data available for late accounts")
    else:
        print(f"  • No late accounts found")

else:
    print(f"❌ Cannot analyze delinquency levels - missing Late or cd columns in available data")

# ================================================================
# CONSOLIDATE RESULTS
# ================================================================
print(f"\n{'='*70}")
print(f"CONSOLIDATED LATE PAYMENT RATE ANALYSIS")
print(f"{'='*70}")

# Collect all valid rates
valid_rates = []
method_names = []

if weighted_late_rate_deciles is not None:
    valid_rates.append(weighted_late_rate_deciles)
    method_names.append("Decile Summary")

if late_rate_assignments is not None:
    valid_rates.append(late_rate_assignments)
    method_names.append("Account Assignments")

if late_rate_master is not None:
    valid_rates.append(late_rate_master)
    method_names.append("Master Dataset")

if late_rate_091 is not None:
    valid_rates.append(late_rate_091)
    method_names.append("09.3.1 Analysis")

if valid_rates:
    avg_late_rate = np.mean(valid_rates)
    min_late_rate = np.min(valid_rates)
    max_late_rate = np.max(valid_rates)
    
    print(f"🎯 FINAL RESULTS:")
    print(f"  • Number of calculation methods: {len(valid_rates)}")
    print(f"  • Average late payment rate: {avg_late_rate:.2f}%")
    print(f"  • Range: {min_late_rate:.2f}% - {max_late_rate:.2f}%")
    print(f"  • Standard deviation: {np.std(valid_rates):.2f}%")
    
    print(f"\n📋 BREAKDOWN BY METHOD:")
    for i, (method, rate) in enumerate(zip(method_names, valid_rates)):
        print(f"  • {method}: {rate:.2f}%")
        
    # Choose the most reliable method (prefer assignments if available)
    if late_rate_assignments is not None:
        recommended_rate = late_rate_assignments
        recommended_method = "Account Assignments (most granular)"
    elif weighted_late_rate_deciles is not None:
        recommended_rate = weighted_late_rate_deciles
        recommended_method = "Decile Summary (weighted)"
    else:
        recommended_rate = avg_late_rate
        recommended_method = "Average of available methods"
    
    print(f"\n⭐ RECOMMENDED RATE:")
    print(f"  • {recommended_rate:.2f}% ({recommended_method})")
    
else:
    print(f"❌ Could not calculate late payment rate - no valid data found")
    avg_late_rate = None
    recommended_rate = None
    recommended_method = "No data available"

# ================================================================
# CREATE SUMMARY DATAFRAME
# ================================================================
summary_stats = {
    'metric': [
        'Total accounts (decile summary)',
        'Total accounts (assignments)',
        'Total accounts (master dataset)',
        'Late accounts (assignments)',
        'Late accounts (master dataset)',
        'Late rate - Decile summary',
        'Late rate - Account assignments',
        'Late rate - Master dataset',
        'Late rate - 09.3.1 analysis',
        'Average late rate across methods',
        'Recommended late payment rate',
        'Late accounts with cd data',
        'Mean delinquency level (cd) for late accounts',
        'Median delinquency level (cd) for late accounts'
    ],
    'value': [
        total_accounts_deciles,
        total_accounts_assignments,
        total_accounts_master,
        late_accounts_assignments,
        late_accounts_master,
        round(weighted_late_rate_deciles, 2) if weighted_late_rate_deciles is not None else 'N/A',
        round(late_rate_assignments, 2) if late_rate_assignments is not None else 'N/A',
        round(late_rate_master, 2) if late_rate_master is not None else 'N/A',
        round(late_rate_091, 2) if late_rate_091 is not None else 'N/A',
        round(avg_late_rate, 2) if avg_late_rate is not None else 'N/A',
        round(recommended_rate, 2) if recommended_rate is not None else 'N/A',
        late_accounts_with_cd,
        round(mean_cd_late, 2) if mean_cd_late is not None else 'N/A',
        round(median_cd_late, 2) if median_cd_late is not None else 'N/A'
    ],
    'unit': [
        'accounts',
        'accounts',
        'accounts',
        'accounts',
        'accounts',
        '%',
        '%',
        '%',
        '%',
        '%',
        '%',
        'accounts',
        'cd level',
        'cd level'
    ]
}

summary_df = pd.DataFrame(summary_stats)

# ================================================================
# SAVE RESULTS
# ================================================================
output_file = profile_dir / '17_payment_profile_summary_stats.csv'
summary_df.to_csv(output_file, index=False)

print(f"\n{'='*70}")
print(f"RESULTS SAVED")
print(f"{'='*70}")
print(f"✓ Summary statistics saved to: {output_file}")

print(f"\n📋 COMPLETE SUMMARY TABLE:")
print(summary_df.to_string(index=False))

print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE!")
print(f"{'='*70}")

# ================================================================
# RETURN KEY VALUES
# ================================================================
print(f"\n🔢 KEY RETURN VALUES:")
if recommended_rate is not None:
    print(f"  • recommended_late_payment_rate = {recommended_rate:.2f}%")
    print(f"  • method = {recommended_method}")
if avg_late_rate is not None:
    print(f"  • average_across_methods = {avg_late_rate:.2f}%")
    print(f"  • number_of_methods = {len(valid_rates)}")
print(f"  • total_accounts_analyzed = {max(total_accounts_assignments, total_accounts_master, total_accounts_deciles)}")

# Add delinquency statistics
if mean_cd_late is not None and median_cd_late is not None:
    print(f"  • mean_delinquency_level_late_accounts = {mean_cd_late:.2f}")
    print(f"  • median_delinquency_level_late_accounts = {median_cd_late:.2f}")
    print(f"  • late_accounts_with_cd_data = {late_accounts_with_cd:,}")
    print(f"  • total_late_accounts_for_cd_analysis = {total_late_for_cd_analysis:,}")
else:
    print(f"  • delinquency_analysis = Not available (missing cd data)")
