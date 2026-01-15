"""
09.3_Payment_profile_ML_clustering.py
========================================
Create payment behavior profile based on InvoiceAmount deciles.

MODIFICATION: For late accounts, directly sample cd level distribution
from the actual cd values present in late accounts for that decile.

Strategy:
1. Order accounts by InvoiceAmount
2. Create 10 equal-sized deciles
3. For each decile:
   - P(late) = probability Late = 1
   - P(cd = k | late) = actual distribution of cd values in late accounts

This can be directly mapped to invoice data by invoice amount.

Author: Chris
Date: January 2026
Modified: January 2026 - Direct cd sampling from late accounts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
print(f"Working directory set to: {os.getcwd()}")

# ================================================================
# Configuration
# ================================================================
INPUT_FILE = r"Payment Profile\master_dataset_complete.csv"
OUTPUT_DIR = Path("Payment Profile")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Payment terms
PAYMENT_TERMS_MONTHS = 20 / 30  # Convert 20 days to months (~0.67 months)

# Number of deciles (10 = deciles, can change to 4 for quartiles, 5 for quintiles, etc.)
N_DECILES = 10

# ================================================================
# Load data
# ================================================================
print("="*70)
print("CREATING DECILE-BASED PAYMENT PROFILE")
print("="*70)
print(f"\nUsing 'Late' column to determine late payments (Late = 1)")
print(f"Payment terms reference: {PAYMENT_TERMS_MONTHS:.2f} months (~{PAYMENT_TERMS_MONTHS * 30:.0f} days)")
print(f"Number of groups: {N_DECILES} deciles")

df = pd.read_csv(INPUT_FILE)
print(f"\n✓ Loaded {len(df):,} records")

# ================================================================
# Validate required columns
# ================================================================
print("\n" + "="*70)
print("DATA VALIDATION")
print("="*70)

required_cols = ['InvoiceAmount', 'avg_time_between_payments', 'cd', 'Late']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"✗ ERROR: Missing required columns: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

print(f"✓ All required columns present")

# ================================================================
# Overall statistics
# ================================================================
print("\n" + "="*70)
print("OVERALL STATISTICS")
print("="*70)

total_accounts = len(df)
late_accounts = (df['Late'] == 1).sum()
late_pct = late_accounts / total_accounts * 100

print(f"\nTotal accounts: {total_accounts:,}")
print(f"Late payments (Late = 1): {late_accounts:,} ({late_pct:.1f}%)")
print(f"On-time payments (Late = 0): {total_accounts - late_accounts:,} ({100-late_pct:.1f}%)")

print(f"\nInvoiceAmount distribution:")
print(f"  Mean: ${df['InvoiceAmount'].mean():,.2f}")
print(f"  Median: ${df['InvoiceAmount'].median():,.2f}")
print(f"  Min: ${df['InvoiceAmount'].min():,.2f}")
print(f"  Max: ${df['InvoiceAmount'].max():,.2f}")

print(f"\nAvg time between payments (months):")
print(f"  Mean: {df['avg_time_between_payments'].mean():.2f} months")
print(f"  Median: {df['avg_time_between_payments'].median():.2f} months")
print(f"  Min: {df['avg_time_between_payments'].min():.2f} months")
print(f"  Max: {df['avg_time_between_payments'].max():.2f} months")

print(f"\nDelinquency level (cd) distribution:")
cd_dist = df['cd'].value_counts().sort_index()
for cd_level, count in cd_dist.items():
    print(f"  cd = {cd_level}: {count:,} ({count/total_accounts*100:.1f}%)")

# ================================================================
# Create deciles based on InvoiceAmount
# ================================================================
print("\n" + "="*70)
print("CREATING DECILES")
print("="*70)

# Create deciles using qcut (equal-sized groups)
df['decile'] = pd.qcut(df['InvoiceAmount'], 
                       q=N_DECILES, 
                       labels=False,  # Use numeric labels 0-9
                       duplicates='drop')

actual_n_deciles = df['decile'].nunique()
print(f"✓ Created {actual_n_deciles} deciles")

if actual_n_deciles < N_DECILES:
    print(f"⚠ Note: Created {actual_n_deciles} groups instead of {N_DECILES}")
    print(f"  (Due to duplicate InvoiceAmount values)")

# Show decile boundaries
print(f"\nDecile boundaries:")
decile_summary = df.groupby('decile')['InvoiceAmount'].agg(['min', 'max', 'count'])
for decile_num, row in decile_summary.iterrows():
    print(f"  Decile {decile_num}: ${row['min']:,.2f} - ${row['max']:,.2f} ({row['count']:,} accounts)")

# ================================================================
# Build payment profile for each decile
# ================================================================
print("\n" + "="*70)
print("BUILDING DECILE PAYMENT PROFILES")
print("="*70)

payment_profile = {
    'metadata': {
        'created_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_records': len(df),
        'payment_terms_months': PAYMENT_TERMS_MONTHS,
        'payment_terms_days': PAYMENT_TERMS_MONTHS * 30,
        'n_deciles': actual_n_deciles,
        'method': 'equal-sized deciles by InvoiceAmount with direct cd sampling',
        'metric': 'avg_time_between_payments (months)',
        'cd_sampling_method': 'Direct sampling from late accounts only'
    },
    'deciles': {}
}

summary_rows = []

for decile_num in sorted(df['decile'].unique()):
    decile_data = df[df['decile'] == decile_num].copy()
    n_accounts = len(decile_data)
    
    # Invoice amount range
    min_amount = decile_data['InvoiceAmount'].min()
    max_amount = decile_data['InvoiceAmount'].max()
    avg_amount = decile_data['InvoiceAmount'].mean()
    
    print(f"\n{'='*70}")
    print(f"DECILE {decile_num}: ${min_amount:,.2f} - ${max_amount:,.2f}")
    print(f"{'='*70}")
    print(f"Accounts: {n_accounts:,}")
    print(f"Average InvoiceAmount: ${avg_amount:,.2f}")
    
    # ================================================================
    # LATE PAYMENT PROBABILITY
    # ================================================================
    n_late = (decile_data['Late'] == 1).sum()
    prob_late = n_late / n_accounts
    
    print(f"\nPayment Timing:")
    print(f"  Late payments (Late = 1): {n_late:,} ({prob_late*100:.1f}%)")
    print(f"  On-time payments (Late = 0): {n_accounts - n_late:,} ({(1-prob_late)*100:.1f}%)")
    print(f"  Average time between payments: {decile_data['avg_time_between_payments'].mean():.2f} months")
    
    # Months overdue for late payments
    late_payments = decile_data[decile_data['Late'] == 1]
    if len(late_payments) > 0:
        months_overdue = late_payments['avg_time_between_payments'] - PAYMENT_TERMS_MONTHS
        avg_months_overdue = months_overdue.mean()
        median_months_overdue = months_overdue.median()
        print(f"  Avg months overdue (when late): {avg_months_overdue:.2f}")
        print(f"  Median months overdue (when late): {median_months_overdue:.2f}")
    else:
        avg_months_overdue = 0
        median_months_overdue = 0
    
    # ================================================================
    # DELINQUENCY LEVEL DISTRIBUTION (DIRECT SAMPLING FROM LATE ACCOUNTS)
    # ================================================================
    print(f"\nDelinquency Level Distribution (cd) | Given Late:")
    print(f"  [MODIFIED: Direct sampling from late accounts only]")
    
    # Overall cd distribution in this decile
    cd_distribution_all = decile_data['cd'].value_counts(normalize=True).sort_index()
    
    # cd distribution GIVEN late payment - directly from late accounts
    if n_late > 0:
        # Get cd values for late accounts only
        cd_values_late = late_payments['cd'].values
        
        # Count occurrences of each cd level
        cd_counts_late = late_payments['cd'].value_counts().sort_index()
        
        # Calculate probabilities by dividing counts by total late accounts
        cd_distribution_late = (cd_counts_late / n_late).sort_index()
        
        print(f"\n  {'cd':<4} {'Count (Late)':<15} {'Prob (Late)':<15} {'All Accounts':<15}")
        print(f"  {'-'*4} {'-'*15} {'-'*15} {'-'*15}")
        
        # Get all unique cd levels in late payments
        cd_levels_late = sorted(late_payments['cd'].unique())
        
        cd_given_late = {}
        for cd_level in cd_levels_late:
            count_late = cd_counts_late.get(cd_level, 0)
            prob_cd_late = cd_distribution_late.get(cd_level, 0)
            prob_all = cd_distribution_all.get(cd_level, 0)
            
            cd_given_late[int(cd_level)] = float(prob_cd_late)
            print(f"  {cd_level:<4} {count_late:>6}          {prob_cd_late*100:>6.1f}%        {prob_all*100:>6.1f}%")
        
        print(f"\n  Total late accounts: {n_late}")
        print(f"  Verification: Sum of probabilities = {sum(cd_given_late.values()):.4f}")
        
        # Additional verification: show raw counts
        print(f"\n  Raw cd counts in late accounts:")
        for cd_level in cd_levels_late:
            count = cd_counts_late.get(cd_level, 0)
            print(f"    cd = {cd_level}: {count} accounts")
        
    else:
        print(f"  No late payments in this decile")
        cd_given_late = {}
    
    # ================================================================
    # BUILD PROFILE OBJECT
    # ================================================================
    decile_profile = {
        'decile_number': int(decile_num),
        'invoice_amount_range': {
            'min': float(min_amount),
            'max': float(max_amount),
            'mean': float(avg_amount),
            'median': float(decile_data['InvoiceAmount'].median())
        },
        'n_accounts': int(n_accounts),
        'payment_behavior': {
            'prob_late': float(prob_late),
            'prob_on_time': float(1 - prob_late),
            'n_late': int(n_late),
            'n_on_time': int(n_accounts - n_late),
            'avg_time_between_payments_months': float(decile_data['avg_time_between_payments'].mean()),
            'median_time_between_payments_months': float(decile_data['avg_time_between_payments'].median()),
            'avg_months_overdue_given_late': float(avg_months_overdue),
            'median_months_overdue_given_late': float(median_months_overdue),
            'payment_terms_months': PAYMENT_TERMS_MONTHS
        },
        'delinquency_distribution': {
            'cd_given_late': cd_given_late,  # P(cd = k | late) - DIRECTLY SAMPLED FROM LATE ACCOUNTS
            'cd_overall': {int(k): float(v) for k, v in cd_distribution_all.items()},
            'sampling_method': 'direct_from_late_accounts'  # Flag for downstream use
        }
    }
    
    payment_profile['deciles'][f'decile_{decile_num}'] = decile_profile
    
    # ================================================================
    # ADD TO SUMMARY
    # ================================================================
    summary_row = {
        'decile': decile_num,
        'min_amount': f"${min_amount:,.2f}",
        'max_amount': f"${max_amount:,.2f}",
        'n_accounts': n_accounts,
        'n_late': n_late,
        'prob_late_pct': f"{prob_late*100:.1f}%",
        'avg_months_overdue': f"{avg_months_overdue:.2f}",
        'avg_time_between_payments': f"{decile_data['avg_time_between_payments'].mean():.2f}"
    }
    
    # Add most common cd level when late
    if cd_given_late:
        most_common_cd = max(cd_given_late, key=cd_given_late.get)
        summary_row['most_common_cd_when_late'] = f"cd={most_common_cd} ({cd_given_late[most_common_cd]*100:.1f}%)"
    else:
        summary_row['most_common_cd_when_late'] = "N/A"
    
    summary_rows.append(summary_row)

# ================================================================
# Save payment profile
# ================================================================
print("\n" + "="*70)
print("SAVING PAYMENT PROFILE")
print("="*70)

# Save pickle
output_file = OUTPUT_DIR / "decile_payment_profile_MODIFIED.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(payment_profile, f)
print(f"✓ Saved payment profile to: {output_file}")

# Save CSV summary
summary_df = pd.DataFrame(summary_rows)

summary_file = OUTPUT_DIR / "decile_payment_profile_summary_MODIFIED.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✓ Saved summary to: {summary_file}")

# Save decile assignments
decile_assignment_file = OUTPUT_DIR / "decile_assignments_MODIFIED.csv"
output_cols = ['InvoiceAmount', 'decile', 'avg_time_between_payments', 'cd', 'Late']
df[output_cols].to_csv(decile_assignment_file, index=False)
print(f"✓ Saved decile assignments to: {decile_assignment_file}")

# ================================================================
# Create detailed cd distribution analysis
# ================================================================
print("\n" + "="*70)
print("CD DISTRIBUTION ANALYSIS")
print("="*70)

cd_analysis = []
for decile_num in sorted(df['decile'].unique()):
    decile_profile = payment_profile['deciles'][f'decile_{decile_num}']
    cd_given_late = decile_profile['delinquency_distribution']['cd_given_late']
    
    for cd_level, prob in cd_given_late.items():
        cd_analysis.append({
            'decile': decile_num,
            'cd_level': cd_level,
            'probability': prob,
            'n_late_total': decile_profile['payment_behavior']['n_late']
        })

cd_analysis_df = pd.DataFrame(cd_analysis)
cd_analysis_file = OUTPUT_DIR / "cd_distribution_by_decile_MODIFIED.csv"
cd_analysis_df.to_csv(cd_analysis_file, index=False)
print(f"✓ Saved cd distribution analysis to: {cd_analysis_file}")

# ================================================================
# Summary
# ================================================================
print("\n" + "="*70)
print("DECILE PAYMENT PROFILE COMPLETE! (MODIFIED VERSION)")
print("="*70)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"\nKey files:")
print(f"  1. decile_payment_profile_MODIFIED.pkl - Profile for predictions")
print(f"  2. decile_payment_profile_summary_MODIFIED.csv - Human-readable summary")
print(f"  3. decile_assignments_MODIFIED.csv - All accounts with decile assignments")
print(f"  4. cd_distribution_by_decile_MODIFIED.csv - Detailed cd distribution")
print(f"\nProfile contains {actual_n_deciles} deciles with equal sample sizes")
print(f"Payment terms: {PAYMENT_TERMS_MONTHS:.2f} months (~{PAYMENT_TERMS_MONTHS * 30:.0f} days)")
print(f"\nMODIFICATION APPLIED:")
print(f"  - cd distributions are sampled DIRECTLY from late accounts only")
print(f"  - P(cd = k | late) uses only the cd values present in late payments")
print(f"  - Example: If 5 late accounts have cd=[3,3,3,4,4], then:")
print(f"            P(cd=3|late) = 3/5 = 60%")
print(f"            P(cd=4|late) = 2/5 = 40%")
print(f"\nFor each decile, you have:")
print(f"  - P(late): Probability of late payment (Late = 1)")
print(f"  - P(cd = k | late): cd level distribution FROM LATE ACCOUNTS ONLY")
print(f"  - Avg months overdue (when late)")
print(f"\nNext step: Map this profile to invoice data by invoice amount")