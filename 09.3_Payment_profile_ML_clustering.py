"""
09.5_Decile_payment_profile.py
===============================
Create payment behavior profile based on InvoiceAmount deciles.

Strategy:
1. Order accounts by InvoiceAmount
2. Create 10 equal-sized deciles
3. For each decile:
   - P(late) = probability payment_days > 20
   - P(cd = k | late) = distribution of delinquency levels given late

This can be directly mapped to invoice data by invoice amount.

Author: Chris
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

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
print(f"\nPayment terms: {PAYMENT_TERMS_MONTHS:.2f} months (~{PAYMENT_TERMS_MONTHS * 30:.0f} days)")
print(f"Late if avg_time_between_payments > {PAYMENT_TERMS_MONTHS:.2f} months")
print(f"Number of groups: {N_DECILES} deciles")

df = pd.read_csv(INPUT_FILE)
print(f"\n✓ Loaded {len(df):,} records")

# ================================================================
# Validate required columns
# ================================================================
print("\n" + "="*70)
print("DATA VALIDATION")
print("="*70)

required_cols = ['InvoiceAmount', 'avg_time_between_payments', 'cd']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"✗ ERROR: Missing required columns: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

print(f"✓ All required columns present")

# Optional columns
has_interest = 'ytd_interest' in df.columns
if has_interest:
    print(f"✓ ytd_interest column found - will include interest analysis")

# ================================================================
# Overall statistics
# ================================================================
print("\n" + "="*70)
print("OVERALL STATISTICS")
print("="*70)

total_accounts = len(df)
late_accounts = (df['avg_time_between_payments'] > PAYMENT_TERMS_MONTHS).sum()
late_pct = late_accounts / total_accounts * 100

print(f"\nTotal accounts: {total_accounts:,}")
print(f"Late payments (>{PAYMENT_TERMS_MONTHS:.2f} months between payments): {late_accounts:,} ({late_pct:.1f}%)")
print(f"On-time payments (≤{PAYMENT_TERMS_MONTHS:.2f} months): {total_accounts - late_accounts:,} ({100-late_pct:.1f}%)")

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

print(f"\nInvoiceAmount distribution:")
print(f"  Mean: ${df['InvoiceAmount'].mean():,.2f}")
print(f"  Median: ${df['InvoiceAmount'].median():,.2f}")
print(f"  Min: ${df['InvoiceAmount'].min():,.2f}")
print(f"  Max: ${df['InvoiceAmount'].max():,.2f}")

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
        'method': 'equal-sized deciles by InvoiceAmount',
        'metric': 'avg_time_between_payments (months)'
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
    is_late = decile_data['avg_time_between_payments'] > PAYMENT_TERMS_MONTHS
    n_late = is_late.sum()
    prob_late = is_late.mean()
    
    print(f"\nPayment Timing:")
    print(f"  Late payments (>{PAYMENT_TERMS_MONTHS:.2f} months): {n_late:,} ({prob_late*100:.1f}%)")
    print(f"  On-time payments: {n_accounts - n_late:,} ({(1-prob_late)*100:.1f}%)")
    print(f"  Average time between payments: {decile_data['avg_time_between_payments'].mean():.2f} months")
    
    # Months overdue for late payments
    late_payments = decile_data[is_late]
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
    # DELINQUENCY LEVEL DISTRIBUTION (CONDITIONAL ON LATE)
    # ================================================================
    print(f"\nDelinquency Level Distribution (cd) | Given Late:")
    
    # Overall cd distribution in this decile
    cd_distribution_all = decile_data['cd'].value_counts(normalize=True).sort_index()
    
    # cd distribution GIVEN late payment
    if n_late > 0:
        cd_distribution_late = late_payments['cd'].value_counts(normalize=True).sort_index()
        
        print(f"  {'cd':<4} {'All Accounts':<15} {'Late Only':<15}")
        print(f"  {'-'*4} {'-'*15} {'-'*15}")
        
        # Get all unique cd levels in this decile
        all_cd_levels = sorted(decile_data['cd'].unique())
        
        cd_given_late = {}
        for cd_level in all_cd_levels:
            prob_all = cd_distribution_all.get(cd_level, 0)
            prob_late = cd_distribution_late.get(cd_level, 0)
            cd_given_late[int(cd_level)] = float(prob_late)
            print(f"  {cd_level:<4} {prob_all*100:>6.1f}%        {prob_late*100:>6.1f}%")
    else:
        print(f"  No late payments in this decile")
        cd_given_late = {}
    
    # ================================================================
    # INTEREST CHARGES (if available)
    # ================================================================
    if has_interest:
        total_interest = decile_data['ytd_interest'].sum()
        avg_interest = decile_data['ytd_interest'].mean()
        n_with_interest = (decile_data['ytd_interest'] > 0).sum()
        prob_interest = n_with_interest / n_accounts
        
        print(f"\nInterest Charges:")
        print(f"  Total: ${total_interest:,.2f}")
        print(f"  Average per account: ${avg_interest:.2f}")
        print(f"  Accounts with interest: {n_with_interest:,} ({prob_interest*100:.1f}%)")
        
        if n_with_interest > 0:
            avg_interest_given_charged = decile_data[decile_data['ytd_interest'] > 0]['ytd_interest'].mean()
            print(f"  Average interest (when >0): ${avg_interest_given_charged:.2f}")
    
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
            'cd_given_late': cd_given_late,  # P(cd = k | late)
            'cd_overall': {int(k): float(v) for k, v in cd_distribution_all.items()}
        }
    }
    
    if has_interest:
        decile_profile['interest'] = {
            'total_interest': float(total_interest),
            'avg_interest': float(avg_interest),
            'prob_interest': float(prob_interest),
            'n_with_interest': int(n_with_interest)
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
    
    if has_interest:
        summary_row['avg_interest'] = f"${avg_interest:.2f}"
    
    summary_rows.append(summary_row)

# ================================================================
# Save payment profile
# ================================================================
print("\n" + "="*70)
print("SAVING PAYMENT PROFILE")
print("="*70)

# Save pickle
output_file = OUTPUT_DIR / "decile_payment_profile.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(payment_profile, f)
print(f"✓ Saved payment profile to: {output_file}")

# Save CSV summary
summary_df = pd.DataFrame(summary_rows)
summary_file = OUTPUT_DIR / "decile_payment_profile_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✓ Saved summary to: {summary_file}")

# Save decile assignments
decile_assignment_file = OUTPUT_DIR / "decile_assignments.csv"
output_cols = ['InvoiceAmount', 'decile', 'avg_time_between_payments', 'cd']
if has_interest:
    output_cols.append('ytd_interest')
df[output_cols].to_csv(decile_assignment_file, index=False)
print(f"✓ Saved decile assignments to: {decile_assignment_file}")

# # ================================================================
# # Create visualizations
# # ================================================================
# print("\n" + "="*70)
# print("CREATING VISUALIZATIONS")
# print("="*70)

# fig = plt.figure(figsize=(16, 12))
# gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# # Plot 1: Probability of late payment by decile
# ax1 = fig.add_subplot(gs[0, 0])
# prob_late_values = [float(row['prob_late_pct'].rstrip('%')) for _, row in summary_df.iterrows()]
# decile_labels = [f"D{d}" for d in summary_df['decile']]
# ax1.bar(range(len(summary_df)), prob_late_values, color='coral', alpha=0.7)
# ax1.set_xlabel('Decile (by InvoiceAmount)', fontsize=11)
# ax1.set_ylabel(f'Probability of Late Payment (%)', fontsize=11)
# ax1.set_title(f'P(Late | Decile) - Avg Time Between Payments >{PAYMENT_TERMS_MONTHS:.2f} months', fontsize=12, fontweight='bold')
# ax1.set_xticks(range(len(summary_df)))
# ax1.set_xticklabels(decile_labels)
# ax1.grid(True, alpha=0.3, axis='y')
# for i, v in enumerate(prob_late_values):
#     ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

# # Plot 2: Average time between payments by decile
# ax2 = fig.add_subplot(gs[0, 1])
# avg_time_between = [float(row['avg_time_between_payments']) for _, row in summary_df.iterrows()]
# ax2.bar(range(len(summary_df)), avg_time_between, color='steelblue', alpha=0.7)
# ax2.axhline(y=PAYMENT_TERMS_MONTHS, color='red', linestyle='--', linewidth=2, label=f'{PAYMENT_TERMS_MONTHS:.2f}-month terms')
# ax2.set_xlabel('Decile (by InvoiceAmount)', fontsize=11)
# ax2.set_ylabel('Avg Time Between Payments (months)', fontsize=11)
# ax2.set_title('Average Time Between Payments by Decile', fontsize=12, fontweight='bold')
# ax2.set_xticks(range(len(summary_df)))
# ax2.set_xticklabels(decile_labels)
# ax2.legend(fontsize=9)
# ax2.grid(True, alpha=0.3, axis='y')

# # Plot 3: InvoiceAmount range by decile
# ax3 = fig.add_subplot(gs[1, 0])
# min_amounts = [float(row['min_amount'].replace('$', '').replace(',', '')) for _, row in summary_df.iterrows()]
# max_amounts = [float(row['max_amount'].replace('$', '').replace(',', '')) for _, row in summary_df.iterrows()]
# ax3.bar(range(len(summary_df)), max_amounts, color='mediumseagreen', alpha=0.7, label='Max')
# ax3.bar(range(len(summary_df)), min_amounts, color='lightgreen', alpha=0.7, label='Min')
# ax3.set_xlabel('Decile', fontsize=11)
# ax3.set_ylabel('InvoiceAmount ($)', fontsize=11)
# ax3.set_title('InvoiceAmount Range by Decile', fontsize=12, fontweight='bold')
# ax3.set_xticks(range(len(summary_df)))
# ax3.set_xticklabels(decile_labels)
# ax3.legend(fontsize=9)
# ax3.grid(True, alpha=0.3, axis='y')

# # Plot 4: Average months overdue (when late) by decile
# ax4 = fig.add_subplot(gs[1, 1])
# avg_overdue = [float(row['avg_months_overdue']) for _, row in summary_df.iterrows()]
# ax4.bar(range(len(summary_df)), avg_overdue, color='indianred', alpha=0.7)
# ax4.set_xlabel('Decile (by InvoiceAmount)', fontsize=11)
# ax4.set_ylabel('Avg Months Overdue (when late)', fontsize=11)
# ax4.set_title('Average Months Overdue by Decile', fontsize=12, fontweight='bold')
# ax4.set_xticks(range(len(summary_df)))
# ax4.set_xticklabels(decile_labels)
# ax4.grid(True, alpha=0.3, axis='y')

# # Plot 5: Delinquency level distribution across deciles (stacked bar)
# ax5 = fig.add_subplot(gs[2, :])
# decile_nums = sorted(df['decile'].unique())
# cd_levels = sorted(df['cd'].unique())

# # Build matrix of P(cd | decile, late)
# cd_matrix = []
# for decile_num in decile_nums:
#     decile_profile = payment_profile['deciles'][f'decile_{decile_num}']
#     cd_given_late = decile_profile['delinquency_distribution']['cd_given_late']
    
#     row = [cd_given_late.get(cd_level, 0) * 100 for cd_level in cd_levels]
#     cd_matrix.append(row)

# cd_matrix = np.array(cd_matrix).T  # Transpose for stacking

# # Create stacked bar
# bottom = np.zeros(len(decile_nums))
# colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cd_levels)))

# for i, cd_level in enumerate(cd_levels):
#     ax5.bar(range(len(decile_nums)), cd_matrix[i], bottom=bottom, 
#             label=f'cd={cd_level}', color=colors[i], alpha=0.8)
#     bottom += cd_matrix[i]

# ax5.set_xlabel('Decile (by InvoiceAmount)', fontsize=11)
# ax5.set_ylabel('Distribution (%)', fontsize=11)
# ax5.set_title('Delinquency Level Distribution P(cd | Decile, Late)', fontsize=12, fontweight='bold')
# ax5.set_xticks(range(len(decile_nums)))
# ax5.set_xticklabels([f'D{d}' for d in decile_nums])
# ax5.legend(title='cd Level', fontsize=9, loc='upper right')
# ax5.grid(True, alpha=0.3, axis='y')

# plt.suptitle(f'Decile Payment Profile - {PAYMENT_TERMS_MONTHS:.2f} Month Payment Terms', 
#              fontsize=16, fontweight='bold', y=0.995)

# viz_file = OUTPUT_DIR / 'decile_payment_profile.png'
# plt.savefig(viz_file, dpi=300, bbox_inches='tight')
# print(f"✓ Saved visualization to: {viz_file}")
# plt.close()

# # ================================================================
# # Summary
# # ================================================================
# print("\n" + "="*70)
# print("DECILE PAYMENT PROFILE COMPLETE!")
# print("="*70)
# print(f"\nOutputs saved to: {OUTPUT_DIR}")
# print(f"\nKey files:")
# print(f"  1. decile_payment_profile.pkl - Profile for predictions")
# print(f"  2. decile_payment_profile_summary.csv - Human-readable summary")
# print(f"  3. decile_assignments.csv - All accounts with decile assignments")
# print(f"  4. decile_payment_profile.png - Visualizations")
# print(f"\nProfile contains {actual_n_deciles} deciles with equal sample sizes")
# print(f"Payment terms: {PAYMENT_TERMS_MONTHS:.2f} months (~{PAYMENT_TERMS_MONTHS * 30:.0f} days)")
# print(f"\nFor each decile, you have:")
# print(f"  - P(late): Probability of late payment (avg_time_between_payments > {PAYMENT_TERMS_MONTHS:.2f} months)")
# print(f"  - P(cd = k | late): Delinquency level distribution given late")
# print(f"  - Avg months overdue (when late)")
# print(f"\nNext step: Map this profile to invoice data by invoice amount")