'''
Docstring for 09.3.2_decile_visualization

This script creates visualizations for the decile payment profile analysis,
showing late payment patterns, delinquency levels, and payment behavior across
invoice amount deciles.

inputs:
- decile_payment_profile.pkl
- decile_payment_profile_summary.csv

outputs:
- 9.3.2_dual_axis_late_payments.png
- 9.3.2_multi_panel_dashboard.png
- 9.3.2_delinquency_heatmap.png
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

# ================================================================
# Configuration
# ================================================================
# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
profile_dir = base_dir / "payment_profile"
profile_dir.mkdir(exist_ok=True)

# ================================================================
# Load data
# ================================================================
print("="*70)
print("LOADING DECILE PAYMENT PROFILE DATA")
print("="*70)

# Load pickled payment profile
profile_file = profile_dir / "decile_payment_profile.pkl"
with open(profile_file, 'rb') as f:
    payment_profile = pickle.load(f)

# Load summary CSV
summary_file = profile_dir / "decile_payment_profile_summary.csv"
summary_df = pd.read_csv(summary_file)

print(f"✓ Loaded payment profile with {len(payment_profile['deciles'])} deciles")
print(f"✓ Loaded summary table with {len(summary_df)} rows")

# ================================================================
# Parse summary data for plotting
# ================================================================
print("\n" + "="*70)
print("PREPARING DATA FOR VISUALIZATION")
print("="*70)

# Extract numeric values from formatted strings
summary_df['prob_late_numeric'] = summary_df['prob_late_pct'].str.rstrip('%').astype(float)
summary_df['avg_months_overdue_numeric'] = summary_df['avg_months_overdue'].astype(float)

# Extract most common CD information
summary_df['most_common_cd_code'] = summary_df['most_common_cd_when_late'].str.extract(r'cd=(\d+)')[0]
summary_df['most_common_cd_pct'] = summary_df['most_common_cd_when_late'].str.extract(r'\(([0-9.]+)%\)')[0].astype(float)

print("✓ Data prepared for visualization")
print(f"\nDecile range: {summary_df['decile'].min()} to {summary_df['decile'].max()}")
print(f"Late payment probability range: {summary_df['prob_late_numeric'].min():.1f}% to {summary_df['prob_late_numeric'].max():.1f}%")

# ================================================================
# VISUALIZATION 1: Dual-Axis Bar Chart (RECOMMENDED)
# ================================================================
print("\n" + "="*70)
print("CREATING DUAL-AXIS BAR CHART")
print("="*70)

fig, ax1 = plt.subplots(figsize=(16, 9))

# Create color map for avg_months_overdue
norm = plt.Normalize(vmin=summary_df['avg_months_overdue_numeric'].min(), 
                     vmax=summary_df['avg_months_overdue_numeric'].max())
cmap = plt.cm.RdYlGn_r  # Red (bad) to Green (good), reversed
colors = [cmap(norm(val)) for val in summary_df['avg_months_overdue_numeric']]

# Primary axis: Number of late payments (bars)
x_pos = np.arange(len(summary_df))
bars = ax1.bar(x_pos, summary_df['n_late'], 
               color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
               label='Number of Late Payments')

ax1.set_xlabel('Decile (by Invoice Amount)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Number of Late Payments', fontsize=13, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'D{d}' for d in summary_df['decile']], fontsize=11)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Secondary axis: Probability of late payment (line)
ax2 = ax1.twinx()
line = ax2.plot(x_pos, summary_df['prob_late_numeric'], 
                color='#2E86AB', linewidth=3, marker='o', 
                markersize=10, markerfacecolor='#2E86AB', 
                markeredgecolor='white', markeredgewidth=2,
                label='Probability of Late Payment')

ax2.set_ylabel('Probability of Late Payment (%)', fontsize=13, fontweight='bold', color='#2E86AB')
ax2.tick_params(axis='y', labelcolor='#2E86AB')

# Add annotations with most common CD code
for i, row in summary_df.iterrows():
    # Add CD code annotation above bars
    if pd.notna(row['most_common_cd_code']):
        ax1.text(i, row['n_late'], 
                f"cd={int(float(row['most_common_cd_code']))}",
                ha='center', va='bottom', fontsize=9, 
                fontweight='bold', color='darkred')
    
    # Add percentage on the line
    ax2.text(i, row['prob_late_numeric'], 
            f"{row['prob_late_numeric']:.1f}%",
            ha='center', va='bottom', fontsize=9, 
            fontweight='bold', color='#2E86AB')

# Title and legend
plt.title('Late Payment Analysis by Invoice Amount Decile\n(Bar Color = Avg Months Overdue)', 
          fontsize=16, fontweight='bold', pad=20)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

# Add colorbar for avg_months_overdue
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, pad=0.02)
cbar.set_label('Avg Months Overdue (When Late)', fontsize=11, fontweight='bold')

plt.tight_layout()
output_file_1 = profile_dir / '9.3.2_dual_axis_late_payments.png'
plt.savefig(output_file_1, dpi=300, bbox_inches='tight')
print(f"✓ Saved dual-axis chart to: {output_file_1.name}")
plt.close()

# ================================================================
# VISUALIZATION 2: Multi-Panel Dashboard
# ================================================================
print("\n" + "="*70)
print("CREATING MULTI-PANEL DASHBOARD")
print("="*70)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 1, hspace=0.3)

x_pos = np.arange(len(summary_df))
x_labels = [f'D{d}' for d in summary_df['decile']]

# Panel 1: Number of late payments
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(x_pos, summary_df['n_late'], 
                color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=1)
ax1.set_ylabel('Number of Late Payments', fontsize=12, fontweight='bold')
ax1.set_title('Late Payment Count by Decile', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels
for i, v in enumerate(summary_df['n_late']):
    ax1.text(i, v, f'{v}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 2: Probability of late payment
ax2 = fig.add_subplot(gs[1, 0])
line2 = ax2.plot(x_pos, summary_df['prob_late_numeric'], 
                 color='#3498DB', linewidth=3, marker='o', 
                 markersize=10, markerfacecolor='#3498DB', 
                 markeredgecolor='white', markeredgewidth=2)
ax2.fill_between(x_pos, summary_df['prob_late_numeric'], 
                  alpha=0.3, color='#3498DB')
ax2.set_ylabel('Probability of Late Payment (%)', fontsize=12, fontweight='bold')
ax2.set_title('Late Payment Probability by Decile', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels)
ax2.grid(True, alpha=0.3, linestyle='--')

# Add value labels
for i, v in enumerate(summary_df['prob_late_numeric']):
    ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 3: Most common CD level when late
ax3 = fig.add_subplot(gs[2, 0])

# Extract CD codes as numeric values
cd_codes = []
for _, row in summary_df.iterrows():
    cd_code = row['most_common_cd_code']
    if pd.notna(cd_code):
        cd_codes.append(int(float(cd_code)))
    else:
        cd_codes.append(0)

# Create color gradient based on CD level (higher = worse)
norm = plt.Normalize(vmin=0, vmax=10)
cmap = plt.cm.RdYlGn_r  # Red (bad) to Green (good), reversed
colors3 = [cmap(norm(cd)) for cd in cd_codes]

bars3 = ax3.barh(x_pos, cd_codes, 
                 color=colors3, alpha=0.7, edgecolor='black', linewidth=1)

ax3.set_xlabel('Most Common CD Level (When Late)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Decile', fontsize=12, fontweight='bold')
ax3.set_title('Most Common Delinquency Code for Late Payments by Decile', 
              fontsize=13, fontweight='bold')
ax3.set_yticks(x_pos)
ax3.set_yticklabels(x_labels)
ax3.set_xlim(0, 10)
ax3.set_xticks(range(0, 11, 1))
ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
ax3.invert_yaxis()  # Highest decile at top

# Add value labels with percentage
for i, row in summary_df.iterrows():
    cd_code = cd_codes[i]
    cd_pct = row['most_common_cd_pct']
    
    if cd_code > 0:
        label_text = f"cd={cd_code} ({cd_pct:.0f}%)"
        ax3.text(cd_code + 0.2, i, label_text, 
                ha='left', va='center', fontsize=9, fontweight='bold')
    else:
        ax3.text(0.1, i, "N/A", 
                ha='left', va='center', fontsize=9, fontweight='bold')

plt.suptitle('Payment Profile Dashboard - Decile Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
output_file_2 = profile_dir / '9.3.2_multi_panel_dashboard.png'
plt.savefig(output_file_2, dpi=300, bbox_inches='tight')
print(f"✓ Saved multi-panel dashboard to: {output_file_2.name}")
plt.close()

# ================================================================
# VISUALIZATION 3: Heatmap
# ================================================================
print("\n" + "="*70)
print("CREATING DELINQUENCY HEATMAP")
print("="*70)

# Build data matrix for heatmap
heatmap_data = []
for _, row in summary_df.iterrows():
    heatmap_data.append([
        row['n_late'],
        row['prob_late_numeric'],
        row['avg_months_overdue_numeric'],
    ])

heatmap_df = pd.DataFrame(
    heatmap_data,
    columns=['Late Count', 'Late Prob (%)', 'Avg Months Overdue'],
    index=[f'D{d}' for d in summary_df['decile']]
)

# Normalize each column for better color visualization
heatmap_normalized = heatmap_df.copy()
for col in heatmap_normalized.columns:
    col_min = heatmap_normalized[col].min()
    col_max = heatmap_normalized[col].max()
    if col_max > col_min:
        heatmap_normalized[col] = (heatmap_normalized[col] - col_min) / (col_max - col_min)
    else:
        heatmap_normalized[col] = 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Heatmap 1: Normalized values
sns.heatmap(heatmap_normalized, annot=False, cmap='RdYlGn_r', 
            cbar_kws={'label': 'Normalized Intensity (0=Best, 1=Worst)'},
            linewidths=1, linecolor='white', ax=ax1)
ax1.set_title('Payment Behavior Intensity by Decile\n(Normalized)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax1.set_ylabel('Decile', fontsize=12, fontweight='bold')

# Add actual values as text annotations
for i in range(len(summary_df)):
    for j, col in enumerate(heatmap_df.columns):
        value = heatmap_df.iloc[i, j]
        ax1.text(j + 0.5, i + 0.5, f'{value:.1f}', 
                ha='center', va='center', fontsize=10, 
                fontweight='bold', color='black')

# Heatmap 2: Add CD information
cd_data = []
for _, row in summary_df.iterrows():
    cd_code = row['most_common_cd_code']
    cd_pct = row['most_common_cd_pct']
    
    if pd.notna(cd_code):
        cd_data.append(f"cd={int(float(cd_code))}\n({cd_pct:.0f}%)")
    else:
        cd_data.append("N/A")

cd_df = pd.DataFrame(cd_data, columns=['Most Common CD'], 
                     index=[f'D{d}' for d in summary_df['decile']])

# Create color coding for CD codes
cd_numeric = []
for _, row in summary_df.iterrows():
    cd_code = row['most_common_cd_code']
    if pd.notna(cd_code):
        cd_numeric.append(float(cd_code))
    else:
        cd_numeric.append(0)

cd_matrix = np.array(cd_numeric).reshape(-1, 1)

im = ax2.imshow(cd_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=10)
ax2.set_yticks(np.arange(len(summary_df)))
ax2.set_yticklabels([f'D{d}' for d in summary_df['decile']])
ax2.set_xticks([0])
ax2.set_xticklabels(['Delinquency Code'])
ax2.set_title('Most Common Delinquency Code\nWhen Late', 
              fontsize=14, fontweight='bold', pad=15)

# Add CD labels
for i, label in enumerate(cd_data):
    ax2.text(0, i, label, ha='center', va='center', 
            fontsize=11, fontweight='bold', color='black')

cbar2 = plt.colorbar(im, ax=ax2)
cbar2.set_label('CD Code Level', fontsize=11, fontweight='bold')

plt.suptitle('Delinquency Analysis Heatmap - By Invoice Amount Decile', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
output_file_3 = profile_dir / '9.3.2_delinquency_heatmap.png'
plt.savefig(output_file_3, dpi=300, bbox_inches='tight')
print(f"✓ Saved delinquency heatmap to: {output_file_3.name}")
plt.close()

# ================================================================
# Summary
# ================================================================
print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"\nFiles created in: {profile_dir}")
print(f"\n  1. 9.3.2_dual_axis_late_payments.png")
print(f"     - Dual-axis chart showing late payment count and probability")
print(f"     - Bar colors indicate avg months overdue")
print(f"     - Annotations show most common CD code")
print(f"\n  2. 9.3.2_multi_panel_dashboard.png")
print(f"     - Three-panel dashboard with:")
print(f"       • Late payment counts")
print(f"       • Late payment probabilities")
print(f"       • Most common delinquency code (CD level)")
print(f"\n  3. 9.3.2_delinquency_heatmap.png")
print(f"     - Heatmap showing payment behavior intensity")
print(f"     - Most common delinquency codes by decile")
print("="*70)