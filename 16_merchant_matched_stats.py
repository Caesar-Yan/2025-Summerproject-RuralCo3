"""
Script: 16_merchant_matched_stats.py

Purpose:
    Calculate statistics on merchant matching progress, specifically:
    1. Count rows with match_layer = 'unmatched' from 13.7_matching_progress.csv
    2. Count total rows in ats_invoice_line_item.csv and invoice_line_item.csv
    3. Calculate percentage of unmatched rows relative to total original rows

Inputs:
    - 13.7_matching_progress.csv (matching results)
    - ats_invoice_line_item.csv (original ATS line items)
    - invoice_line_item.csv (original invoice line items)

Outputs:
    - Console output with matching statistics
    - 16_merchant_matching_stats.csv (summary statistics)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories (using same format as other scripts)
base_dir = Path("T:/projects/2025/RuralCo")
invoices_dir = base_dir / "Data provided by RuralCo 20251202/invoices_export/20251121"
ruralco3_dir = base_dir / "Data provided by RuralCo 20251202/RuralCo3"
data_cleaning_dir = ruralco3_dir / "data_cleaning"
merchant_dir = ruralco3_dir / "merchant"

# Create output directory if it doesn't exist
merchant_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("MERCHANT MATCHING STATISTICS ANALYSIS")
print("="*70)

# ================================================================
# LOAD MATCHING PROGRESS DATA
# ================================================================
print(f"\n{'='*70}")
print(f"LOADING MATCHING PROGRESS DATA")
print(f"{'='*70}")

matching_progress_file = merchant_dir / '13.7_matching_progress.csv'
if matching_progress_file.exists():
    matching_df = pd.read_csv(matching_progress_file)
    print(f"✓ Loaded matching progress data: {len(matching_df):,} rows")
else:
    print(f"❌ Error: {matching_progress_file} not found")
    exit(1)

# Count unmatched rows
unmatched_count = len(matching_df[matching_df['match_layer'] == 'unmatched'])
total_matching_rows = len(matching_df)

print(f"📊 MATCHING PROGRESS SUMMARY:")
print(f"  • Total rows in matching file: {total_matching_rows:,}")
print(f"  • Rows with 'unmatched' label: {unmatched_count:,}")
print(f"  • Percentage unmatched: {(unmatched_count/total_matching_rows)*100:.2f}%")

# ================================================================
# LOAD ORIGINAL INVOICE LINE ITEM DATA
# ================================================================
print(f"\n{'='*70}")
print(f"LOADING ORIGINAL INVOICE LINE ITEM DATA")
print(f"{'='*70}")

# Load ATS invoice line items
ats_file = invoices_dir / 'ats_invoice_line_item.csv'
if ats_file.exists():
    ats_df = pd.read_csv(ats_file, low_memory=False)
    ats_total_rows = len(ats_df)
    print(f"✓ Loaded ATS invoice line items: {ats_total_rows:,} rows")
else:
    print(f"❌ Error: {ats_file} not found")
    ats_total_rows = 0

# Load regular invoice line items
invoice_file = invoices_dir / 'invoice_line_item.csv'
if invoice_file.exists():
    invoice_df = pd.read_csv(invoice_file, low_memory=False)
    invoice_total_rows = len(invoice_df)
    print(f"✓ Loaded Invoice line items: {invoice_total_rows:,} rows")
else:
    print(f"❌ Error: {invoice_file} not found")
    invoice_total_rows = 0

# Calculate totals
total_original_rows = ats_total_rows + invoice_total_rows
unmatched_percentage = (unmatched_count / total_original_rows) * 100 if total_original_rows > 0 else 0

print(f"\n📊 ORIGINAL DATA SUMMARY:")
print(f"  • ATS invoice line items: {ats_total_rows:,}")
print(f"  • Regular invoice line items: {invoice_total_rows:,}")
print(f"  • Total original rows: {total_original_rows:,}")

# ================================================================
# CALCULATE FINAL STATISTICS
# ================================================================
print(f"\n{'='*70}")
print(f"FINAL MATCHING STATISTICS")
print(f"{'='*70}")

print(f"🎯 KEY METRICS:")
print(f"  • Total original invoice line items: {total_original_rows:,}")
print(f"  • Rows still unmatched: {unmatched_count:,}")
print(f"  • Unmatched as % of original data: {unmatched_percentage:.2f}%")
print(f"  • Successfully matched: {total_matching_rows - unmatched_count:,}")
print(f"  • Match success rate: {((total_matching_rows - unmatched_count)/total_original_rows)*100:.2f}%")

# Check if matching file covers all original data
coverage_percentage = (total_matching_rows / total_original_rows) * 100 if total_original_rows > 0 else 0
print(f"\n🔍 DATA COVERAGE CHECK:")
print(f"  • Matching file coverage: {coverage_percentage:.2f}%")
if coverage_percentage < 99:
    print(f"  ⚠️  WARNING: Matching file doesn't cover all original data")
else:
    print(f"  ✓ Good coverage of original data")

# ================================================================
# LOAD AVERAGE DISCOUNT BY MATCH LAYER FROM 14.1
# ================================================================
print(f"\n{'='*70}")
print(f"LOADING AVERAGE DISCOUNT BY MATCH LAYER")
print(f"{'='*70}")

discount_by_layer_file = merchant_dir / '14.1_average_discount_by_match_layer.csv'
if discount_by_layer_file.exists():
    discount_df = pd.read_csv(discount_by_layer_file)
    print(f"✓ Loaded discount by match_layer data: {len(discount_df):,} layers")
    
    print(f"\n📊 DISCOUNT BY MATCH LAYER:")
    print(discount_df[['match_layer', 'avg_discount', 'count']].to_string(index=False))
    
    # Calculate overall statistics
    overall_avg_discount = discount_df['avg_discount'].mean()
    weighted_avg_discount = (discount_df['avg_discount'] * discount_df['count']).sum() / discount_df['count'].sum()
    
    print(f"\n🎯 DISCOUNT SUMMARY:")
    print(f"  • Number of match layers: {len(discount_df)}")
    print(f"  • Simple average discount: {overall_avg_discount:.4f}%")
    print(f"  • Weighted average discount: {weighted_avg_discount:.4f}%")
    print(f"  • Highest discount layer: {discount_df.loc[discount_df['avg_discount'].idxmax(), 'match_layer']} ({discount_df['avg_discount'].max():.4f}%)")
    print(f"  • Lowest discount layer: {discount_df.loc[discount_df['avg_discount'].idxmin(), 'match_layer']} ({discount_df['avg_discount'].min():.4f}%)")
    
    # Merge with matching progress to see which layers have unmatched items
    if 'match_layer' in matching_df.columns:
        layer_match_stats = matching_df.groupby('match_layer').size().reset_index(name='total_items')
        unmatched_by_layer = matching_df[matching_df['match_layer'] == 'unmatched'].groupby('match_layer').size().reset_index(name='unmatched_items')
        
        # Merge discount data with match statistics
        layer_analysis = discount_df.merge(layer_match_stats, on='match_layer', how='left')
        
        print(f"\n📈 LAYER COVERAGE ANALYSIS:")
        for _, row in layer_analysis.iterrows():
            total_items = row['total_items'] if pd.notna(row['total_items']) else 0
            print(f"  • {row['match_layer']}: {row['avg_discount']:.4f}% discount, {total_items:,} items")
        
else:
    print(f"❌ Warning: {discount_by_layer_file} not found")
    discount_df = None
    overall_avg_discount = None
    weighted_avg_discount = None

# ================================================================
# CREATE SUMMARY DATAFRAME
# ================================================================
summary_stats = {
    'metric': [
        'Total ATS line items',
        'Total Invoice line items', 
        'Total original line items',
        'Total rows in matching file',
        'Rows with unmatched label',
        'Successfully matched rows',
        'Unmatched percentage of original',
        'Match success rate',
        'Data coverage percentage',
        'Number of match layers with discounts',
        'Simple average discount across layers',
        'Weighted average discount across layers'
    ],
    'value': [
        ats_total_rows,
        invoice_total_rows,
        total_original_rows,
        total_matching_rows,
        unmatched_count,
        total_matching_rows - unmatched_count,
        round(unmatched_percentage, 2),
        round(((total_matching_rows - unmatched_count)/total_original_rows)*100, 2),
        round(coverage_percentage, 2),
        len(discount_df) if discount_df is not None else 0,
        round(overall_avg_discount, 4) if overall_avg_discount is not None else 'N/A',
        round(weighted_avg_discount, 4) if weighted_avg_discount is not None else 'N/A'
    ],
    'unit': [
        'rows',
        'rows',
        'rows', 
        'rows',
        'rows',
        'rows',
        '%',
        '%',
        '%',
        'layers',
        '%',
        '%'
    ]
}

summary_df = pd.DataFrame(summary_stats)

# ================================================================
# SAVE RESULTS
# ================================================================
output_file = merchant_dir / '16_merchant_matching_stats.csv'
summary_df.to_csv(output_file, index=False)

print(f"\n{'='*70}")
print(f"RESULTS SAVED")
print(f"{'='*70}")
print(f"✓ Summary statistics saved to: {output_file}")
print(f"✓ Contains {len(summary_df)} metrics")

print(f"\n📋 SUMMARY TABLE:")
print(summary_df.to_string(index=False))

print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE!")
print(f"{'='*70}")

# ================================================================
# RETURN KEY VALUES (for potential use in other scripts)
# ================================================================
print(f"\n🔢 KEY RETURN VALUES:")
print(f"  • unmatched_count = {unmatched_count}")
print(f"  • total_original_rows = {total_original_rows}")  
print(f"  • unmatched_percentage = {unmatched_percentage:.2f}%")
if overall_avg_discount is not None:
    print(f"  • simple_avg_discount = {overall_avg_discount:.4f}%")
    print(f"  • weighted_avg_discount = {weighted_avg_discount:.4f}%")
    print(f"  • num_discount_layers = {len(discount_df)}")
