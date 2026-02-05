"""
Script: 19_merchant_matching_labels_stats.py

Purpose:
    Provide comprehensive statistics on match_layer labels from merchant matching scripts (13 and 14),
    including:
    1. All unique match_layer labels and their counts
    2. Associated discount rates for each label
    3. Number of matches per label
    4. Summary statistics and breakdown

Inputs:
    - Latest matching progress CSV (13.7_matching_progress.csv or most recent)
    - Discount rates by match layer (14.1_average_discount_by_match_layer.csv)

Outputs:
    - Console output with comprehensive statistics
    - 19_match_layer_stats_summary.csv (detailed statistics)
    - 19_match_layer_distribution.csv (label distribution)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
merchant_folder_dir = base_dir / "merchant"
output_dir = merchant_folder_dir

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MERCHANT MATCHING LABELS STATISTICS ANALYSIS")
print("="*80)

# ================================================================
# FIND LATEST MATCHING PROGRESS FILE
# ================================================================
print(f"\n{'='*80}")
print(f"FINDING LATEST MATCHING PROGRESS FILE")
print(f"{'='*80}")

# List all matching progress files and get the most recent
matching_files = list(merchant_folder_dir.glob('13.*_matching_progress.csv'))
matching_files.sort()

if matching_files:
    latest_matching_file = matching_files[-1]
    print(f"✓ Found latest matching file: {latest_matching_file.name}")
else:
    print("❌ No matching progress files found. Checking for alternative files...")
    # Try alternative file names
    alt_files = ['13.7_matching_progress.csv', '13.6_matching_progress.csv', '13_matching_progress.csv']
    for alt_file in alt_files:
        alt_path = merchant_folder_dir / alt_file
        if alt_path.exists():
            latest_matching_file = alt_path
            print(f"✓ Found alternative file: {alt_file}")
            break
    else:
        print("❌ No matching progress files found. Cannot proceed.")
        exit(1)

# ================================================================
# LOAD MATCHING PROGRESS DATA
# ================================================================
print(f"\n{'='*80}")
print(f"LOADING MATCHING PROGRESS DATA")
print(f"{'='*80}")

try:
    matching_df = pd.read_csv(latest_matching_file)
    print(f"✓ Loaded matching data: {len(matching_df):,} rows")
    print(f"Columns: {matching_df.columns.tolist()}")
except Exception as e:
    print(f"❌ Error loading matching data: {e}")
    exit(1)

# ================================================================
# LOAD DISCOUNT RATES DATA
# ================================================================
print(f"\n{'='*80}")
print(f"LOADING DISCOUNT RATES DATA")
print(f"{'='*80}")

discount_file = merchant_folder_dir / '14.1_average_discount_by_match_layer.csv'
if discount_file.exists():
    try:
        discount_df = pd.read_csv(discount_file)
        print(f"✓ Loaded discount rates: {len(discount_df):,} rows")
        print(f"Columns: {discount_df.columns.tolist()}")
    except Exception as e:
        print(f"⚠️ Warning: Error loading discount data: {e}")
        discount_df = None
else:
    print(f"⚠️ Warning: Discount rates file not found: {discount_file.name}")
    discount_df = None

# ================================================================
# ANALYZE MATCH_LAYER DISTRIBUTION
# ================================================================
print(f"\n{'='*80}")
print(f"ANALYZING MATCH_LAYER DISTRIBUTION")
print(f"{'='*80}")

# Count match_layer values
if 'match_layer' in matching_df.columns:
    match_layer_counts = matching_df['match_layer'].value_counts().sort_index()
    
    # Create comprehensive stats DataFrame
    match_stats = pd.DataFrame({
        'match_layer': match_layer_counts.index,
        'count': match_layer_counts.values,
        'percentage': (match_layer_counts.values / len(matching_df) * 100).round(2)
    })
    
    # Calculate some summary statistics
    total_rows = len(matching_df)
    total_matched_layers = match_layer_counts.sum()
    unmatched_count = matching_df[matching_df['match_layer'] == 'unmatched'].shape[0] if 'unmatched' in match_layer_counts.index else 0
    
    print(f"\nTotal rows in dataset: {total_rows:,}")
    print(f"Rows with match_layer: {total_matched_layers:,}")
    print(f"Unmatched rows: {unmatched_count:,}")
    print(f"Unique match_layer labels: {len(match_layer_counts)}")
    
    print(f"\n{'Match Layer Distribution:'}")
    print("="*50)
    print(match_stats.to_string(index=False))
    
else:
    print("❌ Error: 'match_layer' column not found in matching data")
    match_stats = None

# ================================================================
# MERGE WITH DISCOUNT RATES
# ================================================================
print(f"\n{'='*80}")
print(f"MERGING MATCH LAYER STATS WITH DISCOUNT RATES")
print(f"{'='*80}")

if match_stats is not None and discount_df is not None:
    try:
        # Merge match layer stats with discount rates
        combined_stats = match_stats.merge(
            discount_df[['match_layer', 'avg_discount', 'count', 'std_discount', 'min_discount', 'max_discount']],
            on='match_layer',
            how='left',
            suffixes=('_total', '_with_discount')
        )
        
        # Rename columns for clarity
        combined_stats.rename(columns={
            'count_total': 'total_matches',
            'count_with_discount': 'matches_with_discount_data'
        }, inplace=True)
        
        # Round discount values
        discount_cols = ['avg_discount', 'std_discount', 'min_discount', 'max_discount']
        for col in discount_cols:
            if col in combined_stats.columns:
                combined_stats[col] = combined_stats[col].round(4)
        
        print("✓ Successfully merged match layer stats with discount rates")
        print(f"\nCombined Statistics:")
        print("="*60)
        print(combined_stats.to_string(index=False))
        
    except Exception as e:
        print(f"⚠️ Warning: Error merging discount data: {e}")
        combined_stats = match_stats
        
else:
    print("⚠️ Using match layer stats only (discount data not available)")
    combined_stats = match_stats

# ================================================================
# ANALYZE MATCHING TYPES
# ================================================================
print(f"\n{'='*80}")
print(f"ANALYZING MATCHING TYPES AND PATTERNS")
print(f"{'='*80}")

if match_stats is not None:
    # Categorize match layers
    layer_categories = {
        'exact_matches': [],
        'fuzzy_matches': [],
        'manual_matches': [],
        'unmatched': [],
        'other': []
    }
    
    for layer in match_stats['match_layer']:
        layer_str = str(layer).lower()
        if layer_str == 'unmatched':
            layer_categories['unmatched'].append(layer)
        elif any(x in layer_str for x in ['l1', 'l2', 'l3']):
            layer_categories['exact_matches'].append(layer)
        elif any(x in layer_str for x in ['l4', 'l5', 'l6', 'l7', 'l8', 'l9']):
            layer_categories['fuzzy_matches'].append(layer)
        elif 'manual' in layer_str or 'map' in layer_str:
            layer_categories['manual_matches'].append(layer)
        else:
            layer_categories['other'].append(layer)
    
    # Calculate category stats
    category_stats = []
    for category, layers in layer_categories.items():
        if layers:
            count = match_stats[match_stats['match_layer'].isin(layers)]['count'].sum()
            percentage = match_stats[match_stats['match_layer'].isin(layers)]['percentage'].sum()
            category_stats.append({
                'category': category,
                'layers': len(layers),
                'total_matches': count,
                'percentage': round(percentage, 2)
            })
    
    category_stats_df = pd.DataFrame(category_stats)
    
    print(f"\nMatching Categories Summary:")
    print("="*40)
    print(category_stats_df.to_string(index=False))

# ================================================================
# SAVE RESULTS
# ================================================================
print(f"\n{'='*80}")
print(f"SAVING RESULTS")
print(f"{'='*80}")

# Save detailed statistics
if combined_stats is not None:
    detailed_output_file = output_dir / '19_match_layer_stats_summary.csv'
    combined_stats.to_csv(detailed_output_file, index=False)
    print(f"✓ Saved detailed statistics: {detailed_output_file.name}")

# Save distribution data
if match_stats is not None:
    distribution_output_file = output_dir / '19_match_layer_distribution.csv'
    match_stats.to_csv(distribution_output_file, index=False)
    print(f"✓ Saved distribution data: {distribution_output_file.name}")

# Save category stats if available
if 'category_stats_df' in locals():
    category_output_file = output_dir / '19_match_layer_categories.csv'
    category_stats_df.to_csv(category_output_file, index=False)
    print(f"✓ Saved category statistics: {category_output_file.name}")

# ================================================================
# FINAL SUMMARY
# ================================================================
print(f"\n{'='*80}")
print(f"FINAL SUMMARY")
print(f"{'='*80}")

if match_stats is not None:
    print(f"\n📊 MATCH LAYER STATISTICS:")
    print(f"  • Total dataset rows: {total_rows:,}")
    print(f"  • Unique match layers: {len(match_layer_counts)}")
    print(f"  • Most common layer: {match_layer_counts.index[0]} ({match_layer_counts.iloc[0]:,} matches)")
    print(f"  • Unmatched rows: {unmatched_count:,} ({unmatched_count/total_rows*100:.1f}%)")
    
    if discount_df is not None:
        print(f"\n💰 DISCOUNT RATE INFORMATION:")
        avg_discount_overall = discount_df['avg_discount'].mean()
        print(f"  • Layers with discount data: {len(discount_df)}")
        print(f"  • Average discount rate: {avg_discount_overall:.4f}")
        print(f"  • Discount rate range: {discount_df['avg_discount'].min():.4f} - {discount_df['avg_discount'].max():.4f}")
    
    print(f"\n📁 OUTPUT FILES CREATED:")
    print(f"  • {detailed_output_file.name if 'detailed_output_file' in locals() else 'N/A'}")
    print(f"  • {distribution_output_file.name if 'distribution_output_file' in locals() else 'N/A'}")
    print(f"  • {category_output_file.name if 'category_stats_df' in locals() else 'N/A'}")

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE!")
print(f"{'='*80}")
