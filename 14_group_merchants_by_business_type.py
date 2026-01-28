'''
Docstring for 14_map_merchants_to_match_layers

This script maps merchants from the Merchant Discount Detail file to specific 
match_layer labels using keyword matching on Account Name, Discount Offered, 
and Discount Offered 2 columns.

inputs:
- Merchant Discount Detail.xlsx

outputs:
- 14_merchants_mapped_to_match_layers.csv: All merchants with assigned match_layer
- 14_match_layer_summary.csv: Summary of match layers with merchant counts
- 14_merchants_unmapped.csv: Merchants not mapped to any match layer
'''

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Set up paths
data_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202')
output_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3/merchant')

# =========================================================
# LOAD MERCHANT DISCOUNT DATA
# =========================================================
print("\n" + "="*70)
print("LOADING MERCHANT DISCOUNT DATA")
print("="*70)

merchant_df = pd.read_excel(data_dir / 'Merchant Discount Detail.xlsx')

print(f"Loaded {len(merchant_df):,} merchant records")
print(f"Columns: {merchant_df.columns.tolist()}")

# =========================================================
# DEFINE MATCH LAYER KEYWORDS
# =========================================================

match_layer_keywords = {
    'L4_diesel_no_merchant': [
        'diesel', 'automotive diesel', 'agri diesel', 'off road diesel',
        'farm diesel', 'bulk diesel'
    ],
    
    'L4_petrol_no_merchant': [
        'petrol', 'premium petrol', 'unleaded', '91', '95', '98',
        'automotive petrol', 'pump petrol'
    ],
    
    'L5_Gas': [
        'gas', 'lpg', 'rockgas', 'elgas', 'gas bottle', 'lng', 'cng',
        'bottled gas', 'natural gas', 'propane', 'butane'
    ],
    
    'L5_Vet': [
        'vet', 'veterinary', 'animal health', 'vetlife', 'evolution vet',
        'vetserve', 'afterhours vet', 'town country vet', 'vet clinic',
        'vet service', 'animal care', 'livestock health'
    ],
    
    'L6_equipment_hire_fuel': [
        'equipment hire fuel', 'hire fuel', 'rental fuel', 'contractor fuel',
        'plant hire fuel', 'machinery fuel'
    ],
    
    'L7_cattle_feed': [
        'cattle feed', 'stock feed', 'animal feed', 'livestock feed',
        'grain', 'meal', 'supplement', 'hay', 'silage', 'fodder',
        'dairy feed', 'beef feed', 'calf milk', 'molasses'
    ],
    
    'L7_equipment_hire': [
        'hire', 'rental', 'equipment hire', 'u-hire', 'porter hire',
        'plant hire', 'machinery hire', 'tool hire', 'scaffold hire',
        'equipment rental', 'machinery rental'
    ],
    
    'L7_mechanic': [
        'mechanic', 'mechanical', 'auto repair', 'workshop', 'garage',
        'service centre', 'automotive repair', 'vehicle service',
        'maintenance', 'auto service'
    ],
    
    'L8_Landscaping': [
        'landscaping', 'landscape', 'nursery', 'garden', 'lawn', 'turf',
        'irrigation', 'tree', 'plants', 'gardening', 'groundcare',
        'horticulture', 'arborist'
    ],
    
    'L9_infrastructure_consumables': [
        'infrastructure', 'consumables', 'concrete', 'aggregate', 'gravel',
        'sand', 'metal', 'chip', 'culvert', 'pipe', 'drainage',
        'steel', 'reinforcing', 'mesh', 'fastener', 'bolt', 'screw',
        'wire', 'fencing', 'post', 'strainer'
    ]
}

# =========================================================
# HELPER FUNCTION: NORMALIZE TEXT
# =========================================================

def normalize_text(text):
    """
    Normalize text for keyword matching:
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================================================
# CLASSIFICATION FUNCTION
# =========================================================

def classify_merchant_to_layer(row):
    """
    Classify merchant into match_layer based on keyword matching across
    Account Name, Discount Offered, and Discount Offered 2.
    Returns tuple: (match_layer, confidence, matching_keyword, source_column)
    """
    # Combine all searchable text
    account_name = normalize_text(row.get('Account Name', ''))
    discount_1 = normalize_text(row.get('Discount Offered', ''))
    discount_2 = normalize_text(row.get('Discount Offered 2', ''))
    
    combined_text = f"{account_name} {discount_1} {discount_2}"
    
    # Track all potential matches
    matches = []
    
    for match_layer, keywords in match_layer_keywords.items():
        for keyword in keywords:
            keyword_normalized = normalize_text(keyword)
            
            # Check each field separately to track source
            if keyword_normalized in account_name:
                confidence = 80 + len(keyword)
                if account_name.startswith(keyword_normalized):
                    confidence += 10
                matches.append({
                    'match_layer': match_layer,
                    'confidence': min(confidence, 99),
                    'keyword': keyword,
                    'source': 'Account Name'
                })
            
            if keyword_normalized in discount_1:
                confidence = 85 + len(keyword)  # Higher weight for discount fields
                matches.append({
                    'match_layer': match_layer,
                    'confidence': min(confidence, 99),
                    'keyword': keyword,
                    'source': 'Discount Offered'
                })
            
            if keyword_normalized in discount_2:
                confidence = 85 + len(keyword)
                matches.append({
                    'match_layer': match_layer,
                    'confidence': min(confidence, 99),
                    'keyword': keyword,
                    'source': 'Discount Offered 2'
                })
    
    if not matches:
        return ('Unmapped', 0, '', '')
    
    # Return the match with highest confidence
    best_match = max(matches, key=lambda x: x['confidence'])
    return (
        best_match['match_layer'], 
        best_match['confidence'], 
        best_match['keyword'],
        best_match['source']
    )

# =========================================================
# APPLY CLASSIFICATION
# =========================================================

print("\n" + "="*70)
print("CLASSIFYING MERCHANTS TO MATCH LAYERS")
print("="*70)

merchant_df[['match_layer', 'confidence', 'matching_keyword', 'source_column']] = merchant_df.apply(
    lambda row: pd.Series(classify_merchant_to_layer(row)), axis=1
)

# =========================================================
# ANALYSIS OF RESULTS
# =========================================================

mapped = merchant_df[merchant_df['match_layer'] != 'Unmapped']
unmapped = merchant_df[merchant_df['match_layer'] == 'Unmapped']

print(f"\nTotal merchants: {len(merchant_df):,}")
print(f"Mapped to match layers: {len(mapped):,} ({len(mapped)/len(merchant_df)*100:.1f}%)")
print(f"Unmapped: {len(unmapped):,} ({len(unmapped)/len(merchant_df)*100:.1f}%)")

print("\n" + "="*70)
print("MATCH LAYER DISTRIBUTION")
print("="*70)
match_layer_counts = merchant_df['match_layer'].value_counts()
print(match_layer_counts.to_string())

print("\n" + "="*70)
print("SOURCE COLUMN DISTRIBUTION")
print("="*70)
source_counts = mapped['source_column'].value_counts()
print(source_counts.to_string())

print("\n" + "="*70)
print("CONFIDENCE SCORE STATISTICS")
print("="*70)
print(f"Mean confidence: {mapped['confidence'].mean():.1f}")
print(f"Median confidence: {mapped['confidence'].median():.1f}")
print(f"Min confidence: {mapped['confidence'].min():.1f}")
print(f"Max confidence: {mapped['confidence'].max():.1f}")

print("\n" + "="*70)
print("SAMPLE MAPPINGS (by match_layer)")
print("="*70)

for match_layer in match_layer_counts.head(15).index:
    if match_layer != 'Unmapped':
        sample = mapped[mapped['match_layer'] == match_layer].head(5)
        print(f"\n{match_layer}:")
        print(sample[['Account Name', 'confidence', 'matching_keyword', 'source_column']].to_string(index=False))

print("\n" + "="*70)
print("UNMAPPED MERCHANTS (First 30)")
print("="*70)
print(unmapped[['Account Name', 'Discount Offered', 'Discount Offered 2']].head(30).to_string(index=False))

# =========================================================
# SAVE OUTPUTS
# =========================================================

print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

# Save full merchant file with match layers
merchant_df.to_csv(output_dir / '14_merchants_mapped_to_match_layers.csv', index=False)
print(f"Saved: 14_merchants_mapped_to_match_layers.csv")

# Create match layer summary
match_layer_summary = merchant_df.groupby('match_layer').agg({
    'Account Name': 'count',
    'confidence': ['mean', 'min', 'max']
}).round(1)
match_layer_summary.columns = ['merchant_count', 'avg_confidence', 'min_confidence', 'max_confidence']
match_layer_summary = match_layer_summary.sort_values('merchant_count', ascending=False)
match_layer_summary.to_csv(output_dir / '14_match_layer_summary.csv')
print(f"Saved: 14_match_layer_summary.csv")

# Save unmapped merchants for manual review
unmapped[['ATS Number', 'Account Name', 'Discount Offered', 'Discount Offered 2']].to_csv(
    output_dir / '14_merchants_unmapped.csv', index=False
)
print(f"Saved: 14_merchants_unmapped.csv")

# =========================================================
# SAVE MERCHANT DISCOUNT DETAIL WITH MATCH_LAYER
# =========================================================

print("\n" + "="*70)
print("SAVING MERCHANT DISCOUNT DETAIL WITH MATCH_LAYER")
print("="*70)

# Create a copy of the merchant dataframe with match_layer column
merchant_with_layer_df = merchant_df.copy()

# Save as CSV
merchant_with_layer_output = output_dir / '14_merchant_discount_detail_with_match_layer.csv'
merchant_with_layer_df.to_csv(merchant_with_layer_output, index=False)

print(f"Saved: 14_merchant_discount_detail_with_match_layer.csv")
print(f"Total columns: {len(merchant_with_layer_df.columns)}")
print(f"New columns added: match_layer, confidence, matching_keyword, source_column")

print("\n" + "="*70)

print("\n" + "="*70)
print("PROCESS COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Review '14_merchants_mapped_to_match_layers.csv'")
print("2. Manually map merchants in '14_merchants_unmapped.csv'")
print("3. Use this mapping to update invoice line items in matching progress file")