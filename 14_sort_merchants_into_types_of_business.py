'''
Docstring for 14_group_merchants_by_business_type

This script uses fuzzy matching and keyword clustering to automatically group 
merchants from the Merchant Discount Detail file into business types.

inputs:
- Merchant Discount Detail.xlsx

outputs:
- merchants_grouped_by_business_type.csv: All merchants with assigned business_type
- business_type_summary.csv: Summary of business types with merchant counts
'''

import pandas as pd
import numpy as np
import re
from pathlib import Path
from fuzzywuzzy import fuzz, process
from collections import defaultdict

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
# DEFINE BUSINESS TYPE CATEGORIES WITH KEYWORDS
# =========================================================

business_type_keywords = {
    'Fuel_Stations': [
        'fuelstop', 'fuel', 'mobil', 'npd', 'bp', 'z energy', 'caltex', 'shell', 
        'service station', 'petrol', 'petroleum', 'gas station', 'diesel'
    ],
    
    'Veterinary': [
        'vet', 'veterinary', 'animal health', 'vetlife', 'evolution vet', 
        'vetserve', 'afterhours vet', 'town country vet'
    ],
    
    'Automotive_Mechanical': [
        'auto', 'automotive', 'mechanical', 'mechanic', 'motors', 'motor group',
        'panel', 'tyres', 'tyre', 'repco', 'brake', 'clutch', 'transmission', 
        'workshop', 'auto electric', 'radiator', 'muffler', 'exhaust'
    ],
    
    'Building_Hardware': [
        'mitre 10', 'placemakers', 'itm', 'hammer hardware', 'building', 
        'timber', 'lumber', 'hardware', 'toolshed'
    ],
    
    'Concrete_Construction': [
        'concrete', 'Allied Concrete', 'pre-stress', 'paving'
    ],
    
    'Plumbing_Supplies': [
        'plumbing', 'pipe', 'oakleys plumbing', 'mico plumbing'
    ],
    
    'Paint_Supplies': [
        'paint', 'resene', 'colour shop'
    ],
    
    'Equipment_Hire': [
        'hire', 'rental', 'equipment hire', 'u-hire', 'porter hire'
    ],
    
    'Accommodation_Travel': [
        'house of travel', 'travel', 'hotel', 'motel', 'lodge', 
        'accommodation', 'motor inn', 'motor lodge'
    ],
    
    'Farm_Supplies': [
        'farm supplies', 'farmlands', 'rural', 'agri', 'seed', 'grain', 
        'stock yard', 'farmside', 'agriline', 'saddlery'
    ],
    
    'Landscaping_Nursery': [
        'nursery', 'garden', 'landscap', 'irrigation', 'lawn', 'tree'
    ],
    
    'Gas_LPG': [
        'rockgas', 'lpg', 'gas bottle', 'elgas'
    ],
    
    'Retail_Office': [
        'whitcoulls', 'paper plus', 'office', 'stationery'
    ],
    
    'Pharmacy_Medical': [
        'pharmacy', 'dental', 'medical', 'health', 'eyecare', 'optom'
    ],
    
    'Telecommunications': [
        'ubb', 'broadband', 'wireless', 'spark', 'vodafone', 'telecom'
    ],
    
    'Supermarket_Grocery': [
        'new world', 'four square', 'supermarket', 'freshchoice', 
        'paknsave', 'countdown', 'supervalue', 'raeward fresh'
    ],
    
    'Restaurant_Hospitality': [
        'restaurant', 'cafe', 'bar', 'tavern', 'pub', 'hotel', 'bakery'
    ],
    
    'Furniture_Appliances': [
        'furniture', 'appliance', 'harvey norman', 'beds', 'mattress'
    ],
    
    'Jewellery': [
        'jewel', 'diamond', 'showcase jewel'
    ],
    
    'Clothing_Fashion': [
        'fashion', 'clothing', 'boutique', 'mens wear', 'ladies wear'
    ]
}

# =========================================================
# KEYWORD-BASED CLASSIFICATION FUNCTION
# =========================================================

def classify_merchant(merchant_name):
    """
    Classify merchant into business type based on keyword matching.
    Returns tuple: (business_type, confidence, matching_keyword)
    """
    if pd.isna(merchant_name):
        return ('Unclassified', 0, '')
    
    merchant_lower = str(merchant_name).lower()
    
    # Track all potential matches
    matches = []
    
    for business_type, keywords in business_type_keywords.items():
        for keyword in keywords:
            if keyword in merchant_lower:
                # Calculate confidence score
                confidence = 70 + len(keyword)  # Longer keywords = higher confidence
                
                # Boost if keyword is at the start
                if merchant_lower.startswith(keyword):
                    confidence += 10
                
                # Boost if keyword is the whole name (exact match)
                if merchant_lower == keyword:
                    confidence += 15
                
                matches.append({
                    'business_type': business_type,
                    'confidence': min(confidence, 99),
                    'keyword': keyword
                })
    
    if not matches:
        return ('Unclassified', 0, '')
    
    # Return the match with highest confidence
    best_match = max(matches, key=lambda x: x['confidence'])
    return (best_match['business_type'], best_match['confidence'], best_match['keyword'])

# =========================================================
# APPLY CLASSIFICATION
# =========================================================

print("\n" + "="*70)
print("CLASSIFYING MERCHANTS BY BUSINESS TYPE")
print("="*70)

merchant_df[['business_type', 'confidence', 'matching_keyword']] = merchant_df['Account Name'].apply(
    lambda x: pd.Series(classify_merchant(x))
)

# =========================================================
# ANALYSIS OF RESULTS
# =========================================================

classified = merchant_df[merchant_df['business_type'] != 'Unclassified']
unclassified = merchant_df[merchant_df['business_type'] == 'Unclassified']

print(f"\nTotal merchants: {len(merchant_df):,}")
print(f"Classified: {len(classified):,} ({len(classified)/len(merchant_df)*100:.1f}%)")
print(f"Unclassified: {len(unclassified):,} ({len(unclassified)/len(merchant_df)*100:.1f}%)")

print("\n" + "="*70)
print("BUSINESS TYPE DISTRIBUTION")
print("="*70)
business_type_counts = merchant_df['business_type'].value_counts()
print(business_type_counts.to_string())

print("\n" + "="*70)
print("CONFIDENCE SCORE STATISTICS")
print("="*70)
print(f"Mean confidence: {classified['confidence'].mean():.1f}")
print(f"Median confidence: {classified['confidence'].median():.1f}")
print(f"Min confidence: {classified['confidence'].min():.1f}")
print(f"Max confidence: {classified['confidence'].max():.1f}")

print("\n" + "="*70)
print("SAMPLE CLASSIFICATIONS (by business type)")
print("="*70)

for business_type in business_type_counts.head(10).index:
    if business_type != 'Unclassified':
        sample = classified[classified['business_type'] == business_type].head(5)
        print(f"\n{business_type}:")
        print(sample[['Account Name', 'confidence', 'matching_keyword']].to_string(index=False))

print("\n" + "="*70)
print("UNCLASSIFIED MERCHANTS (First 30)")
print("="*70)
print(unclassified[['Account Name']].head(30).to_string(index=False))

# =========================================================
# SAVE OUTPUTS
# =========================================================

print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

# Save full merchant file with business types
merchant_df.to_csv(output_dir / '14_merchants_grouped_by_business_type.csv', index=False)
print(f"Saved: {output_dir / 'merchants_grouped_by_business_type.csv'}")

# Create business type summary
business_summary = merchant_df.groupby('business_type').agg({
    'Account Name': 'count',
    'confidence': ['mean', 'min', 'max']
}).round(1)
business_summary.columns = ['merchant_count', 'avg_confidence', 'min_confidence', 'max_confidence']
business_summary = business_summary.sort_values('merchant_count', ascending=False)
business_summary.to_csv(output_dir / '14_business_type_summary.csv')
print(f"Saved: {output_dir / '14_business_type_summary.csv'}")

# Save unclassified merchants for manual review
unclassified[['ATS Number', 'Account Name', 'Discount Offered', 'Discount Offered 2']].to_csv(
    output_dir / '14_merchants_unclassified.csv', index=False
)
print(f"Saved: {output_dir / '14_merchants_unclassified.csv'}")

print("\n" + "="*70)
print("PROCESS COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Review 'merchants_grouped_by_business_type.csv'")
print("2. Manually classify merchants in 'merchants_unclassified.csv'")
print("3. Then map business_type to match_layer labels")