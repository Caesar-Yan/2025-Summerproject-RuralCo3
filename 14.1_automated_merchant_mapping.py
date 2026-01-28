"""
Script: 14.1_automated_merchant_mapping.py

Purpose:
    Use Google Places API to map merchants that have no industry label from manual mapping.

Inputs:
    - 14_merchants_no_industry_label.csv
    - 14_filtered_merchants_with_manual_labels.csv
    - Google API Key

Outputs:
    - 14.1_merchants_mapped_google_api.csv (unmapped merchants with Google API results)
    - 14.1_filtered_merchants_complete.csv (full dataset with manual + automated mappings)
    - 14.1_automated_mapping_summary.csv
"""

import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path

# Set up paths
data_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202')
output_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3/merchant')

# =========================================================
# GOOGLE API KEY - SET YOUR KEY HERE
# =========================================================
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

# =========================================================
# GOOGLE PLACE TYPES TO INDUSTRY/MATCH_LAYER MAPPING
# =========================================================

google_type_to_mapping = {
    # Format: 'google_type': ('industry_label', 'match_layer')
    
    # Fuel/Petrol
    'gas_station': ('petrol', 'L4_petrol_no_merchant'),
    
    # Veterinary
    'veterinary_care': ('veterinary', 'L5_Vet'),
    
    # Equipment/Machinery Hire
    'car_rental': ('equipment_hire', 'L7_equipment_hire'),
    'moving_company': ('equipment_hire', 'L7_equipment_hire'),
    
    # Mechanics/Auto
    'car_repair': ('mechanic', 'L7_mechanic'),
    'car_dealer': ('automotive', 'L7_mechanic'),
    'car_wash': ('automotive', 'L7_mechanic'),
    'auto_parts_store': ('automotive', 'L7_mechanic'),
    
    # Landscaping/Garden
    'florist': ('landscaping', 'L8_Landscaping'),
    'garden_center': ('landscaping', 'L8_Landscaping'),
    
    # Hardware/Building supplies
    'hardware_store': ('hardware', 'L9_infrastructure_consumables'),
    'home_goods_store': ('hardware', 'L9_infrastructure_consumables'),
    'plumber': ('trades', 'L9_infrastructure_consumables'),
    'electrician': ('trades', 'L9_infrastructure_consumables'),
    'roofing_contractor': ('trades', 'L9_infrastructure_consumables'),
    'general_contractor': ('construction', 'L9_infrastructure_consumables'),
}

# Keyword-based fallback mapping
keyword_to_mapping = {
    # Format: ('industry_label', 'match_layer'): [keywords]
    
    ('diesel', 'L4_diesel_no_merchant'): ['diesel', 'bulk fuel'],
    ('petrol', 'L4_petrol_no_merchant'): ['petrol', 'fuel', 'bp', 'z energy', 'mobil', 'caltex'],
    ('gas', 'L5_Gas'): ['lpg', 'rockgas', 'elgas', 'bottled gas', 'propane', 'gas bottles'],
    ('veterinary', 'L5_Vet'): ['veterinary', 'animal health', 'animal care'],
    ('stock_feed', 'L7_cattle_feed'): ['stock feed', 'animal feed', 'cattle feed', 'grain', 'fodder'],
    ('equipment_hire', 'L7_equipment_hire'): ['hire', 'rental', 'equipment rental', 'tool hire'],
    ('mechanic', 'L7_mechanic'): ['mechanic', 'auto repair', 'garage', 'workshop'],
    ('landscaping', 'L8_Landscaping'): ['landscape', 'garden', 'nursery', 'lawn', 'tree service'],
    ('hardware', 'L9_infrastructure_consumables'): ['hardware', 'building supplies', 'plumbing', 
                                                     'timber', 'concrete', 'steel', 'mitre 10', 
                                                     'placemakers'],
}

# =========================================================
# LOAD DATA
# =========================================================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Load merchants with no industry label
unmapped_file = output_dir / '14_merchants_no_industry_label.csv'
unmapped_df = pd.read_csv(unmapped_file)
print(f"Loaded {len(unmapped_df):,} rows with no industry label")

# Load full merchant list with manual labels
full_file = output_dir / '14_filtered_merchants_with_manual_labels.csv'
full_df = pd.read_csv(full_file)
print(f"Loaded {len(full_df):,} total rows from full dataset")

# Get unique merchants to process
unique_unmapped = unmapped_df[['ATS Number', 'Account Name']].drop_duplicates('ATS Number')
print(f"\nUnique merchants to process with Google API: {len(unique_unmapped):,}")

# =========================================================
# GOOGLE PLACES API FUNCTIONS
# =========================================================

def search_google_places(business_name, location="New Zealand", api_key=None):
    """Search Google Places API for business information"""
    if not api_key:
        print("Error: No API key provided")
        return None
    
    try:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        
        params = {
            'query': f"{business_name} {location}",
            'key': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                place = data['results'][0]
                
                return {
                    'name': place.get('name'),
                    'types': place.get('types', []),
                    'formatted_address': place.get('formatted_address', ''),
                    'rating': place.get('rating'),
                    'place_id': place.get('place_id')
                }
            elif data.get('status') == 'ZERO_RESULTS':
                return {'status': 'not_found'}
            elif data.get('status') == 'OVER_QUERY_LIMIT':
                print("    ⚠️  API quota exceeded!")
                return {'status': 'quota_exceeded'}
            else:
                return {'status': data.get('status')}
        else:
            print(f"    HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"    Error: {type(e).__name__}: {e}")
        return None

def classify_business(business_name, place_data):
    """
    Classify business based on Google Places data
    Returns: (industry_label, match_layer, confidence, source, matched_types)
    """
    if not place_data or 'types' not in place_data:
        return None, None, 0, 'no_data', ''
    
    place_types = place_data.get('types', [])
    place_name = place_data.get('name', business_name).lower()
    
    # First, try to match Google Place types
    for place_type in place_types:
        if place_type in google_type_to_mapping:
            industry_label, match_layer = google_type_to_mapping[place_type]
            return industry_label, match_layer, 85, 'google_type', place_type
    
    # Second, try keyword matching on business name
    combined_text = f"{business_name} {place_name}".lower()
    
    best_match = None
    best_confidence = 0
    best_keywords = []
    
    for (industry_label, match_layer), keywords in keyword_to_mapping.items():
        score = 0
        matched = []
        
        for keyword in keywords:
            if keyword.lower() in combined_text:
                score += 10 + len(keyword)
                matched.append(keyword)
        
        if score > best_confidence:
            best_confidence = score
            best_match = (industry_label, match_layer)
            best_keywords = matched
    
    if best_match and best_confidence > 15:
        industry_label, match_layer = best_match
        return industry_label, match_layer, min(best_confidence, 80), 'keyword_match', ', '.join(best_keywords[:3])
    
    return None, None, 0, 'no_match', ''

# =========================================================
# PROCESS MERCHANTS WITH GOOGLE API
# =========================================================

print("\n" + "="*70)
print("PROCESSING MERCHANTS WITH GOOGLE PLACES API")
print("="*70)

if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
    print("\n❌ ERROR: Please set your Google API key in the script!")
    print("Set GOOGLE_API_KEY = 'your-actual-api-key'\n")
    exit()

estimated_cost = len(unique_unmapped) * 0.017
print(f"Estimated cost: ${estimated_cost:.2f} USD")
print(f"Processing {len(unique_unmapped):,} unique merchants")
print("="*70 + "\n")

results = []
api_calls = 0
quota_exceeded = False

for idx, row in unique_unmapped.iterrows():
    if quota_exceeded:
        # Add remaining as unmapped
        results.append({
            'ATS Number': row['ATS Number'],
            'Account Name': row['Account Name'],
            'industry_label': None,
            'match_layer': None,
            'confidence': 0,
            'mapping_source': 'quota_exceeded',
            'matched_info': '',
            'google_place_name': '',
            'google_address': ''
        })
        continue
    
    ats_number = row['ATS Number']
    business_name = row['Account Name']
    
    if idx % 10 == 0:
        print(f"\nProgress: {idx}/{len(unique_unmapped)} ({idx/len(unique_unmapped)*100:.1f}%)")
        print(f"API calls: {api_calls}, Cost: ${api_calls * 0.017:.2f}")
    
    print(f"  {idx+1}. {business_name}")
    
    # Search Google Places
    place_data = search_google_places(business_name, "New Zealand", GOOGLE_API_KEY)
    api_calls += 1
    
    if place_data and place_data.get('status') == 'quota_exceeded':
        quota_exceeded = True
    
    # Classify business
    industry_label, match_layer, confidence, source, matched_info = classify_business(
        business_name, place_data
    )
    
    if industry_label:
        print(f"    → {industry_label} / {match_layer} (conf: {confidence}, src: {source})")
    else:
        print(f"    → Unmapped")
    
    results.append({
        'ATS Number': ats_number,
        'Account Name': business_name,
        'industry_label': industry_label,
        'match_layer': match_layer,
        'confidence': confidence,
        'mapping_source': source if industry_label else 'unmapped',
        'matched_info': matched_info,
        'google_place_name': place_data.get('name', '') if place_data else '',
        'google_address': place_data.get('formatted_address', '') if place_data else ''
    })
    
    # Save progress every 50
    if (idx + 1) % 50 == 0:
        progress_df = pd.DataFrame(results)
        progress_file = output_dir / '14.1_automated_mapping_PROGRESS.csv'
        progress_df.to_csv(progress_file, index=False)
        print(f"    Progress saved")
    
    time.sleep(0.1)

print(f"\n\nTotal API calls: {api_calls}")
print(f"Total cost: ${api_calls * 0.017:.2f} USD")

# =========================================================
# CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

# Merge automated results back with unmapped data
unmapped_df = unmapped_df.merge(
    results_df[['ATS Number', 'industry_label', 'match_layer', 'confidence', 
                'mapping_source', 'matched_info', 'google_place_name', 'google_address']],
    on='ATS Number',
    how='left',
    suffixes=('', '_google')
)

# Update columns (use Google results)
unmapped_df['industry_label'] = unmapped_df['industry_label_google']
unmapped_df['match_layer'] = unmapped_df['match_layer_google']
unmapped_df['mapping_source'] = unmapped_df['mapping_source_google']

# Drop duplicate columns
cols_to_drop = [col for col in unmapped_df.columns if col.endswith('_google')]
unmapped_df = unmapped_df.drop(columns=cols_to_drop)

# =========================================================
# MERGE WITH FULL DATASET
# =========================================================

print("\n" + "="*70)
print("MERGING WITH FULL DATASET")
print("="*70)

# Update full dataset with automated mappings
full_df_updated = full_df.copy()

# For rows that were in unmapped_df, update with Google API results
for idx, row in unmapped_df.iterrows():
    mask = (full_df_updated['ATS Number'] == row['ATS Number']) & \
           (full_df_updated['Account Name'] == row['Account Name'])
    
    if row['industry_label'] is not None and pd.notna(row['industry_label']):
        full_df_updated.loc[mask, 'industry_label'] = row['industry_label']
        full_df_updated.loc[mask, 'match_layer'] = row['match_layer']
        full_df_updated.loc[mask, 'mapping_source'] = row['mapping_source']
        full_df_updated.loc[mask, 'confidence'] = row.get('confidence', 0)
        full_df_updated.loc[mask, 'matched_info'] = row.get('matched_info', '')

# =========================================================
# ANALYSIS
# =========================================================

print("\n" + "="*70)
print("AUTOMATED MAPPING RESULTS")
print("="*70)

google_mapped = unmapped_df[unmapped_df['industry_label'].notna()]
still_unmapped = unmapped_df[unmapped_df['industry_label'].isna()]

print(f"\nRows processed: {len(unmapped_df):,}")
print(f"Mapped by Google API: {len(google_mapped):,} ({len(google_mapped)/len(unmapped_df)*100:.1f}%)")
print(f"Still unmapped: {len(still_unmapped):,} ({len(still_unmapped)/len(unmapped_df)*100:.1f}%)")

print(f"\nUnique merchants:")
print(f"Mapped: {google_mapped['ATS Number'].nunique():,}")
print(f"Still unmapped: {still_unmapped['ATS Number'].nunique():,}")

if len(google_mapped) > 0:
    print("\n" + "="*70)
    print("INDUSTRY LABEL DISTRIBUTION (Google API)")
    print("="*70)
    print(google_mapped['industry_label'].value_counts().to_string())
    
    print("\n" + "="*70)
    print("MAPPING SOURCE DISTRIBUTION")
    print("="*70)
    print(google_mapped['mapping_source'].value_counts().to_string())

# Full dataset statistics
print("\n" + "="*70)
print("COMPLETE DATASET STATISTICS")
print("="*70)

total_mapped = full_df_updated[full_df_updated['industry_label'].notna()]
total_unmapped = full_df_updated[full_df_updated['industry_label'].isna()]

print(f"\nTotal rows: {len(full_df_updated):,}")
print(f"Total mapped: {len(total_mapped):,} ({len(total_mapped)/len(full_df_updated)*100:.1f}%)")
print(f"Total unmapped: {len(total_unmapped):,} ({len(total_unmapped)/len(full_df_updated)*100:.1f}%)")

# =========================================================
# SAVE OUTPUTS
# =========================================================

print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

# 1. Save Google API results for unmapped merchants
google_output = output_dir / '14.1_merchants_mapped_google_api.csv'
unmapped_df.to_csv(google_output, index=False)
print(f"Saved: {google_output.name} ({len(unmapped_df):,} rows)")

# 2. Save complete dataset with all mappings
complete_output = output_dir / '14.1_filtered_merchants_complete.csv'
full_df_updated.to_csv(complete_output, index=False)
print(f"Saved: {complete_output.name} ({len(full_df_updated):,} rows)")

# 3. Save summary
summary_df = full_df_updated.groupby(['industry_label', 'match_layer']).agg({
    'ATS Number': 'nunique',
    'Account Name': 'count'
})
summary_df.columns = ['unique_merchants', 'total_rows']
summary_df = summary_df.sort_values('total_rows', ascending=False)
summary_file = output_dir / '14.1_automated_mapping_summary.csv'
summary_df.to_csv(summary_file)
print(f"Saved: {summary_file.name}")

# Delete progress file
progress_file = output_dir / '14.1_automated_mapping_PROGRESS.csv'
if progress_file.exists():
    progress_file.unlink()

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "="*70)
print("PROCESS COMPLETE!")
print("="*70)

print(f"\n💰 COST:")
print(f"  • Total API calls: {api_calls:,}")
print(f"  • Total cost: ${api_calls * 0.017:.2f} USD")

print(f"\n📊 AUTOMATED MAPPING:")
print(f"  • Processed: {len(unmapped_df):,} rows")
print(f"  • Mapped: {len(google_mapped):,} ({len(google_mapped)/len(unmapped_df)*100:.1f}%)")
print(f"  • Still unmapped: {len(still_unmapped):,} ({len(still_unmapped)/len(unmapped_df)*100:.1f}%)")

print(f"\n🎯 COMPLETE DATASET:")
print(f"  • Total rows: {len(full_df_updated):,}")
print(f"  • Mapped (manual + automated): {len(total_mapped):,} ({len(total_mapped)/len(full_df_updated)*100:.1f}%)")
print(f"  • Unmapped: {len(total_unmapped):,} ({len(total_unmapped)/len(full_df_updated)*100:.1f}%)")

print(f"\n📁 OUTPUT FILES:")
print(f"  1. {google_output.name}")
print(f"     - Google API results for previously unmapped merchants")
print(f"  2. {complete_output.name}")
print(f"     - Complete dataset with manual + automated mappings")
print(f"  3. {summary_file.name}")
print(f"     - Summary by industry_label and match_layer")

print("\n✅ Use '14.1_filtered_merchants_complete.csv' as your final working file!")