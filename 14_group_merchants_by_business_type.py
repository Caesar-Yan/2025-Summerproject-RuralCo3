"""
Script: 14_map_merchants_via_web_search.py

Purpose:
    Map merchants to match_layer labels by:
    1. Loading filtered merchants from 13.99.1_filtered_merchants_swipe_unmatched.csv
    2. Searching the web for each business description using DuckDuckGo
    3. Using probabilistic matching to assign match_layer labels

Inputs:
    - 13.99.1_filtered_merchants_swipe_unmatched.csv

Outputs:
    - 14_merchants_mapped_via_web_search.csv
    - 14_web_search_mapping_summary.csv
    - 14_merchants_unmapped_web_search.csv
"""

import pandas as pd
import numpy as np
import re
import time
import requests
from bs4 import BeautifulSoup
import urllib.parse
from pathlib import Path

# Set up paths
data_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202')
output_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3/merchant')

# =========================================================
# DEFINE MATCH LAYER KEYWORDS
# =========================================================

match_layer_keywords = {
    'L4_diesel_no_merchant': [
        'diesel', 'fuel', 'petroleum', 'automotive diesel', 'bulk fuel',
        'fuel supplier', 'fuel distributor', 'oil company', 'fuel depot'
    ],
    
    'L4_petrol_no_merchant': [
        'petrol', 'gasoline', 'fuel', 'service station', 'gas station',
        'unleaded', 'premium fuel', 'fuel supplier', 'fuel retail'
    ],
    
    'L5_Gas': [
        'gas', 'lpg', 'lng', 'bottled gas', 'gas supplier', 'propane',
        'natural gas', 'gas bottles', 'energy supplier', 'rockgas', 'elgas'
    ],
    
    'L5_Vet': [
        'veterinary', 'vet', 'animal health', 'veterinarian', 'animal care',
        'livestock health', 'pet care', 'animal hospital', 'vet clinic',
        'animal services', 'farm animals', 'veterinary services'
    ],
    
    'L6_equipment_hire_fuel': [
        'equipment hire', 'machinery rental', 'plant hire', 'hire fuel',
        'contractor equipment', 'rental equipment fuel'
    ],
    
    'L7_cattle_feed': [
        'stock feed', 'animal feed', 'livestock feed', 'cattle feed',
        'farm supplies', 'grain', 'dairy feed', 'agricultural supplies',
        'feed merchant', 'animal nutrition', 'fodder', 'feed supplier'
    ],
    
    'L7_equipment_hire': [
        'equipment hire', 'machinery rental', 'tool hire', 'plant hire',
        'equipment rental', 'hire centre', 'rental services', 'scaffold',
        'machinery hire', 'construction equipment'
    ],
    
    'L7_mechanic': [
        'mechanic', 'mechanical', 'auto repair', 'vehicle repair', 'workshop',
        'garage', 'automotive', 'car service', 'truck repair', 'machinery repair',
        'maintenance', 'engineering services'
    ],
    
    'L8_Landscaping': [
        'landscaping', 'landscape', 'garden', 'nursery', 'lawn care',
        'grounds maintenance', 'horticulture', 'tree services', 'gardening',
        'turf', 'irrigation', 'outdoor services', 'landscape design'
    ],
    
    'L9_infrastructure_consumables': [
        'building supplies', 'construction materials', 'hardware', 'concrete',
        'steel', 'fencing', 'plumbing', 'infrastructure', 'building materials',
        'civil engineering', 'construction supplies', 'aggregate', 'merchant',
        'timber', 'roofing', 'electrical supplies'
    ]
}

# =========================================================
# LOAD FILTERED MERCHANTS
# =========================================================
print("\n" + "="*70)
print("LOADING FILTERED MERCHANTS")
print("="*70)

filtered_merchants_file = output_dir / '13.99.1_filtered_merchants_swipe_unmatched.csv'
merchant_df = pd.read_csv(filtered_merchants_file)

print(f"Loaded {len(merchant_df):,} filtered merchant records")
print(f"Unique merchants: {merchant_df['ATS Number'].nunique():,}")

# Get unique merchants only
unique_merchants = merchant_df[['ATS Number', 'Account Name']].drop_duplicates('ATS Number')
print(f"\nProcessing {len(unique_merchants):,} unique merchant names")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_text(text):
    """Normalize text for keyword matching"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def score_text_against_layer(text, match_layer, keywords):
    """
    Score how well text matches a specific match_layer
    Returns score (0-100) based on keyword presence and frequency
    """
    normalized_text = normalize_text(text)
    
    if not normalized_text:
        return 0, []
    
    score = 0
    matched_keywords = []
    
    for keyword in keywords:
        keyword_norm = normalize_text(keyword)
        if keyword_norm in normalized_text:
            # Base score for presence
            base_score = 10
            
            # Bonus for longer keywords (more specific)
            length_bonus = min(len(keyword_norm) // 3, 5)
            
            # Bonus for multiple occurrences
            count = normalized_text.count(keyword_norm)
            count_bonus = min(count * 2, 10)
            
            # Bonus if keyword appears early in text
            position = normalized_text.find(keyword_norm)
            position_bonus = 5 if position < 50 else 0
            
            keyword_score = base_score + length_bonus + count_bonus + position_bonus
            score += keyword_score
            matched_keywords.append(keyword)
    
    # Normalize score to 0-100 range
    score = min(score, 100)
    
    return score, matched_keywords

def classify_business_description(business_name, search_results):
    """
    Classify business based on search results
    Returns: (match_layer, confidence, matched_keywords)
    """
    # Combine all search result text
    combined_text = f"{business_name} "
    if search_results:
        combined_text += " ".join(search_results)
    
    # Score against each match layer
    layer_scores = {}
    layer_keywords = {}
    
    for match_layer, keywords in match_layer_keywords.items():
        score, matched = score_text_against_layer(combined_text, match_layer, keywords)
        layer_scores[match_layer] = score
        layer_keywords[match_layer] = matched
    
    # Get best match
    if not layer_scores or max(layer_scores.values()) == 0:
        return 'Unmapped', 0, []
    
    best_layer = max(layer_scores, key=layer_scores.get)
    confidence = layer_scores[best_layer]
    keywords_matched = layer_keywords[best_layer]
    
    return best_layer, confidence, keywords_matched

# =========================================================
# WEB SEARCH FUNCTION - DUCKDUCKGO HTML
# =========================================================

def search_business_description(business_name):
    """
    Search for business description using DuckDuckGo HTML
    Returns: list of text snippets about the business
    """
    try:
        # Add "New Zealand" to improve relevance
        query = f"{business_name} New Zealand business"
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"    Error: HTTP {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = []
        
        # Extract snippets from search results
        for result in soup.find_all('a', class_='result__snippet'):
            snippet = result.get_text(strip=True)
            if snippet:
                results.append(snippet)
        
        # Also try alternative snippet class
        if not results:
            for result in soup.find_all('div', class_='result__snippet'):
                snippet = result.get_text(strip=True)
                if snippet:
                    results.append(snippet)
        
        # Extract from result body text
        if not results:
            for result in soup.find_all('div', class_='links_main'):
                text = result.get_text(strip=True)
                if text and len(text) > 20:
                    results.append(text)
        
        print(f"    Found {len(results)} search results")
        
        # Rate limiting - be polite to DuckDuckGo
        time.sleep(2)
        
        return results[:5]  # Return top 5 results
        
    except requests.Timeout:
        print(f"    Timeout searching for: {business_name}")
        return []
    except Exception as e:
        print(f"    Error searching for {business_name}: {type(e).__name__}: {e}")
        return []

# =========================================================
# PROCESS MERCHANTS
# =========================================================

print("\n" + "="*70)
print("SEARCHING AND CLASSIFYING MERCHANTS")
print("="*70)
print("Using DuckDuckGo HTML search (no API required)")
print("This may take a while due to rate limiting...")
print("="*70 + "\n")

results = []
search_failures = 0

for idx, row in unique_merchants.iterrows():
    ats_number = row['ATS Number']
    business_name = row['Account Name']
    
    if idx % 5 == 0:
        print(f"\nProgress: {idx}/{len(unique_merchants)} ({idx/len(unique_merchants)*100:.1f}%)")
        print(f"Failures so far: {search_failures}")
    
    print(f"  {idx+1}. Searching: {business_name}")
    
    # Search for business description
    search_results = search_business_description(business_name)
    
    if not search_results:
        search_failures += 1
    
    # Classify based on search results
    match_layer, confidence, keywords = classify_business_description(
        business_name, search_results
    )
    
    print(f"    → {match_layer} (confidence: {confidence})")
    
    results.append({
        'ATS Number': ats_number,
        'Account Name': business_name,
        'match_layer': match_layer,
        'confidence': confidence,
        'matched_keywords': ', '.join(keywords[:5]) if keywords else '',
        'search_results_found': len(search_results)
    })
    
    # Save progress every 20 merchants
    if (idx + 1) % 20 == 0:
        progress_df = pd.DataFrame(results)
        progress_file = output_dir / '14_merchants_mapped_via_web_search_PROGRESS.csv'
        progress_df.to_csv(progress_file, index=False)
        print(f"    Progress saved to: {progress_file.name}")

# Create results dataframe
results_df = pd.DataFrame(results)

# Merge back with full merchant data
merchant_df = merchant_df.merge(
    results_df[['ATS Number', 'match_layer', 'confidence', 'matched_keywords', 'search_results_found']],
    on='ATS Number',
    how='left'
)

# =========================================================
# ANALYSIS
# =========================================================

print("\n" + "="*70)
print("MAPPING RESULTS")
print("="*70)

mapped = merchant_df[merchant_df['match_layer'] != 'Unmapped']
unmapped = merchant_df[merchant_df['match_layer'] == 'Unmapped']

print(f"\nTotal merchant rows: {len(merchant_df):,}")
print(f"Unique merchants: {merchant_df['ATS Number'].nunique():,}")
print(f"Mapped rows: {len(mapped):,} ({len(mapped)/len(merchant_df)*100:.1f}%)")
print(f"Unmapped rows: {len(unmapped):,} ({len(unmapped)/len(merchant_df)*100:.1f}%)")
print(f"\nSearch failures: {search_failures}")

print("\n" + "="*70)
print("MATCH LAYER DISTRIBUTION")
print("="*70)
print(merchant_df['match_layer'].value_counts().to_string())

if len(mapped) > 0:
    print("\n" + "="*70)
    print("CONFIDENCE STATISTICS (Mapped merchants)")
    print("="*70)
    print(f"Mean confidence: {mapped['confidence'].mean():.1f}")
    print(f"Median confidence: {mapped['confidence'].median():.1f}")
    print(f"Min confidence: {mapped['confidence'].min():.1f}")
    print(f"Max confidence: {mapped['confidence'].max():.1f}")
    
    print("\n" + "="*70)
    print("SEARCH RESULTS STATISTICS")
    print("="*70)
    print(f"Mean search results per merchant: {results_df['search_results_found'].mean():.1f}")
    print(f"Merchants with 0 results: {(results_df['search_results_found'] == 0).sum():,}")
    print(f"Merchants with 1+ results: {(results_df['search_results_found'] > 0).sum():,}")

print("\n" + "="*70)
print("SAMPLE MAPPED MERCHANTS (Top 20)")
print("="*70)
print(mapped[['ATS Number', 'Account Name', 'match_layer', 'confidence', 
              'matched_keywords', 'search_results_found']].head(20).to_string(index=False))

# =========================================================
# SAVE OUTPUTS
# =========================================================

print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

# Save full results
output_file = output_dir / '14_merchants_mapped_via_web_search.csv'
merchant_df.to_csv(output_file, index=False)
print(f"Saved: {output_file.name}")

# Save summary
summary_df = merchant_df.groupby('match_layer').agg({
    'ATS Number': 'nunique',
    'Account Name': 'count',
    'confidence': ['mean', 'min', 'max'],
    'search_results_found': 'mean'
}).round(1)
summary_df.columns = ['unique_merchants', 'total_rows', 'avg_confidence', 'min_confidence', 'max_confidence', 'avg_search_results']
summary_file = output_dir / '14_web_search_mapping_summary.csv'
summary_df.to_csv(summary_file)
print(f"Saved: {summary_file.name}")

# Save unmapped for review
unmapped_unique = unmapped[['ATS Number', 'Account Name', 'search_results_found']].drop_duplicates()
unmapped_file = output_dir / '14_merchants_unmapped_web_search.csv'
unmapped_unique.to_csv(unmapped_file, index=False)
print(f"Saved: {unmapped_file.name} ({len(unmapped_unique):,} unique merchants)")

# Delete progress file if it exists
progress_file = output_dir / '14_merchants_mapped_via_web_search_PROGRESS.csv'
if progress_file.exists():
    progress_file.unlink()
    print(f"Deleted progress file: {progress_file.name}")

print("\n" + "="*70)
print("PROCESS COMPLETE!")
print("="*70)
print(f"\nTotal processing time estimate: ~{len(unique_merchants) * 2 / 60:.1f} minutes")
print(f"(at 2 seconds per merchant)")