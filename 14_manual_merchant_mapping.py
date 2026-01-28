"""
Script: 14_manual_merchant_mapping.py

Purpose:
    Apply manual mapping rules to filtered merchants based on business name patterns.

Inputs:
    - 13.99.1_filtered_merchants_swipe_unmatched.csv

Outputs:
    - 14_filtered_merchants_with_manual_labels.csv (all merchants with industry_label and match_layer)
    - 14_merchants_no_industry_label.csv (only merchants that need automated mapping)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Set up paths
data_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202')
output_dir = Path('T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3/merchant')

# =========================================================
# MANUAL MAPPING RULES
# =========================================================

manual_mapping_rules = [
    # Rule: (pattern, column_to_check, industry_label, match_layer, description)
    # column_to_check can be: 'account_name', 'discount_offered', 'discount_offered_2', 'discount_combined', 'account_or_discount2'
    
    # Appliances
    (r'100%', 'account_name', 'appliances', 'exclude', 'Contains 100%'),
    (r'appliances', 'account_name', 'appliances', 'exclude', 'Contains appliances'),
    (r'harvey\s+norman', 'account_name', 'appliances', 'exclude', 'Harvey Norman'),
    
    # Petrol/Service Stations
    (r'\bAPL\b', 'account_name', 'petrol', 'L4_petrol_no_merchant', 'Contains APL (capital)'),
    (r'service\s+station', 'account_name', 'petrol', 'L4_petrol_no_merchant', 'Service station'),
    (r'\bmobil\b', 'account_name', 'petrol', 'L4_petrol_no_merchant', 'Contains mobil (exact word)'),
    (r'\bNPD\b', 'account_name', 'petrol', 'L4_petrol_no_merchant', 'Contains NPD'),
    (r'per\s+litre\s+discount', 'discount_combined', 'petrol', 'L4_petrol_no_merchant', 'Discount contains per litre discount'),
    
    # Gas/LPG
    (r'\bgas\b', 'account_name', 'gas', 'L5_Gas', 'Contains gas'),
    (r'lpg', 'discount_offered_2', 'gas', 'L5_Gas', 'Discount Offered 2 contains LPG'),
    
    # Travel/Accommodation
    (r'ASURE', 'account_name', 'travel', 'exclude', 'Contains ASURE'),
    (r'house\s+of\s+travel', 'account_name', 'travel', 'exclude', 'House of Travel'),
    (r'\bMCK\b', 'account_name', 'travel', 'exclude', 'Contains MCK (capital)'),
    (r'motel', 'account_name', 'travel', 'exclude', 'Contains motel'),
    (r'world\s+travellers', 'account_name', 'travel', 'exclude', 'World Travellers'),
    (r'accommodation|accomodation', 'discount_offered_2', 'travel', 'exclude', 'Discount Offered 2 contains accommodation'),
    
    # Groceries/Supermarkets
    (r'four\s+square', 'account_name', 'groceries', 'exclude', 'Four Square'),
    (r'fresh\s*choice', 'account_name', 'groceries', 'exclude', 'FreshChoice'),
    (r'new\s+world\b', 'account_name', 'groceries', 'exclude', 'New World (exact words)'),
    (r"pak.{0,3}save", 'account_name', 'groceries', 'exclude', "Pak'n'Save"),
    (r'supervalue', 'account_name', 'groceries', 'exclude', 'Supervalue'),
    
    # Hunting/Firearms
    (r'gun\s+city', 'account_name', 'hunting', 'exclude', 'Gun City'),
    (r'firearms', 'discount_offered_2', 'hunting', 'exclude', 'Discount Offered 2 contains firearms'),
    
    # Hardware/Home Improvement
    (r'guthrie\s+bowron', 'account_name', 'hardware', 'L9_infrastructure_consumables', 'Guthrie Bowron'),
    (r'paint', 'discount_offered_2', 'hardware', 'L9_infrastructure_consumables', 'Discount Offered 2 contains paint'),
    (r'building', 'discount_offered_2', 'hardware', 'L9_infrastructure_consumables', 'Discount Offered 2 contains building'),
    
    # Optometrists
    (r'matthews\s+eyewear', 'account_name', 'optometrist', 'exclude', 'Matthews Eyewear'),
    (r'visique', 'account_name', 'optometrist', 'exclude', 'Visique'),
    (r'eye', 'discount_offered_2', 'optometrist', 'exclude', 'Discount Offered 2 contains eye'),
    (r'eye', 'account_name', 'optometrist', 'exclude', 'Contains eye'),
    (r'eye', 'account_or_discount2', 'optometrist', 'exclude', 'Contains eye'),
    
    # Liquor
    (r'liquor', 'account_name', 'liquor', 'exclude', 'Contains liquor'),
    (r'bottle', 'discount_offered_2', 'liquor', 'exclude', 'Discount Offered 2 contains bottle'),
    (r'tavern', 'account_name', 'liquor', 'exclude', 'Contains tavern'),
    
    # Pharmacy
    (r'pharmacy', 'account_name', 'pharmacy', 'exclude', 'Contains pharmacy'),
    
    # Veterinary
    (r'\bvets?\b', 'account_name', 'veterinary', 'L5_Vet', 'Contains vet/vets'),
    
    # Furniture
    (r'furniture', 'discount_offered_2', 'furniture', 'exclude', 'Discount Offered 2 contains furniture'),
    
    # Jewellers
    (r'jeweller', 'account_name', 'jewellers', 'exclude', 'Contains jewellers'),
    
    # Cycles
    (r'cycle', 'account_name', 'cycles', 'exclude', 'Contains cycle'),
    
    # Shoes and Clothing
    (r'shoe', 'account_or_discount2', 'shoes_and_clothing', 'exclude', 'Contains shoe'),
    (r'footwear', 'account_or_discount2', 'shoes_and_clothing', 'exclude', 'Contains footwear'),
    
    # Butcher
    (r'butcher', 'account_or_discount2', 'butcher', 'exclude', 'Contains butcher'),

    # Machinery/Mechanic
(r'machinery', 'account_name', 'mechanic', 'L7_mechanic', 'Contains machinery'),

# Sporting goods
(r'sport', 'account_name', 'sporting_goods', 'exclude', 'Contains sport'),

# Food/Hospitality
(r'cafe', 'account_name', 'food', 'exclude', 'Contains cafe'),
(r'restaurant', 'account_name', 'food', 'exclude', 'Contains restaurant'),
(r'\binn\b', 'account_name', 'food', 'exclude', 'Contains inn'),
(r'bake', 'account_name', 'food', 'exclude', 'Contains bake'),

# Appliances (singular)
(r'appliance', 'account_name', 'appliances', 'exclude', 'Contains appliance'),

# Hardware/Flooring
(r'floor', 'account_name', 'hardware', 'L9_infrastructure_consumables', 'Contains floor'),

# Furniture/Mattress
(r'mattress', 'account_name', 'furniture', 'exclude', 'Contains mattress'),

# Clothing/Footwear
(r'wear', 'account_name', 'shoes_and_clothing', 'exclude', 'Contains wear'),
]

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
print(f"Columns: {merchant_df.columns.tolist()}")

# =========================================================
# APPLY MANUAL MAPPING RULES
# =========================================================
print("\n" + "="*70)
print("APPLYING MANUAL MAPPING RULES")
print("="*70)

def apply_manual_rules(account_name, discount_offered, discount_offered_2):
    """
    Apply manual mapping rules to business name and discount columns
    Returns: (industry_label, match_layer, rule_description) or (None, None, None) if no match
    """
    if pd.isna(account_name):
        account_name = ""
    if pd.isna(discount_offered):
        discount_offered = ""
    if pd.isna(discount_offered_2):
        discount_offered_2 = ""
    
    # Convert to strings for matching
    name_lower = str(account_name).lower()
    discount_1_lower = str(discount_offered).lower()
    discount_2_lower = str(discount_offered_2).lower()
    discount_combined = f"{discount_1_lower} {discount_2_lower}"
    account_or_discount2 = f"{name_lower} {discount_2_lower}"
    
    for pattern, column_to_check, industry_label, match_layer, description in manual_mapping_rules:
        # Select the appropriate column to check
        if column_to_check == 'account_name':
            text_to_check = name_lower
        elif column_to_check == 'discount_offered':
            text_to_check = discount_1_lower
        elif column_to_check == 'discount_offered_2':
            text_to_check = discount_2_lower
        elif column_to_check == 'discount_combined':
            text_to_check = discount_combined
        elif column_to_check == 'account_or_discount2':
            text_to_check = account_or_discount2
        else:
            continue
        
        # Use case-insensitive search
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return industry_label, match_layer, description
    
    return None, None, None

# Apply rules to each merchant
merchant_df[['industry_label', 'match_layer', 'mapping_rule']] = merchant_df.apply(
    lambda row: pd.Series(apply_manual_rules(row['Account Name'], row['Discount Offered'], row['Discount Offered 2'])),
    axis=1
)

# Add mapping source
merchant_df['mapping_source'] = merchant_df['industry_label'].apply(
    lambda x: 'manual_rule' if pd.notna(x) else None
)

# =========================================================
# ANALYSIS
# =========================================================
print("\n" + "="*70)
print("MANUAL MAPPING RESULTS")
print("="*70)

manual_matched = merchant_df[merchant_df['industry_label'].notna()]
manual_unmapped = merchant_df[merchant_df['industry_label'].isna()]

print(f"\nRows:")
print(f"  Manually mapped: {len(manual_matched):,} ({len(manual_matched)/len(merchant_df)*100:.1f}%)")
print(f"  No industry label: {len(manual_unmapped):,} ({len(manual_unmapped)/len(merchant_df)*100:.1f}%)")

print(f"\nUnique merchants:")
print(f"  Manually mapped: {manual_matched['ATS Number'].nunique():,}")
print(f"  No industry label: {manual_unmapped['ATS Number'].nunique():,}")

print("\n" + "="*70)
print("INDUSTRY LABEL DISTRIBUTION")
print("="*70)
print(merchant_df['industry_label'].value_counts().to_string())

print("\n" + "="*70)
print("MATCH LAYER DISTRIBUTION")
print("="*70)
print(merchant_df['match_layer'].value_counts().to_string())

print("\n" + "="*70)
print("MAPPING RULES APPLIED (Counts)")
print("="*70)
rule_counts = merchant_df[merchant_df['mapping_rule'].notna()]['mapping_rule'].value_counts()
print(rule_counts.to_string())

print("\n" + "="*70)
print("SAMPLE MANUALLY MAPPED MERCHANTS (First 30)")
print("="*70)
print(manual_matched[['ATS Number', 'Account Name', 'industry_label', 
                      'match_layer', 'mapping_rule']].head(30).to_string(index=False))

print("\n" + "="*70)
print("SAMPLE UNMAPPED MERCHANTS (First 20)")
print("="*70)
print(manual_unmapped[['ATS Number', 'Account Name', 'Discount Offered', 
                       'Discount Offered 2']].head(20).to_string(index=False))

# =========================================================
# SAVE OUTPUTS
# =========================================================
print("\n" + "="*70)
print("SAVING OUTPUTS")
print("="*70)

# 1. Save full merchant list with manual labels
output_file_all = output_dir / '14_filtered_merchants_with_manual_labels.csv'
merchant_df.to_csv(output_file_all, index=False)
print(f"Saved: {output_file_all.name} ({len(merchant_df):,} rows)")

# 2. Save merchants with no industry label (for automated mapping)
output_file_unmapped = output_dir / '14_merchants_no_industry_label.csv'
manual_unmapped.to_csv(output_file_unmapped, index=False)
print(f"Saved: {output_file_unmapped.name} ({len(manual_unmapped):,} rows)")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "="*70)
print("PROCESS COMPLETE!")
print("="*70)

print(f"\n📊 SUMMARY:")
print(f"  • Total rows: {len(merchant_df):,}")
print(f"  • Manually mapped: {len(manual_matched):,} ({len(manual_matched)/len(merchant_df)*100:.1f}%)")
print(f"  • Need automated mapping: {len(manual_unmapped):,} ({len(manual_unmapped)/len(merchant_df)*100:.1f}%)")

print(f"\n🏢 UNIQUE MERCHANTS:")
print(f"  • Total: {merchant_df['ATS Number'].nunique():,}")
print(f"  • Manually mapped: {manual_matched['ATS Number'].nunique():,}")
print(f"  • Need automated mapping: {manual_unmapped['ATS Number'].nunique():,}")

print(f"\n📁 OUTPUT FILES:")
print(f"  1. {output_file_all.name}")
print(f"     - All merchants with industry_label and match_layer columns")
print(f"     - Use this as your main working file")
print(f"  2. {output_file_unmapped.name}")
print(f"     - Only merchants without industry labels")
print(f"     - Feed this to 14.1_automated_merchant_mapping.py")

print("\n📋 NEXT STEP:")
print("Run 14.1_automated_merchant_mapping.py to process merchants with no industry label")