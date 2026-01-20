'''
Docstring for 03.1_hardcode_corrections_ATS

this script is hardcoding some corrections in the raw data.
the corrections are made because the values are obviously recording errors, e.g. price is 10 million instaed of 10 thousand

inputs:
- imputed_ats_invoice_line_item.csv

outputs:
- imputed_ats_invoice_line_item.csv
    overwrites same csv with updated figures

'''



import pandas as pd
import numpy as np
from pathlib import Path

# Define base directories (matching your main script)
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
output_dir = base_dir / "data_cleaning"

# Load the imputed dataframe
imputed_ats_invoice_line_item_df = pd.read_csv(output_dir / 'imputed_ats_invoice_line_item.csv')

print(f"Original dataframe shape: {imputed_ats_invoice_line_item_df.shape}")

# ============================================================================================================================
# HARDCODE CORRECTIONS TO ORIGINAL DATA
# ============================================================================================================================

# Define all hardcoded changes
# Format: {Unnamed: 0 value: {'column_name': new_value}}

hardcoded_changes = {
    # First group - exponential notation corrections
    90780: {
        'undiscounted_price': 365.13,
        'discount_offered': 365.13
    },
    91076: {
        'undiscounted_price': 730.25,
        'discount_offered': 730.25
    },
    91690: {
        'undiscounted_price': 73.11,
        'discount_offered': 73.11
    },
    92056: {
        'undiscounted_price': 365.13,
        'discount_offered': 365.13
    },
    
    # Second group - large number corrections
    149240: {
        'undiscounted_price': 104210.00,
        'discount_offered': 25032.50
    },
    176071: {
        'undiscounted_price': 109310.00,
        'discount_offered': 30132.50
    },
    178831: {
        'undiscounted_price': 104880.00,
        'discount_offered': 19816.80
    },
    186266: {
        'undiscounted_price': 109310.00,
        'discount_offered': 30132.50
    },
    191241: {
        'undiscounted_price': 108524.80,
        'discount_offered': 17372.40
    },
    191242: {
        'undiscounted_price': 144573.00,
        'discount_offered': 23133.00
    },
    191243: {
        'undiscounted_price': 140893.50,
        'discount_offered': 22552.20
    },
    342228: {
        'undiscounted_price': 128128.00,
        'discount_offered': 37689.60
    },
    365576: {
        'undiscounted_price': 240955.00,
        'discount_offered': 40482.75
    }
}

# Apply all hardcoded changes
print("\nApplying hardcoded corrections:")
for unnamed_0_value, changes_dict in hardcoded_changes.items():
    # Check if the row exists
    if (imputed_ats_invoice_line_item_df['Unnamed: 0'] == unnamed_0_value).any():
        print(f"\n  Unnamed: 0 = {unnamed_0_value}:")
        
        for column_name, new_value in changes_dict.items():
            # Get original value
            original_value = imputed_ats_invoice_line_item_df.loc[
                imputed_ats_invoice_line_item_df['Unnamed: 0'] == unnamed_0_value, 
                column_name
            ].values[0]
            
            # Apply the change
            imputed_ats_invoice_line_item_df.loc[
                imputed_ats_invoice_line_item_df['Unnamed: 0'] == unnamed_0_value, 
                column_name
            ] = new_value
            
            print(f"    {column_name}: {original_value} → {new_value}")
    else:
        print(f"  WARNING: Unnamed: 0 = {unnamed_0_value} not found in dataframe!")

# Verify all changes
print("\n" + "="*80)
print("Verification of all hardcoded changes:")
print("="*80)
for unnamed_0_value in hardcoded_changes.keys():
    if (imputed_ats_invoice_line_item_df['Unnamed: 0'] == unnamed_0_value).any():
        row_data = imputed_ats_invoice_line_item_df[
            imputed_ats_invoice_line_item_df['Unnamed: 0'] == unnamed_0_value
        ][['Unnamed: 0', 'undiscounted_price', 'discounted_price', 'discount_offered', 'quantity']]
        print(f"\nUnnamed: 0 = {unnamed_0_value}:")
        print(row_data.to_string(index=False))

# Save to CSV
imputed_ats_invoice_line_item_df.to_csv(output_dir / 'imputed_ats_invoice_line_item.csv', index=False, mode='w')
print(f"\n{'='*80}")
print(f"Saved updated dataframe to {output_dir / 'imputed_ats_invoice_line_item.csv'}")
print(f"Final dataframe shape: {imputed_ats_invoice_line_item_df.shape}")