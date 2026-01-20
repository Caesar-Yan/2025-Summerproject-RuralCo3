# ============================================================================================================
# VALIDATE DISCOUNTED VS UNDISCOUNTED PRICE RELATIONSHIP
# ============================================================================================================

price_check_df = imputed_ats_invoice_line_item_df.copy()

# Only check rows where both prices are present
valid_price_mask = (
    price_check_df['undiscounted_price'].notnull() &
    price_check_df['discounted_price'].notnull()
)

price_check_df = price_check_df[valid_price_mask]

# Create comparison flags
price_check_df['price_relation'] = np.where(
    price_check_df['discounted_price'] < price_check_df['undiscounted_price'],
    'discounted_less',
    np.where(
        price_check_df['discounted_price'] == price_check_df['undiscounted_price'],
        'equal_price',
        'discounted_greater'
    )
)

# Summary counts
relation_counts = price_check_df['price_relation'].value_counts()
print("\nDiscounted vs Undiscounted price relationship summary:")
print(relation_counts)

# Percentage breakdown
relation_percentages = relation_counts / len(price_check_df) * 100
print("\nPercentage breakdown:")
print(relation_percentages.round(2))

# Extract problematic rows where discounted > undiscounted
invalid_price_df = price_check_df[
    price_check_df['price_relation'] == 'discounted_greater'
]

print(f"\nNumber of rows where discounted_price > undiscounted_price: {len(invalid_price_df)}")

# Save invalid rows for inspection (if any)
if len(invalid_price_df) > 0:
    invalid_price_df.to_csv('invalid_discount_price_rows.csv', index=False)
    print("⚠ Saved invalid rows to invalid_discount_price_rows.csv")
else:
    print("✓ No violations found: discounted_price never exceeds undiscounted_price")


#  ========================================================================
 # Check quantity validity
invalid_quantity_df = imputed_ats_invoice_line_item_df[
    imputed_ats_invoice_line_item_df['quantity'] <= 0
]

print(f"Rows with quantity <= 0: {len(invalid_quantity_df)}")

if len(invalid_quantity_df) > 0:
    invalid_quantity_df.to_csv('invalid_quantity_rows.csv', index=False)


# =============================================================================== 
#Identify negative discount values, which should not occur under normal pricing rules.

negative_discount_df = imputed_ats_invoice_line_item_df[
    imputed_ats_invoice_line_item_df['discount_offered'] < 0
]

print(f"Rows with discount_offered < 0: {len(negative_discount_df)}")

# =============================================================================
# Check net amount not exceeding gross amount

invalid_net_gross_df = imputed_ats_invoice_line_item_df[
    imputed_ats_invoice_line_item_df['line_net_amt_received'] >
    imputed_ats_invoice_line_item_df['line_gross_amt_received']
]

print(f"Rows where net > gross: {len(invalid_net_gross_df)}")

# ===============================================================================
# Verify line-level gross amount matches unit price × quantity (within rounding tolerance).

gross_check_df = imputed_ats_invoice_line_item_df.copy()

gross_check_df['gross_calc'] = (
    gross_check_df['unit_gross_amt_received'] *
    gross_check_df['quantity']
).round(2)

gross_check_df['gross_diff'] = (
    gross_check_df['line_gross_amt_received'].round(2) -
    gross_check_df['gross_calc']
)

gross_mismatch_df = gross_check_df[
    gross_check_df['gross_diff'].abs() > 0.01
]

print(f"Rows with gross mismatch > 1 cent: {len(gross_mismatch_df)}")

# ================================================================================
# Detect cases where discount exceeds gross amount (only valid for specific edge cases).

over_discount_df = imputed_ats_invoice_line_item_df[
    imputed_ats_invoice_line_item_df['discount_offered'] >
    imputed_ats_invoice_line_item_df['line_gross_amt_received']
]

print(f"Rows where discount_offered > line_gross_amt_received: {len(over_discount_df)}")

# ================================================================================
#  Validate consistency between pricing flags and their implied numeric relationships.

regular_df = imputed_ats_invoice_line_item_df[
    imputed_ats_invoice_line_item_df['flag'] == 'discount_regular'
]

regular_df['check_diff'] = (
    regular_df['undiscounted_price'] -
    regular_df['discounted_price'] -
    regular_df['discount_offered']
).round(2)

violations = regular_df[regular_df['check_diff'] != 0]
print(f"discount_regular flag violations: {len(violations)}")