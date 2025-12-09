
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

imputed_invoice_df = invoice_df
imputed_invoice_line_item_df = invoice_line_item_df

# ======================================= imputing invoice_line_item ==================================================================== #
print(imputed_invoice_line_item_df.columns)

# Save to CSV
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False, mode='w')

imputed_invoice_line_item_df = imputed_invoice_line_item_df[[
    'Unnamed', 'produce_code', 'description', 'quantity', 'unit_gross_amt_received',
    'discount_offered', 'line_gross_amt_received', 'line_gst_amt_received', 'line_net_amt_received',
    'extended_line_description1', 'extended_line_description2',
    'gst_indicator', 'unit_gst_amt_derived', 'unit_gross_amt_derived', 'line_discount_derived', 
    'line_net_amt_derived', 'line_gst_total_derived', 'line_gross_amt_derived', 'merchant_identifier',
    'merchant_branch', 'gst_rate',
    'amount_excluding_gst', 'discount_delta', 'total_discount_delta',
    'line_gross_amt_derived_excl_gst', 'extras'
]]

# Save to CSV
Path('imputed_invoice_line_item.csv').unlink(missing_ok=True)
imputed_invoice_line_item_df.to_csv('imputed_invoice_line_item.csv', index=False)
