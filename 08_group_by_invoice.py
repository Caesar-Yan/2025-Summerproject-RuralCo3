import pandas as pd
import numpy as np
import pickle

# ================================================================
# Load the two imputed line-item datasets
# ================================================================
ats_path = "datetime_parsed_imputed_ats_invoice_line_item.csv"
invoice_path = "datetime_parsed_imputed_invoice_line_item.csv"

ats = pd.read_csv(ats_path)
invoice = pd.read_csv(invoice_path)

