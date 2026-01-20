"""
In this project, we need to evaluate both the direct financial gains from canceling
the late-payment discount and the potential impact on customers. The final objective
is to determine whether the net benefit can cover the associated costs, and to estimate
the remaining surplus.

Therefore, the code structure can be divided into two main parts:

1. Calculate the direct revenue generated from discount cancellation.
2. Analyze the customer impact caused by discount removal, including potential
   customer churn and the resulting financial loss.

Below is an initial implementation for Part 1: computing the direct revenue from
discount cancellation. Further adjustments may be made as needed based on the data
and project requirements.
"""

import pandas as pd

def calculate_extra_revenue(df,
                            invoice_col='invoice_amount',
                            discount_rate_col='discount_rate',
                            days_late_col='days_late',
                            cutoff_days=14):
    """
    Calculate extra revenue gained from discount cancellation
    when late payment exceeds the cutoff threshold.

    Parameters:
    df (DataFrame): Input data containing invoice and payment info
    invoice_col (str): Column name for invoice amount
    discount_rate_col (str): Column name for discount rate
    days_late_col (str): Column name for late days
    cutoff_days (int): Threshold for canceling discount (default = 14)

    Returns:
    DataFrame: original df with extra columns
    float: total extra revenue for the company
    """

    # Compute the original discount amount
    df['original_discount_amount'] = df[invoice_col] * df[discount_rate_col]

    # Determine if discount should be canceled
    df['discount_canceled'] = df[days_late_col] > cutoff_days

    # Extra revenue occurs only when discount is canceled
    df['extra_revenue'] = df['original_discount_amount'].where(
        df['discount_canceled'], 0
    )

    # Total extra revenue
    total_extra_revenue = df['extra_revenue'].sum()

    return df, total_extra_revenue

# Some examples    
df = pd.DataFrame({
    'customer_id': ['C001', 'C002', 'C003', 'C004'],
    'invoice_amount': [500, 1200, 800, 700],
    'discount_rate': [0.02, 0.015, 0.02, 0.03],
    'days_late': [15, 0, 20, 5]
})

df_result, total_revenue = calculate_extra_revenue(df)

print(df_result[['customer_id', 'extra_revenue']])
print(f"\nTotal extra revenue: ${total_revenue:.2f}")