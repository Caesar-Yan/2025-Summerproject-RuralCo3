import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Configuration
num_customers = 250
start_date = datetime(2023, 6, 1)
months = 18

# Product categories for agricultural co-op
categories = [
    {'name': 'Feed & Nutrition', 'avg_price': 450, 'std': 200, 'gst_exempt_prob': 0.1,
     'seasonality': [1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.2, 1.2, 1.1, 1.0, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.0, 0.9]},
    {'name': 'Farm Supplies', 'avg_price': 280, 'std': 150, 'gst_exempt_prob': 0.05,
     'seasonality': [0.8, 0.9, 1.2, 1.3, 1.2, 1.0, 0.9, 0.9, 1.0, 1.1, 1.2, 1.1, 0.9, 0.9, 1.2, 1.3, 1.1, 1.0]},
    {'name': 'Fertilizer', 'avg_price': 850, 'std': 300, 'gst_exempt_prob': 0.15,
     'seasonality': [0.7, 0.8, 1.3, 1.4, 1.2, 0.9, 0.8, 0.8, 1.1, 1.2, 1.3, 1.2, 0.8, 0.9, 1.3, 1.4, 1.1, 0.9]},
    {'name': 'Seeds & Plants', 'avg_price': 320, 'std': 120, 'gst_exempt_prob': 0.2,
     'seasonality': [0.6, 0.7, 1.4, 1.5, 1.3, 0.9, 0.7, 0.7, 1.0, 1.2, 1.3, 1.0, 0.7, 0.8, 1.4, 1.5, 1.2, 0.8]},
    {'name': 'Animal Health', 'avg_price': 180, 'std': 80, 'gst_exempt_prob': 0.3,
     'seasonality': [1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0]},
    {'name': 'Equipment Parts', 'avg_price': 520, 'std': 250, 'gst_exempt_prob': 0.0,
     'seasonality': [0.9, 0.9, 1.1, 1.2, 1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.1, 1.0, 0.9, 1.0, 1.1, 1.2, 1.0, 0.9]},
    {'name': 'Irrigation', 'avg_price': 650, 'std': 280, 'gst_exempt_prob': 0.0,
     'seasonality': [0.7, 0.8, 1.0, 1.2, 1.3, 1.2, 1.1, 1.0, 1.0, 1.1, 1.2, 1.0, 0.8, 0.9, 1.1, 1.3, 1.2, 1.0]},
    {'name': 'Fuel & Energy', 'avg_price': 400, 'std': 150, 'gst_exempt_prob': 0.0,
     'seasonality': [1.0, 1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.1, 1.0, 1.0, 1.0, 1.1, 1.2, 1.0, 1.0]}
]

# Customer types
customer_types = [
    {'type': 'Dairy Farmer', 'frequency': 8, 'spend_multiplier': 1.4, 'proportion': 0.35},
    {'type': 'Sheep Farmer', 'frequency': 6, 'spend_multiplier': 1.0, 'proportion': 0.25},
    {'type': 'Arable Farmer', 'frequency': 5, 'spend_multiplier': 1.2, 'proportion': 0.20},
    {'type': 'Mixed Farm', 'frequency': 7, 'spend_multiplier': 1.3, 'proportion': 0.15},
    {'type': 'Small Holder', 'frequency': 3, 'spend_multiplier': 0.5, 'proportion': 0.05}
]

# Generate customers
customers = []
cust_id = 1000

for cust_type in customer_types:
    count = int(num_customers * cust_type['proportion'])
    for i in range(count):
        customers.append({
            'customer_id': f"CUST{cust_id}",
            'type': cust_type['type'],
            'frequency': max(1, int(np.random.normal(cust_type['frequency'], 2))),
            'spend_multiplier': cust_type['spend_multiplier'] * np.random.uniform(0.8, 1.2),
            'loyalty_score': np.random.uniform(0.3, 1.0),
            'discount_scheme': np.random.random() > 0.5  # 50% in scheme
        })
        cust_id += 1

customers_df = pd.DataFrame(customers)

# Generate transactions
transactions = []
txn_id = 10000

for _, customer in customers_df.iterrows():
    for month in range(months):
        # Discount scheme starts at month 6 (Dec 2023)
        in_scheme = month >= 6 and customer['discount_scheme']
        
        # Customers in scheme become more loyal and spend slightly more
        loyalty_boost = 1.15 if in_scheme else 1.0
        frequency_this_month = max(1, int(customer['frequency'] * loyalty_boost * np.random.uniform(0.8, 1.2)))
        
        for txn in range(frequency_this_month):
            # Generate transaction date
            day = np.random.randint(1, 29)
            txn_date = start_date + timedelta(days=30*month + day)
            
            # Skip if date is in future
            if txn_date > datetime(2024, 11, 30):
                continue
            
            # Select category and calculate amount
            category = categories[np.random.randint(0, len(categories))]
            seasonal_factor = category['seasonality'][month]
            
            # Base transaction amount (before any discounts or GST)
            base_amount = max(50, np.random.normal(
                category['avg_price'] * customer['spend_multiplier'] * seasonal_factor,
                category['std']
            ))
            
            # Early settlement discount (0-15%, random)
            early_settlement_discount_pct = np.random.uniform(0, 15)
            early_settlement_discount_amount = base_amount * (early_settlement_discount_pct / 100)
            
            # Days until early settlement discount expires (0-14 days)
            days_until_discount_expires = np.random.randint(0, 15)
            
            # Amount after early settlement discount
            amount_after_discount = base_amount - early_settlement_discount_amount
            
            # GST exemption
            is_gst_exempt = np.random.random() < category['gst_exempt_prob']
            gst_amount = 0 if is_gst_exempt else amount_after_discount * 0.15
            
            # Transaction amount (includes GST)
            transaction_amount = amount_after_discount + gst_amount
            
            # Retention scheme discount (2% cashback on transaction amount)
            retention_scheme_discount = transaction_amount * 0.02 if in_scheme else 0
            
            # Final net amount
            net_amount = transaction_amount - retention_scheme_discount
            
            transactions.append({
                'transaction_id': f"TXN{txn_id}",
                'customer_id': customer['customer_id'],
                'customer_type': customer['type'],
                'date': txn_date.strftime('%Y-%m-%d'),
                'category': category['name'],
                'base_amount': round(base_amount, 2),
                'early_settlement_discount_pct': round(early_settlement_discount_pct, 2),
                'early_settlement_discount_amount': round(early_settlement_discount_amount, 2),
                'days_until_discount_expires': days_until_discount_expires,
                'amount_after_early_discount': round(amount_after_discount, 2),
                'is_gst_exempt': is_gst_exempt,
                'gst_amount': round(gst_amount, 2),
                'transaction_amount': round(transaction_amount, 2),
                'in_retention_scheme': 'Yes' if in_scheme else 'No',
                'retention_scheme_discount': round(retention_scheme_discount, 2),
                'net_amount': round(net_amount, 2),
                'month': month + 1,
                'quarter': f"Q{(month // 3) + 1}"
            })
            txn_id += 1

transactions_df = pd.DataFrame(transactions)

# Display summary statistics
print(f"Dataset Generated Successfully!")
print(f"\nCustomers: {len(customers_df)}")
print(f"Transactions: {len(transactions_df)}")
print(f"\nCustomers in Retention Scheme: {customers_df['discount_scheme'].sum()}")
print(f"Customers in Control Group: {(~customers_df['discount_scheme']).sum()}")
print(f"\nTotal Base Amount: ${transactions_df['base_amount'].sum():,.2f}")
print(f"Total Early Settlement Discounts: ${transactions_df['early_settlement_discount_amount'].sum():,.2f}")
print(f"Total GST Collected: ${transactions_df['gst_amount'].sum():,.2f}")
print(f"Total Transaction Amount: ${transactions_df['transaction_amount'].sum():,.2f}")
print(f"Total Retention Scheme Discounts: ${transactions_df['retention_scheme_discount'].sum():,.2f}")
print(f"Total Net Revenue: ${transactions_df['net_amount'].sum():,.2f}")
print(f"\nGST Exempt Transactions: {transactions_df['is_gst_exempt'].sum()} ({transactions_df['is_gst_exempt'].sum()/len(transactions_df)*100:.1f}%)")
print(f"Average Early Settlement Discount: {transactions_df['early_settlement_discount_pct'].mean():.2f}%")
print(f"Average Days to Discount Expiry: {transactions_df['days_until_discount_expires'].mean():.1f} days")

print("\n--- Sample Transactions ---")
print(transactions_df.head(10))
print(len(transactions_df))

# Save to CSV if desired
# transactions_df.to_csv('agricultural_coop_transactions.csv', index=False)
# customers_df.to_csv('agricultural_coop_customers.csv', index=False)