import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime, timedelta

# random seed
np.random.seed(42)
random.seed(42)

# Helper functions

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randrange(delta.days))

start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 1, 1)

def random_invoice_number():
    if random.random() < 0.5:
        return f"{random.randint(10000,99999)}-{random.randint(100,999)}-{random.randint(100,999)}"
    else:
        return str(random.randint(10000000,999999999))

def random_auth_code():
    if random.random() < 0.4:
        return ""
    else:
        return str(random.randint(1, 999999))

def random_inv_auth_status():
    length = random.randint(1, 3)
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=length))

def random_extras():
    data = {
        "identifyMemberFields": {
            "isFound": random.choice([True, False]),
            "memberId": str(random.randint(1000, 9999)),
            "accountNumber": str(random.randint(1000, 9999)),
            "cardActivated": random.choice([True, False]),
            "failureReason": random.choice(["", "TIMEOUT", "INVALID_CARD"]),
            "processStatus": random.choice(["PROCESSED", "FAILED", "PENDING"]),
            "migrationStatus": random.choice(["CMS MIGRATED CUSTOMER", "LEGACY CUSTOMER"]),
            "memberIdentifier": "ACCOUNTNUMBER",
            "paymentProcessor": random.choice(["FISERV", "PAYMARK", "BAMBORA"]),
            "cardActivationDate": (
                start_date + timedelta(days=random.randint(0, 700))
            ).isoformat() + "Z",
        }
    }
    return json.dumps(data)

def maybe_empty_amount():
    if random.random() < 0.3:
        return ""
    else:
        return round(random.uniform(10, 20000), 2)

def maybe_rebates():
    if random.random() < 0.8:
        return ""
    else:
        return round(random.uniform(1, 200), 2)

def maybe_float():
    if random.random() < 0.6:
        return ""
    else:
        return round(random.uniform(1, 500), 2)

def maybe_text():
    return "" if random.random() < 0.7 else random.choice(["INFO", "DESC", "DATA"])

def random_time():
    h = random.randint(0,23)
    m = random.randint(0,59)
    s = random.randint(0,59)
    return f"{h:02d}:{m:02d}:{s:02d}"


# generate invoice 

n = 100

df_invoice = pd.DataFrame({
    "created_at": [(start_date + timedelta(days=random.randint(0,700))).isoformat()+"+00:00" for _ in range(n)],
    "updated_at": [(start_date + timedelta(days=random.randint(0,700))).isoformat()+"+00:00" for _ in range(n)],
    "id": [str(uuid.uuid4()) for _ in range(n)],
    "file_id": [str(uuid.uuid4()) for _ in range(n)],
    "merchant_id": [random.randint(10000, 99999) for _ in range(n)],
    "member_id": [random.randint(1000, 9999) for _ in range(n)],
    "date": [(start_date + timedelta(days=random.randint(0,700))).strftime("%Y-%m-%d") for _ in range(n)],
    "total_amount": [round(random.uniform(10, 20000), 2) for _ in range(n)],
    "invoice_number": [random_invoice_number() for _ in range(n)],
    "gst_amount": [round(random.uniform(0, 500), 2) for _ in range(n)],
    "gross_transaction_amount": [maybe_empty_amount() for _ in range(n)],
    "merchant_gst_number": [f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(100,999)}" for _ in range(n)],
    "rebates": [maybe_rebates() for _ in range(n)],
    "extras": ["{}" for _ in range(n)],
    "debitCreditIndicator": random.choices(["C", "D"], k=n),
    "inv_auth_status": [random_inv_auth_status() for _ in range(n)],
    "inv_tracker": ["".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(1,3))) for _ in range(n)],
    "cms_transaction_status": random.choices(["OK", "REJECTED", "PENDING", ""], k=n),
    "auth_code": [random_auth_code() for _ in range(n)],
    "pdf_inv_category": random.choices(["AUTHORISED", "STANDARD", "DETAILED", ""], k=n),
    "process_status": random.choices(["PROCESSED", "FAILED", "PENDING", ""], k=n),
    "failure_reason": random.choices(["", "TIMEOUT", "INVALID", "SYSTEM_ERROR"], k=n),
    "discount_delta": random.choices(["", round(random.uniform(-20,20),2)], k=n),
    "total_discount_delta": random.choices(["", round(random.uniform(-50,50),2)], k=n),
    "rejected_stage": random.choices(["", "STAGE1", "STAGE2"], k=n),
    "rejected_reason": random.choices(["", "AUTH_FAIL", "DATA_INVALID"], k=n),
    "amtx_file_id": [str(uuid.uuid4()) for _ in range(n)],
    "payment_processor": random.choices(["FISERV", "PAYMARK", "BAMBORA"], k=n),
    "cms_barcode_ref_number": [f"BC{random.randint(100000,999999)}" for _ in range(n)],
    "location_identifier": random.choices(["LOC1", "LOC2", "LOC3", ""], k=n),
    "audit": random.choices([True, False], k=n),
    "manual_review_status": random.choices(["", "FLAGGED", "REVIEWED"], k=n),
    "extracted_invoice_date": [(start_date + timedelta(days=random.randint(0,700))).strftime("%Y-%m-%d") for _ in range(n)],
    "cms_process_attempts": [random.randint(1,5) for _ in range(n)],
    "lbmx_file_name": [f"file_{i}.txt" for i in range(n)],
    "pdf_generated_file_name": [f"invoice_{i}.pdf" for i in range(n)],
    "mrp_comment": random.choices(["", "CHECK_AMOUNT", "VERIFY_CUSTOMER"], k=n)
})


# generate invoice_line_item 

def generate_invoice_line_item(df_invoice, max_items=5):

    line_items = []

    for _, inv in df_invoice.iterrows():
        invoice_id = inv["id"]
        n_lines = random.randint(1, max_items)

        for i in range(n_lines):

            qty = round(random.uniform(1, 50), 2)
            line_total = round(qty * random.uniform(2, 4), 2)

            line_items.append({
                "created_at": inv["created_at"],
                "updated_at": inv["updated_at"],
                "id": str(uuid.uuid4()),
                "invoice_id": invoice_id,
                "product_code": f"{random.randint(1,99):02d}",
                "description": random.choice(["Petrol 91", "Petrol 95/96", "Diesel", "Farm Supplies", "Merchandise"]),
                "quantity": qty,
                "unit_gross_amt_received": maybe_float(),
                "discount_offered": maybe_float(),
                "line_gross_amt_received": line_total,
                "discount_type": maybe_text(),
                "measure_unit": random.choice(["L","KG","EA",""]),
                "line_gst_amt_received": maybe_float(),
                "line_net_amt_received": maybe_float(),
                "extended_line_description1": maybe_text(),
                "extended_line_description2": maybe_text(),
                "gst_indicator": random.choice(["GST","","Z"]),
                "unit_gst_amt_derived": maybe_float(),
                "unit_gross_amt_derived": maybe_float(),
                "line_discount_derived": maybe_float(),
                "line_net_amt_derived": maybe_float(),
                "line_gst_total_derived": maybe_float(),
                "line_gross_amt_derived": maybe_float(),
                "cardholder_identifier": "",
                "card_holder_name1": "",
                "card_holder_name2": "",
                "card_holder_name3": "",
                "transaction_date": inv["date"],
                "merchant_identifier": str(random.randint(100000,999999)),
                "merchant_branch": random.choice(["01","02","03",""]),
                "debit_credit_indicator": random.choice(["D","C"]),
                "gst_rate": random.choice([0.15,None]),
                "amount_excluding_gst": round(line_total / 1.15, 2),
                "price_zone": random.choice(["A","B","C",""]),
                "transaction_time": random_time(),
                "discount_delta": maybe_float(),
                "total_discount_delta": maybe_float(),
                "line_gross_amt_derived_excl_gst": maybe_float(),
                "extras": "{}"
            })

    return pd.DataFrame(line_items)

df_invoice_line_item = generate_invoice_line_item(df_invoice)


# output

output_invoice = "F:/学习/DATA 605 Summer project/mock_invoice_data.csv"
df_invoice.to_csv(output_invoice, index=False, encoding="utf-8")

output_line_item = "F:/学习/DATA 605 Summer project/mock_invoice_line_item_data.csv"
df_invoice_line_item.to_csv(output_line_item, index=False, encoding="utf-8")

print("生成完成！")
print("invoice 行数:", df_invoice.shape)
print("line item 行数:", df_invoice_line_item.shape)
print("文件已保存至：")
print(output_invoice)
print(output_line_item)
