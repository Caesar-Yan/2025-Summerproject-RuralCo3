import pandas as pd

# lord data
df_ats = pd.read_csv("ats_invoice.csv", low_memory=False)
df_inv = pd.read_csv("invoice.csv", low_memory=False)

ats_cols = ["id", "merchant_id", "member_id"]

invoice_cols = [
    "id",
    "merchant_id",
    "member_id",
    "file_id",
    "invoice_number",
    "amtx_file_id",
    "cms_barcode_ref_number"
]


def overlap_stats(df1, c1, df2, c2):
    s1 = set(df1[c1].dropna().astype(str))
    s2 = set(df2[c2].dropna().astype(str))
    inter = s1 & s2

    return {
        "ats_col": c1,
        "invoice_col": c2,
        "ats_unique": len(s1),
        "invoice_unique": len(s2),
        "overlap_count": len(inter),
        "overlap_ratio_vs_ats": len(inter) / len(s1) if s1 else 0,
        "overlap_ratio_vs_invoice": len(inter) / len(s2) if s2 else 0
    }

results = []

for c_ats in ats_cols:
    for c_inv in invoice_cols:
        if c_ats in df_ats.columns and c_inv in df_inv.columns:
            stats = overlap_stats(df_ats, c_ats, df_inv, c_inv)
            results.append(stats)

df_overlap = pd.DataFrame(results)

# look at all overlap stats
df_overlap.sort_values("overlap_count", ascending=False)

df_overlap = pd.DataFrame(results)

# only save overlap_count > 0 for easier viewing
df_nonzero = df_overlap[df_overlap["overlap_count"] > 0]

print("\n=== Non-zero overlaps ===")
if len(df_nonzero) == 0:
    print("No overlapping IDs found between ATS and invoice tables.")
else:
    print(df_nonzero.sort_values("overlap_count", ascending=False))