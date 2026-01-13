import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================================================================
# Config
# ================================================================

# Use the latest datetime-parsed & imputed datasets
ATS_CANDIDATES = [
    "datetime_parsed_ats_invoice_line_item_df.csv",
]

INVOICE_CANDIDATES = [
    "datetime_parsed_invoice_line_item_df.csv",
]


# Which time column to use for the curve (priority order).
# If date_imputed exists, it is preferred; otherwise invoice_period; otherwise date_from_invoice.
TIME_COL_PRIORITY = ["date_imputed", "invoice_period", "date_from_invoice", "transaction_date"]

# Scenario assumptions (same as your sketch)
LATE_RATES = [0.01, 0.03]  # 1% and 3%

# Breakeven reference line (relative to current scheme = 0)
BREAKEVEN_Y = 0.0

# Optional extreme filter to avoid explosive outliers ruining the plot
EXTREME_DISCOUNT_THRESHOLD = 1_000_000

# Outputs
OUT_FIG = "discount_scenario_curves.png"
OUT_TABLE = "discount_scenario_timeseries.csv"


# ================================================================
# Helpers
# ================================================================
def pick_existing(candidates):
    """Return the first existing file path from a list of candidates."""
    for f in candidates:
        if Path(f).exists():
            return f
    raise FileNotFoundError(f"No candidate files found. Tried: {candidates}")


def choose_time_col(df):
    """Pick the best available time column based on TIME_COL_PRIORITY."""
    for c in TIME_COL_PRIORITY:
        if c in df.columns:
            return c
    raise ValueError(f"No usable time column found. Expected one of: {TIME_COL_PRIORITY}")


def prepare_invoice_level(df, dataset_name):
    """
    Convert line-item data to invoice-level totals:
    - Sum discounted/undiscounted per invoice_id
    - Use a representative time value per invoice (first non-null within invoice)
    """
    required = ["invoice_id", "discounted_price", "undiscounted_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{dataset_name}] Missing required columns: {missing}")

    # Ensure numeric
    df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")
    df["undiscounted_price"] = pd.to_numeric(df["undiscounted_price"], errors="coerce")

    # Pick a time column and parse it
    time_col = choose_time_col(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Drop rows with no invoice_id
    df = df[df["invoice_id"].notna()].copy()

    # Aggregate to invoice level
    inv = (
        df.groupby("invoice_id", as_index=False)
          .agg(
              total_discounted_price=("discounted_price", "sum"),
              total_undiscounted_price=("undiscounted_price", "sum"),
              invoice_time=(time_col, "first"),
          )
    )

    # Compute discount amount per invoice
    inv["discount_amount"] = (inv["total_undiscounted_price"] - inv["total_discounted_price"]).round(2)
    inv["discount_amount"] = inv["discount_amount"].clip(lower=0)

    # Optional extreme removal
    extreme_n = (inv["discount_amount"] > EXTREME_DISCOUNT_THRESHOLD).sum()
    if extreme_n > 0:
        inv.loc[inv["discount_amount"] > EXTREME_DISCOUNT_THRESHOLD].to_csv(
            f"{dataset_name}_extreme_invoices_removed.csv", index=False
        )
        inv = inv[inv["discount_amount"] <= EXTREME_DISCOUNT_THRESHOLD].copy()
        print(f"[{dataset_name}] Removed extreme invoices: {extreme_n}")

    # Drop invoices without a time
    inv = inv[inv["invoice_time"].notna()].copy()

    return inv


# ================================================================
# 1) Load datetime_parsed_imputed datasets (ATS & invoice)
# ================================================================
ats_path = pick_existing(ATS_CANDIDATES)
inv_path = pick_existing(INVOICE_CANDIDATES)

print(f"Using ATS file: {ats_path}")
print(f"Using Invoice file: {inv_path}")

ats_li = pd.read_csv(ats_path, low_memory=False)
inv_li = pd.read_csv(inv_path, low_memory=False)

print(f"Loaded ATS line-items: {len(ats_li):,}")
print(f"Loaded Invoice line-items: {len(inv_li):,}")

# ================================================================
# 2) Convert to invoice-level and combine
# ================================================================
ats_inv = prepare_invoice_level(ats_li, "ATS")
inv_inv = prepare_invoice_level(inv_li, "INVOICE")

all_inv = pd.concat(
    [
        ats_inv.assign(source="ATS"),
        inv_inv.assign(source="INVOICE"),
    ],
    ignore_index=True
)

print(f"Invoice-level rows (ATS): {len(ats_inv):,}")
print(f"Invoice-level rows (Invoice): {len(inv_inv):,}")
print(f"Combined invoice-level rows: {len(all_inv):,}")

# ================================================================
# 3) Build time series and scenario curves
# ================================================================
# Group by time (daily). If you prefer monthly, replace "D" with "M".
all_inv["time_bucket"] = all_inv["invoice_time"].dt.to_period("M").dt.to_timestamp()

ts = (
    all_inv.groupby("time_bucket", as_index=False)
           .agg(
               total_discount=("discount_amount", "sum"),
               invoice_count=("invoice_id", "nunique"),
           )
           .sort_values("time_bucket")
)

ts["cum_total_discount"] = ts["total_discount"].cumsum()

for r in LATE_RATES:
    ts[f"cum_revenue_{int(r*100)}pct_late"] = (ts["cum_total_discount"] * r).round(2)

ts.to_csv(OUT_TABLE, index=False)
print(f"Saved time series table: {OUT_TABLE}")

# ================================================================
# 4) Plot with improved axis scaling and time formatting
# ================================================================

# ----- Scale money to millions for readability -----
SCALE = 1_000_000  # 1 = million dollars

ts_plot = ts.copy()

# Ensure invoice_period is datetime
ts_plot["time_bucket"] = pd.to_datetime(ts_plot["time_bucket"], errors="coerce")

# Drop invalid or clearly incorrect dates
ts_plot = ts_plot[
    (ts_plot["time_bucket"].notna()) &
    (ts_plot["time_bucket"] >= "2023-01-01") &   # adjust if needed
    (ts_plot["time_bucket"] <= "2026-12-31")
].copy()

# Sort again after filtering
ts_plot = ts_plot.sort_values("time_bucket")

ts_plot["cum_total_discount_m"] = ts_plot["cum_total_discount"] / SCALE

for r in LATE_RATES:
    ts_plot[f"cum_revenue_{int(r*100)}pct_late_m"] = (
        ts_plot[f"cum_revenue_{int(r*100)}pct_late"] / SCALE
    )

# ----- Plot -----
plt.figure(figsize=(12, 7))

# Total discounts curve
plt.plot(
    ts_plot["time_bucket"],
    ts_plot["cum_total_discount_m"],
    linewidth=2,
    label="Total discounts accrued (all customers)"
)

# Scenario curves
for r, color in zip(LATE_RATES, ["green", "purple"]):
    plt.plot(
        ts_plot["time_bucket"],
        ts_plot[f"cum_revenue_{int(r*100)}pct_late_m"],
        linewidth=2,
        label=f"Revenue generated ({int(r*100)}% customers pay late)",
        color=color
    )

# Breakeven line (relative to current scheme)
plt.axhline(
    y=BREAKEVEN_Y,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Breakeven (relative to current scheme)"
)

# ----- Axis labels -----
plt.xlabel("Time (Invoice period)", fontsize=11)
plt.ylabel("Cumulative revenue (Million $)", fontsize=11)
plt.title(
    "Cumulative Discounts vs Revenue Scenarios Over Time",
    fontsize=13
)

# ----- Improve time axis readability -----
plt.xticks(
    ts_plot["time_bucket"][::6],  # show one tick every 6 months
    rotation=45
)

# ----- Grid & legend -----
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("discount_scenario_curves_scaled.png", dpi=200)
print("Saved figure: discount_scenario_curves_scaled.png")

plt.show()
