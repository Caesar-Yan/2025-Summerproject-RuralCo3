import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================================================================
# Load and clean data
# ================================================================
ats_grouped = pd.read_csv('ats_grouped_with_discounts.csv')
invoice_grouped = pd.read_csv('invoice_grouped_with_discounts.csv')

print(f"Original ATS data: {len(ats_grouped):,} invoices")
print(f"Original Invoice data: {len(invoice_grouped):,} invoices")

# Filter outliers
OUTLIER_THRESHOLD = 1_000_000
ats_clean = ats_grouped[ats_grouped['discount_amount'] <= OUTLIER_THRESHOLD].copy()
invoice_clean = invoice_grouped[invoice_grouped['discount_amount'] <= OUTLIER_THRESHOLD].copy()

print(f"\nCleaned ATS: {len(ats_clean):,} invoices")
print(f"Cleaned Invoice: {len(invoice_clean):,} invoices")

# Combine
ats_clean['customer_type'] = 'ATS'
invoice_clean['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_clean, invoice_clean], ignore_index=True)

# ================================================================
# Robust date parsing to avoid "1970" junk
# ================================================================
def parse_invoice_period(series: pd.Series) -> pd.Series:
    """
    Robustly parse invoice_period.
    Handles:
      - 'YYYY-MM' / 'YYYY-MM-DD'
      - 'YYYYMM' (e.g., 202312)
      - avoids numeric-to-epoch misparse that often shows up as 1970
    """
    s = series.copy()

    # Convert to string first (prevents pandas treating ints as nanoseconds since epoch)
    s_str = s.astype(str).str.strip()

    # Treat common null-like strings as missing
    s_str = s_str.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})

    # Case 1: looks like YYYYMM (6 digits)
    mask_yyyymm = s_str.str.fullmatch(r"\d{6}", na=False)
    out = pd.Series(pd.NaT, index=s.index)

    if mask_yyyymm.any():
        out.loc[mask_yyyymm] = pd.to_datetime(s_str.loc[mask_yyyymm], format="%Y%m", errors="coerce")

    # Case 2: everything else -> let pandas parse (YYYY-MM, YYYY-MM-DD, etc.)
    mask_other = ~mask_yyyymm
    if mask_other.any():
        out.loc[mask_other] = pd.to_datetime(s_str.loc[mask_other], errors="coerce")

    return out

combined_df['invoice_period'] = parse_invoice_period(combined_df['invoice_period'])

# Drop invalid dates
combined_df = combined_df[combined_df['invoice_period'].notna()].copy()

# Hard filter to remove obvious epoch-era junk like 1970-xx-xx
# (Adjust bounds if your dataset truly includes earlier years)
min_valid = pd.Timestamp("2000-01-01")
max_valid = pd.Timestamp("2035-12-31")
combined_df = combined_df[(combined_df['invoice_period'] >= min_valid) & (combined_df['invoice_period'] <= max_valid)].copy()

combined_df = combined_df.sort_values('invoice_period')

# ================================================================
# Group by period (you are treating invoice_period as month)
# ================================================================
period_summary = combined_df.groupby('invoice_period').agg({
    'discount_amount': 'sum',
    'total_discounted_price': 'sum',
    'total_undiscounted_price': 'sum'
}).reset_index()

# Calculate cumulative values
period_summary['cumulative_discount'] = period_summary['discount_amount'].cumsum()
period_summary['cumulative_discounted_revenue'] = period_summary['total_discounted_price'].cumsum()
period_summary['cumulative_undiscounted_revenue'] = period_summary['total_undiscounted_price'].cumsum()

# Calculate revenue under different default scenarios
default_rate_1pct = 0.01
default_rate_3pct = 0.03

period_summary['revenue_with_1pct_default'] = (
    period_summary['total_discounted_price'] +
    period_summary['discount_amount'] * default_rate_1pct
)

period_summary['revenue_with_3pct_default'] = (
    period_summary['total_discounted_price'] +
    period_summary['discount_amount'] * default_rate_3pct
)

period_summary['cumulative_revenue_1pct'] = period_summary['revenue_with_1pct_default'].cumsum()
period_summary['cumulative_revenue_3pct'] = period_summary['revenue_with_3pct_default'].cumsum()

# ================================================================
# Breakeven line FIXED at 1,043,000
# ================================================================
BREAKEVEN_REVENUE = 1_043_000  # <- fixed red line y-value

print("\n" + "="*70)
print("CUMULATIVE ANALYSIS")
print("="*70)
print(f"Total cumulative discount given: ${period_summary['cumulative_discount'].iloc[-1]:,.2f}")
print(f"Cumulative undiscounted revenue: ${period_summary['cumulative_undiscounted_revenue'].iloc[-1]:,.2f}")
print(f"Cumulative discounted revenue: ${period_summary['cumulative_discounted_revenue'].iloc[-1]:,.2f}")
print(f"Cumulative revenue (1% default): ${period_summary['cumulative_revenue_1pct'].iloc[-1]:,.2f}")
print(f"Cumulative revenue (3% default): ${period_summary['cumulative_revenue_3pct'].iloc[-1]:,.2f}")
print(f"Breakeven line (fixed): ${BREAKEVEN_REVENUE:,.2f}")

# ================================================================
# Create visualization - Alternative view
# ================================================================
fig, ax = plt.subplots(figsize=(18, 10))

mark_every = max(1, len(period_summary)//20)

ax.plot(period_summary['invoice_period'], period_summary['cumulative_undiscounted_revenue'],
        color='#4472C4', linewidth=3,
        label='Total potential revenue (if no discounts offered)',
        marker='o', markersize=5, markevery=mark_every, linestyle='--')

ax.plot(period_summary['invoice_period'], period_summary['cumulative_revenue_3pct'],
        color='#A64D79', linewidth=3,
        label="Revenue generated (3% customers don't pay on time)",
        marker='s', markersize=5, markevery=mark_every)

ax.plot(period_summary['invoice_period'], period_summary['cumulative_revenue_1pct'],
        color='#70AD47', linewidth=3,
        label="Revenue generated (1% customers don't pay on time)",
        marker='^', markersize=5, markevery=mark_every)

ax.plot(period_summary['invoice_period'], period_summary['cumulative_discounted_revenue'],
        color='#FFC000', linewidth=3,
        label='Revenue if all customers pay on time (with discount)',
        marker='D', markersize=5, markevery=mark_every, linestyle='-.')

# FIX: red line at exactly $1,043,000
ax.axhline(y=BREAKEVEN_REVENUE, color='#FF0000', linewidth=2,
           linestyle='-', alpha=0.7, label='Breakeven ($1,043,000)')

# Format x-axis density
date_range_days = (period_summary['invoice_period'].max() - period_summary['invoice_period'].min()).days
if date_range_days > 730:
    interval = 3
elif date_range_days > 365:
    interval = 2
else:
    interval = 1

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
plt.xticks(rotation=45, ha='right')

# Styling
ax.set_xlabel('Invoice Period (Year-Month)', fontsize=15, fontweight='bold')
ax.set_ylabel('$ Revenue', fontsize=15, fontweight='bold')
ax.set_title('Cumulative Revenue Analysis: Impact of Early Payment Discount Program',
             fontsize=20, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_facecolor('#F8F8F8')
plt.tight_layout()

plt.savefig('revenue_analysis_alternative.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved alternative visualization to: revenue_analysis_alternative.png")
plt.show()

# ================================================================
# Show the discount "gap"
# ================================================================
print("\n" + "="*70)
print("REVENUE GAP ANALYSIS")
print("="*70)

total_discount = period_summary['cumulative_discount'].iloc[-1]
undiscounted_total = period_summary['cumulative_undiscounted_revenue'].iloc[-1]
discounted_total = period_summary['cumulative_discounted_revenue'].iloc[-1]
revenue_1pct = period_summary['cumulative_revenue_1pct'].iloc[-1]
revenue_3pct = period_summary['cumulative_revenue_3pct'].iloc[-1]

print(f"\nIf NO discount program existed:")
print(f"  Maximum potential revenue: ${undiscounted_total:,.2f}")
print(f"\nWith current discount program (all pay on time):")
print(f"  Actual revenue: ${discounted_total:,.2f}")
print(f"  Revenue sacrificed: ${total_discount:,.2f} ({total_discount/undiscounted_total*100:.2f}%)")
print(f"\nWith early payment discount program:")
print(f"  Revenue (1% default): ${revenue_1pct:,.2f}")
print(f"  Revenue gain vs baseline: ${revenue_1pct - discounted_total:,.2f}")
print(f"\n  Revenue (3% default): ${revenue_3pct:,.2f}")
print(f"  Revenue gain vs baseline: ${revenue_3pct - discounted_total:,.2f}")

print("\n" + "="*70)
