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
min_valid = pd.Timestamp("2000-01-01")
max_valid = pd.Timestamp("2035-12-31")
combined_df = combined_df[(combined_df['invoice_period'] >= min_valid) & (combined_df['invoice_period'] <= max_valid)].copy()

combined_df = combined_df.sort_values('invoice_period')

# ================================================================
# Group by period - FOCUS ON DISCOUNTS ONLY
# ================================================================
period_summary = combined_df.groupby('invoice_period').agg({
    'discount_amount': 'sum'
}).reset_index()

# Rename for clarity
period_summary.rename(columns={'discount_amount': 'total_discount_offered'}, inplace=True)

# Calculate discount RETAINED under different default scenarios
# (Default = customer doesn't pay on time, so Ruralco keeps the discount)
default_rate_1pct = 0.01
default_rate_3pct = 0.03

period_summary['discount_retained_1pct'] = period_summary['total_discount_offered'] * default_rate_1pct
period_summary['discount_retained_3pct'] = period_summary['total_discount_offered'] * default_rate_3pct

# Calculate cumulative values
period_summary['cumulative_discount_offered'] = period_summary['total_discount_offered'].cumsum()
period_summary['cumulative_retained_1pct'] = period_summary['discount_retained_1pct'].cumsum()
period_summary['cumulative_retained_3pct'] = period_summary['discount_retained_3pct'].cumsum()

# ================================================================
# Breakeven line FIXED at 1,043,000
# ================================================================
BREAKEVEN_DISCOUNT_RETAINED = 1_043_000  # <- fixed red line y-value

print("\n" + "="*70)
print("CUMULATIVE DISCOUNT RETENTION ANALYSIS")
print("="*70)
print(f"Total discount offered to customers: ${period_summary['cumulative_discount_offered'].iloc[-1]:,.2f}")
print(f"Discount retained (1% default rate): ${period_summary['cumulative_retained_1pct'].iloc[-1]:,.2f}")
print(f"Discount retained (3% default rate): ${period_summary['cumulative_retained_3pct'].iloc[-1]:,.2f}")
print(f"Breakeven threshold: ${BREAKEVEN_DISCOUNT_RETAINED:,.2f}")

# Check if breakeven is reached
retained_1pct = period_summary['cumulative_retained_1pct'].iloc[-1]
retained_3pct = period_summary['cumulative_retained_3pct'].iloc[-1]

print(f"\nBreakeven status (1% default): {'✓ REACHED' if retained_1pct >= BREAKEVEN_DISCOUNT_RETAINED else '✗ NOT REACHED'}")
print(f"  Gap: ${retained_1pct - BREAKEVEN_DISCOUNT_RETAINED:,.2f}")

print(f"\nBreakeven status (3% default): {'✓ REACHED' if retained_3pct >= BREAKEVEN_DISCOUNT_RETAINED else '✗ NOT REACHED'}")
print(f"  Gap: ${retained_3pct - BREAKEVEN_DISCOUNT_RETAINED:,.2f}")

# ================================================================
# Create visualization - Discount Retention Focus
# ================================================================
fig, ax = plt.subplots(figsize=(18, 10))

mark_every = max(1, len(period_summary)//20)

# Plot cumulative discount offered (reference line)
ax.plot(period_summary['invoice_period'], period_summary['cumulative_discount_offered'],
        color='#C0C0C0', linewidth=2.5, alpha=0.6,
        label='Total discount offered (if all customers pay early)',
        marker='o', markersize=4, markevery=mark_every, linestyle='--')

# Plot discount retained scenarios
ax.plot(period_summary['invoice_period'], period_summary['cumulative_retained_3pct'],
        color='#A64D79', linewidth=3,
        label="Discount retained by Ruralco (3% customers don't pay early)",
        marker='s', markersize=5, markevery=mark_every)

ax.plot(period_summary['invoice_period'], period_summary['cumulative_retained_1pct'],
        color='#70AD47', linewidth=3,
        label="Discount retained by Ruralco (1% customers don't pay early)",
        marker='^', markersize=5, markevery=mark_every)

# Breakeven line at exactly $1,043,000
ax.axhline(y=BREAKEVEN_DISCOUNT_RETAINED, color='#FF0000', linewidth=2.5,
           linestyle='-', alpha=0.8, label='Breakeven threshold ($1,043,000)')

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
ax.set_ylabel('$ Discount Amount', fontsize=15, fontweight='bold')
ax.set_title('Cumulative Discount Retention Analysis: Early Payment Discount Program ROI',
             fontsize=20, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.set_facecolor('#F8F8F8')
plt.tight_layout()

plt.savefig('discount_retention_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization to: discount_retention_analysis.png")
plt.show()

# ================================================================
# Detailed Analysis
# ================================================================
print("\n" + "="*70)
print("DISCOUNT RETENTION BREAKDOWN")
print("="*70)

total_offered = period_summary['cumulative_discount_offered'].iloc[-1]
retained_1pct = period_summary['cumulative_retained_1pct'].iloc[-1]
retained_3pct = period_summary['cumulative_retained_3pct'].iloc[-1]

print(f"\nTotal discount offered: ${total_offered:,.2f}")
print(f"\nUnder 1% default scenario:")
print(f"  Discount retained: ${retained_1pct:,.2f} ({retained_1pct/total_offered*100:.2f}% of offered)")
print(f"  Discount given away: ${total_offered - retained_1pct:,.2f} ({(1-retained_1pct/total_offered)*100:.2f}% of offered)")

print(f"\nUnder 3% default scenario:")
print(f"  Discount retained: ${retained_3pct:,.2f} ({retained_3pct/total_offered*100:.2f}% of offered)")
print(f"  Discount given away: ${total_offered - retained_3pct:,.2f} ({(1-retained_3pct/total_offered)*100:.2f}% of offered)")

print(f"\nBreakeven analysis:")
print(f"  Target to break even: ${BREAKEVEN_DISCOUNT_RETAINED:,.2f}")
print(f"  Default rate needed: {(BREAKEVEN_DISCOUNT_RETAINED / total_offered * 100):.2f}%")

print("\n" + "="*70)