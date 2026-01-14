# i talked to dave about this
# he recommends trying a handful of different approaches, e.g. simple OLS regression, maybe based on 2025 data only because 
# the shape of the invoice totals cruve changes after start of 2025.
# dont let it stop you from using all data also
# can also try a simple time series regression, maybe ChatGPT can give us some suggestions on methods.
# we can look at trying to model month-by-month, or cumulatively. just try whatever you think might works

# Got it. I built this script based on your 10_calculations_with_transformed_data.py and tried a few different approaches.
# And this script forecasts the next 12 months of invoice totals using time-based regression, then estimates credit card revenue via Monte Carlo simulation of payment delays (interest + late fees) under discount vs no-discount scenarios.
# BTW, my data might be missing some cleaning steps, so please run this script in your environment and check if everything looks correct.
# I tried a few different variations in the next script (named 12)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================
ANNUAL_INTEREST_RATE = 0.2395  # 23.95% p.a.
LATE_FEE = 10.00  # $10 per late invoice
RANDOM_SEED = 42
OUTPUT_DIR = "future_12_months_forecast"
FORECAST_MONTHS = 12  # 预测未来12个月

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("CREDIT CARD REVENUE FORECAST - NEXT 12 MONTHS")
print("="*80)

# ================================================================
# STEP 1: Load historical invoice data
# ================================================================
print("\n" + "="*80)
print("STEP 1: LOADING HISTORICAL INVOICE DATA")
print("="*80)

ats_grouped = pd.read_csv('ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv('invoice_grouped_transformed_with_discounts.csv')

ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"Total historical invoices: {len(combined_df):,}")

# Parse dates
def parse_invoice_period(series: pd.Series) -> pd.Series:
    """Robustly parse invoice_period"""
    s = series.copy()
    s_str = s.astype(str).str.strip()
    s_str = s_str.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
    
    mask_yyyymm = s_str.str.fullmatch(r"\d{6}", na=False)
    out = pd.Series(pd.NaT, index=s.index)
    
    if mask_yyyymm.any():
        out.loc[mask_yyyymm] = pd.to_datetime(s_str.loc[mask_yyyymm], format="%Y%m", errors="coerce")
    
    mask_other = ~mask_yyyymm
    if mask_other.any():
        out.loc[mask_other] = pd.to_datetime(s_str.loc[mask_other], errors="coerce")
    
    return out

combined_df['invoice_period'] = parse_invoice_period(combined_df['invoice_period'])
combined_df = combined_df[combined_df['invoice_period'].notna()].copy()

print(f"Valid invoices: {len(combined_df):,}")
print(f"Date range: {combined_df['invoice_period'].min()} to {combined_df['invoice_period'].max()}")

# ================================================================
# STEP 2: Aggregate historical monthly data
# ================================================================
print("\n" + "="*80)
print("STEP 2: AGGREGATING HISTORICAL MONTHLY TOTALS")
print("="*80)

monthly_historical = combined_df.groupby(
    combined_df['invoice_period'].dt.to_period('M')
).agg({
    'total_undiscounted_price': 'sum',
    'total_discounted_price': 'sum',
    'discount_amount': 'sum',
    'invoice_id': 'count'
}).reset_index()

monthly_historical.columns = ['period', 'total_undiscounted', 'total_discounted', 
                              'total_discount', 'invoice_count']
monthly_historical['period'] = monthly_historical['period'].dt.to_timestamp()
monthly_historical = monthly_historical.sort_values('period').reset_index(drop=True)

print(f"Historical months: {len(monthly_historical)}")
print(f"Last historical month: {monthly_historical['period'].max().strftime('%Y-%m')}")

# ================================================================
# STEP 3: Build regression models to predict invoice totals
# ================================================================
print("\n" + "="*80)
print("STEP 3: BUILDING REGRESSION MODELS FOR INVOICE TOTALS")
print("="*80)

# Add time features
monthly_historical['month_index'] = range(len(monthly_historical))
monthly_historical['year'] = monthly_historical['period'].dt.year
monthly_historical['month'] = monthly_historical['period'].dt.month

# Option 1: Use 2025+ data only (as suggested by Dave)
monthly_2025 = monthly_historical[monthly_historical['year'] >= 2025].copy()
monthly_2025['month_index_2025'] = range(len(monthly_2025))

print(f"\nData split:")
print(f"  All data: {len(monthly_historical)} months")
print(f"  2025+ data: {len(monthly_2025)} months")

# Prepare training data
X_all = monthly_historical[['month_index']].values
y_all_undiscounted = monthly_historical['total_undiscounted'].values
y_all_discounted = monthly_historical['total_discounted'].values
y_all_invoice_count = monthly_historical['invoice_count'].values

# Train models for undiscounted amounts
print("\n--- Training Linear Regression (All Data) ---")
model_linear_all = LinearRegression()
model_linear_all.fit(X_all, y_all_undiscounted)
pred_linear_all = model_linear_all.predict(X_all)
r2_linear_all = r2_score(y_all_undiscounted, pred_linear_all)
print(f"R² Score: {r2_linear_all:.4f}")
print(f"Slope: ${model_linear_all.coef_[0]:,.2f}/month")

# Train polynomial model
print("\n--- Training Polynomial Regression (degree=2) ---")
poly_features = PolynomialFeatures(degree=2)
X_poly_all = poly_features.fit_transform(X_all)
model_poly_all = LinearRegression()
model_poly_all.fit(X_poly_all, y_all_undiscounted)
pred_poly_all = model_poly_all.predict(X_poly_all)
r2_poly_all = r2_score(y_all_undiscounted, pred_poly_all)
print(f"R² Score: {r2_poly_all:.4f}")

# Train 2025+ model if enough data
if len(monthly_2025) >= 3:
    print("\n--- Training Linear Regression (2025+ Only) ---")
    X_2025 = monthly_2025[['month_index_2025']].values
    y_2025_undiscounted = monthly_2025['total_undiscounted'].values
    
    model_linear_2025 = LinearRegression()
    model_linear_2025.fit(X_2025, y_2025_undiscounted)
    pred_linear_2025 = model_linear_2025.predict(X_2025)
    r2_linear_2025 = r2_score(y_2025_undiscounted, pred_linear_2025)
    print(f"R² Score: {r2_linear_2025:.4f}")
    print(f"Slope: ${model_linear_2025.coef_[0]:,.2f}/month")
    use_2025_model = True
else:
    print("\n⚠ Not enough 2025 data, will use all-data model")
    use_2025_model = False

# Select best model
if use_2025_model and r2_linear_2025 > r2_linear_all:
    print(f"\n✓ Using 2025+ Linear Model (R²={r2_linear_2025:.4f})")
    best_model = model_linear_2025
    best_model_name = "Linear 2025+"
    is_2025_model = True
elif r2_poly_all > r2_linear_all:
    print(f"\n✓ Using Polynomial Model (R²={r2_poly_all:.4f})")
    best_model = model_poly_all
    best_model_name = "Polynomial (deg=2)"
    is_2025_model = False
else:
    print(f"\n✓ Using Linear Model - All Data (R²={r2_linear_all:.4f})")
    best_model = model_linear_all
    best_model_name = "Linear (All Data)"
    is_2025_model = False

# ================================================================
# STEP 4: Forecast future invoice totals
# ================================================================
print("\n" + "="*80)
print("STEP 4: FORECASTING FUTURE INVOICE TOTALS (NEXT 12 MONTHS)")
print("="*80)

last_month_index = monthly_historical['month_index'].max()
last_date = monthly_historical['period'].max()

# Create future dates
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1), 
    periods=FORECAST_MONTHS, 
    freq='MS'
)

if is_2025_model:
    # Use 2025 model
    last_2025_index = monthly_2025['month_index_2025'].max()
    future_month_indices = np.arange(last_2025_index + 1, last_2025_index + 1 + FORECAST_MONTHS).reshape(-1, 1)
    forecast_undiscounted = best_model.predict(future_month_indices)
else:
    # Use all-data model
    future_month_indices = np.arange(last_month_index + 1, last_month_index + 1 + FORECAST_MONTHS).reshape(-1, 1)
    
    if best_model_name.startswith("Polynomial"):
        future_month_indices_poly = poly_features.transform(future_month_indices)
        forecast_undiscounted = best_model.predict(future_month_indices_poly)
    else:
        forecast_undiscounted = best_model.predict(future_month_indices)

# Ensure non-negative forecasts
forecast_undiscounted = np.maximum(forecast_undiscounted, 0)

# Calculate average discount rate from historical data
avg_discount_rate = (monthly_historical['total_discount'].sum() / 
                     monthly_historical['total_undiscounted'].sum())
print(f"Historical average discount rate: {avg_discount_rate*100:.2f}%")

# Calculate discounted amounts
forecast_discounted = forecast_undiscounted * (1 - avg_discount_rate)
forecast_discount = forecast_undiscounted - forecast_discounted

# Estimate invoice counts (use recent average)
recent_avg_count = monthly_historical.tail(6)['invoice_count'].mean()
forecast_invoice_count = np.full(FORECAST_MONTHS, recent_avg_count)

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'period': future_dates,
    'forecast_undiscounted': forecast_undiscounted,
    'forecast_discounted': forecast_discounted,
    'forecast_discount': forecast_discount,
    'forecast_invoice_count': forecast_invoice_count.astype(int)
})

print("\nForecasted Monthly Invoice Totals:")
print(forecast_df.to_string(index=False))

# ================================================================
# STEP 5: Load payment behavior profiles
# ================================================================
print("\n" + "="*80)
print("STEP 5: LOADING PAYMENT BEHAVIOR PROFILES")
print("="*80)

try:
    with open('payment_profiles/payment_profiles.pkl', 'rb') as f:
        payment_profiles = pickle.load(f)
    print(f"✓ Loaded {len(payment_profiles)} payment profiles")
    profile = payment_profiles['overall']
except FileNotFoundError:
    print("⚠ Payment profiles not found. Using simplified assumptions.")
    # Create fallback profile with typical payment behavior
    # Assume payments follow a normal distribution around 40 days
    mean_payment_days = 40
    std_payment_days = 20
    profile = {
        'raw_data': np.random.normal(mean_payment_days, std_payment_days, 10000).clip(0, 120)
    }
    print(f"✓ Created fallback profile (mean={mean_payment_days} days, std={std_payment_days} days)")

# ================================================================
# STEP 6: Simulate payment behavior for forecasted invoices
# ================================================================
print("\n" + "="*80)
print("STEP 6: SIMULATING PAYMENT BEHAVIOR - TWO SCENARIOS")
print("="*80)

def simulate_future_revenue(forecast_df, payment_profile, discount_scenario, random_seed=RANDOM_SEED):
    """
    Simulate credit card revenue for forecasted invoices
    
    Parameters:
    - forecast_df: DataFrame with forecasted invoice totals
    - payment_profile: Payment timing profile
    - discount_scenario: 'with_discount' or 'no_discount'
    - random_seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with simulated revenue details
    """
    np.random.seed(random_seed)
    
    results = []
    
    for idx, row in forecast_df.iterrows():
        period = row['period']
        n_invoices = int(row['forecast_invoice_count'])
        
        if discount_scenario == 'with_discount':
            total_amount = row['forecast_discounted']
            avg_invoice_amount = total_amount / n_invoices if n_invoices > 0 else 0
        else:
            total_amount = row['forecast_undiscounted']
            avg_invoice_amount = total_amount / n_invoices if n_invoices > 0 else 0
        
        # Simulate payment timing for each invoice
        raw_payment_days = np.random.choice(
            payment_profile['raw_data'], 
            size=n_invoices, 
            replace=True
        )
        
        # Adjust payment days (subtract 20 for due date offset)
        adjusted_payment_days = np.maximum(raw_payment_days - 20, 0)
        
        # Calculate overdue days
        days_overdue = np.maximum(adjusted_payment_days, 0)
        is_late = days_overdue > 0
        
        # Calculate interest and late fees
        daily_rate = ANNUAL_INTEREST_RATE / 365
        
        # Create individual invoice amounts (with some variation)
        invoice_amounts = np.random.normal(
            avg_invoice_amount, 
            avg_invoice_amount * 0.3,  # 30% standard deviation
            n_invoices
        ).clip(0)
        
        interest_charged = invoice_amounts * daily_rate * days_overdue
        late_fees = is_late.astype(int) * LATE_FEE
        
        # Aggregate monthly results
        month_result = {
            'period': period,
            'total_invoices': n_invoices,
            'invoices_on_time': (~is_late).sum(),
            'invoices_late': is_late.sum(),
            'pct_late': (is_late.sum() / n_invoices * 100) if n_invoices > 0 else 0,
            'avg_days_overdue': days_overdue[is_late].mean() if is_late.any() else 0,
            'total_invoice_amount': total_amount,
            'total_interest': interest_charged.sum(),
            'total_late_fees': late_fees.sum(),
            'total_revenue': interest_charged.sum() + late_fees.sum()
        }
        
        results.append(month_result)
    
    return pd.DataFrame(results)

# Scenario 1: With discount
print("\nSimulating: WITH DISCOUNT scenario...")
forecast_with_discount = simulate_future_revenue(
    forecast_df, 
    profile, 
    discount_scenario='with_discount'
)

# Scenario 2: No discount
print("Simulating: NO DISCOUNT scenario...")
forecast_no_discount = simulate_future_revenue(
    forecast_df, 
    profile, 
    discount_scenario='no_discount'
)

# ================================================================
# STEP 7: Display forecast results
# ================================================================
print("\n" + "="*80)
print("STEP 7: FORECAST RESULTS - NEXT 12 MONTHS")
print("="*80)

def print_forecast_summary(df, scenario_name):
    """Print summary of forecast results"""
    print(f"\n{scenario_name}")
    print("-" * 80)
    
    total_invoices = df['total_invoices'].sum()
    total_late = df['invoices_late'].sum()
    avg_pct_late = df['pct_late'].mean()
    
    print(f"Total forecasted invoices: {int(total_invoices):,}")
    print(f"Expected late invoices: {int(total_late):,} ({avg_pct_late:.1f}%)")
    print(f"Average days overdue: {df['avg_days_overdue'].mean():.1f}")
    
    print(f"\nTotal Invoice Amounts (12 months):")
    print(f"  Total invoice value: ${df['total_invoice_amount'].sum():,.2f}")
    
    print(f"\nExpected Credit Card Revenue (12 months):")
    print(f"  Interest revenue: ${df['total_interest'].sum():,.2f}")
    print(f"  Late fee revenue: ${df['total_late_fees'].sum():,.2f}")
    print(f"  TOTAL REVENUE: ${df['total_revenue'].sum():,.2f}")
    
    print(f"\nMonthly Average:")
    print(f"  Avg monthly revenue: ${df['total_revenue'].mean():,.2f}")
    print(f"  Avg monthly interest: ${df['total_interest'].mean():,.2f}")
    print(f"  Avg monthly late fees: ${df['total_late_fees'].mean():,.2f}")
    
    return {
        'scenario': scenario_name,
        'total_invoices': int(total_invoices),
        'total_late': int(total_late),
        'pct_late': avg_pct_late,
        'total_revenue': df['total_revenue'].sum(),
        'total_interest': df['total_interest'].sum(),
        'total_late_fees': df['total_late_fees'].sum(),
        'avg_monthly_revenue': df['total_revenue'].mean()
    }

summary_with = print_forecast_summary(forecast_with_discount, "WITH DISCOUNT")
summary_no = print_forecast_summary(forecast_no_discount, "NO DISCOUNT")

# Comparison
print("\n" + "="*80)
print("SCENARIO COMPARISON")
print("="*80)

revenue_diff = summary_no['total_revenue'] - summary_with['total_revenue']

print(f"\nExpected 12-Month Revenue:")
print(f"  No Discount scenario: ${summary_no['total_revenue']:,.2f}")
print(f"  With Discount scenario: ${summary_with['total_revenue']:,.2f}")
print(f"  Difference: ${revenue_diff:+,.2f}")

if revenue_diff > 0:
    pct_increase = (revenue_diff / summary_with['total_revenue'] * 100)
    print(f"\n✓ NO DISCOUNT expected to generate {pct_increase:.1f}% MORE revenue")
else:
    pct_increase = (abs(revenue_diff) / summary_no['total_revenue'] * 100)
    print(f"\n✓ WITH DISCOUNT expected to generate {pct_increase:.1f}% MORE revenue")

# ================================================================
# STEP 8: Save results
# ================================================================
print("\n" + "="*80)
print("STEP 8: SAVING FORECAST RESULTS")
print("="*80)

# Combine forecasts with scenarios
forecast_with_discount['scenario'] = 'With Discount'
forecast_no_discount['scenario'] = 'No Discount'

# Save detailed monthly forecasts
output_excel = os.path.join(OUTPUT_DIR, 'revenue_forecast_12_months.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    forecast_with_discount.to_excel(writer, sheet_name='With_Discount', index=False)
    forecast_no_discount.to_excel(writer, sheet_name='No_Discount', index=False)
    forecast_df.to_excel(writer, sheet_name='Invoice_Forecasts', index=False)
    
    # Summary comparison
    comparison_summary = pd.DataFrame([summary_with, summary_no])
    comparison_summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"✓ Saved: {output_excel}")

# Save CSV versions
forecast_with_discount.to_csv(
    os.path.join(OUTPUT_DIR, 'forecast_with_discount.csv'), index=False
)
forecast_no_discount.to_csv(
    os.path.join(OUTPUT_DIR, 'forecast_no_discount.csv'), index=False
)
print(f"✓ Saved CSV files")

# ================================================================
# STEP 9: Create visualizations
# ================================================================
print("\n" + "="*80)
print("STEP 9: CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Plot 1: Forecasted Invoice Totals
ax1 = axes[0, 0]
ax1.plot(monthly_historical['period'], monthly_historical['total_undiscounted'], 
         marker='o', linewidth=2, label='Historical (Undiscounted)', color='black', alpha=0.7)
ax1.plot(forecast_df['period'], forecast_df['forecast_undiscounted'], 
         marker='s', linewidth=2.5, linestyle='--', label='Forecast (Undiscounted)', 
         color='#4472C4')
ax1.plot(forecast_df['period'], forecast_df['forecast_discounted'], 
         marker='^', linewidth=2.5, linestyle='--', label='Forecast (Discounted)', 
         color='#70AD47')
ax1.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax1.set_title('Invoice Total Forecast (Next 12 Months)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Period', fontsize=12)
ax1.set_ylabel('Invoice Total ($)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: Forecasted Monthly Revenue - Both Scenarios
ax2 = axes[0, 1]
ax2.plot(forecast_with_discount['period'], forecast_with_discount['total_revenue'], 
         marker='o', linewidth=2.5, label='With Discount', color='#70AD47')
ax2.plot(forecast_no_discount['period'], forecast_no_discount['total_revenue'], 
         marker='s', linewidth=2.5, label='No Discount', color='#4472C4')
ax2.set_title('Forecasted Monthly Revenue Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Period', fontsize=12)
ax2.set_ylabel('Monthly Revenue ($)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 3: Cumulative Revenue Forecast
ax3 = axes[1, 0]
cumulative_with = forecast_with_discount['total_revenue'].cumsum()
cumulative_no = forecast_no_discount['total_revenue'].cumsum()

ax3.plot(forecast_with_discount['period'], cumulative_with, 
         marker='o', linewidth=2.5, label='With Discount', color='#70AD47')
ax3.plot(forecast_no_discount['period'], cumulative_no, 
         marker='s', linewidth=2.5, label='No Discount', color='#4472C4')
ax3.set_title('Cumulative Revenue Forecast (12 Months)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Period', fontsize=12)
ax3.set_ylabel('Cumulative Revenue ($)', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 4: Revenue Components Breakdown (With Discount scenario)
ax4 = axes[1, 1]
ax4.bar(forecast_with_discount['period'], forecast_with_discount['total_interest'], 
        label='Interest', color='#70AD47', alpha=0.7)
ax4.bar(forecast_with_discount['period'], forecast_with_discount['total_late_fees'], 
        bottom=forecast_with_discount['total_interest'],
        label='Late Fees', color='#A9D18E', alpha=0.7)
ax4.set_title('Revenue Components - With Discount Scenario', fontsize=14, fontweight='bold')
ax4.set_xlabel('Period', fontsize=12)
ax4.set_ylabel('Revenue ($)', fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Format x-axes
for ax in axes.flat:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = os.path.join(OUTPUT_DIR, 'revenue_forecast_visualization.png')
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_viz}")
plt.show()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("FORECAST COMPLETE!")
print("="*80)

print(f"\nModel used: {best_model_name}")
print(f"Forecast period: {forecast_df['period'].min().strftime('%Y-%m')} to {forecast_df['period'].max().strftime('%Y-%m')}")

print(f"\nExpected 12-Month Results:")
print(f"  Total invoices: {summary_with['total_invoices']:,}")
print(f"  With Discount revenue: ${summary_with['total_revenue']:,.2f}")
print(f"  No Discount revenue: ${summary_no['total_revenue']:,.2f}")
print(f"  Revenue difference: ${revenue_diff:+,.2f}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("Files created:")
print("  1. revenue_forecast_12_months.xlsx - Complete forecast details")
print("  2. forecast_with_discount.csv - Monthly forecast (with discount)")
print("  3. forecast_no_discount.csv - Monthly forecast (no discount)")
print("  4. revenue_forecast_visualization.png - Visualization charts")
print("="*80)