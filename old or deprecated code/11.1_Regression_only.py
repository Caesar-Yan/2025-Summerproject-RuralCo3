"""
11_Forecast_invoice_totals.py
==============================
Forecast next 12 months of invoice totals using time series regression.

This script:
1. Loads historical invoice data (Dec 2023 onwards)
2. Aggregates to monthly totals
3. Trains regression models (Linear, Polynomial, 2025+ only)
4. Forecasts next 12 months of invoice totals
5. Saves forecast to CSV/Excel for use in revenue estimation

Inputs:
-------
- ats_grouped_transformed_with_discounts.csv (historical ATS invoices)
- invoice_grouped_transformed_with_discounts.csv (historical invoice invoices)

Outputs:
--------
- visualisations/11_invoice_forecast_12_months.csv (monthly forecast data)
- visualisations/11_invoice_forecast_with_historical.xlsx (forecast + historical + model info)
- visualisations/11_invoice_forecast_visualization.png (4-panel visualization)

Author: Chris & Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

# ========================
# PATH CONFIGURATION
# ========================
# IMPORTANT: Update BASE_PATH to match your project location
# Example Windows: r"C:\Users\YourName\Projects\RuralCo3"
# Example Mac/Linux: "/Users/yourname/Projects/RuralCo3"

BASE_PATH = r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3"

# Input and output paths (derived from BASE_PATH)
INPUT_ATS = os.path.join(BASE_PATH, "data_cleaning", "ats_grouped_transformed_with_discounts.csv")
INPUT_INVOICE = os.path.join(BASE_PATH, "data_cleaning", "invoice_grouped_transformed_with_discounts.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "visualisations")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ========================
# DISPLAY PATH INFO
# ========================
print("\n" + "="*80)
print("PATH CONFIGURATION")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"Input Files:")
print(f"  - {os.path.basename(INPUT_ATS)}")
print(f"  - {os.path.basename(INPUT_INVOICE)}")
print(f"Output Folder: {OUTPUT_PATH}")
print("="*80)

# Check if input files exist
if not os.path.exists(INPUT_ATS):
    print(f"\n❌ ERROR: ATS input file not found!")
    print(f"   Expected: {INPUT_ATS}")
    print(f"\n   Please update BASE_PATH in the script to match your project location.")
    exit(1)

if not os.path.exists(INPUT_INVOICE):
    print(f"\n❌ ERROR: Invoice input file not found!")
    print(f"   Expected: {INPUT_INVOICE}")
    print(f"\n   Please update BASE_PATH in the script to match your project location.")
    exit(1)

# ========================
# PLOTTING CONFIGURATION
# ========================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ================================================================
# CONFIGURATION
# ================================================================
FORECAST_MONTHS = 12  # Forecast next 12 months
HISTORICAL_CUTOFF_DATE = pd.Timestamp("2023-12-01")  # Only use data from this date onwards
HISTORICAL_END_DATE = pd.Timestamp("2025-11-30")  # Last date to use (exclude incomplete months)

print("\n" + "="*80)
print("🚀 INVOICE TOTAL FORECAST - NEXT 12 MONTHS")
print("="*80)
print(f"Using historical data from {HISTORICAL_CUTOFF_DATE.strftime('%Y-%m-%d')} to {HISTORICAL_END_DATE.strftime('%Y-%m-%d')}")

# ================================================================
# STEP 1: Load historical invoice data
# ================================================================
print("\n" + "="*80)
print("📁 [Step 1/6] LOADING HISTORICAL INVOICE DATA")
print("="*80)

print(f"  Reading ATS data: {os.path.basename(INPUT_ATS)}")
ats_grouped = pd.read_csv(INPUT_ATS)
ats_grouped['customer_type'] = 'ATS'

print(f"  Reading Invoice data: {os.path.basename(INPUT_INVOICE)}")
invoice_grouped = pd.read_csv(INPUT_INVOICE)
invoice_grouped['customer_type'] = 'Invoice'

combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)

print(f"  ✓ Total historical invoices loaded: {len(combined_df):,}")

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

# Filter out negative prices
initial_count = len(combined_df)
combined_df = combined_df[combined_df['total_undiscounted_price'] >= 0].copy()
filtered_negatives = initial_count - len(combined_df)
if filtered_negatives > 0:
    print(f"  ⚠ Filtered out {filtered_negatives:,} invoices with negative prices")

# Filter to December 2023 onwards
initial_count = len(combined_df)
combined_df = combined_df[combined_df['invoice_period'] >= HISTORICAL_CUTOFF_DATE].copy()
filtered_old = initial_count - len(combined_df)
if filtered_old > 0:
    print(f"  ⚠ Filtered out {filtered_old:,} invoices before {HISTORICAL_CUTOFF_DATE.strftime('%B %Y')}")

# Filter to November 2025 and earlier (exclude incomplete months)
initial_count = len(combined_df)
combined_df = combined_df[combined_df['invoice_period'] <= HISTORICAL_END_DATE].copy()
filtered_recent = initial_count - len(combined_df)
if filtered_recent > 0:
    print(f"  ⚠ Filtered out {filtered_recent:,} invoices after {HISTORICAL_END_DATE.strftime('%B %Y')} (incomplete data)")

print(f"\n  ✓ Valid invoices for analysis: {len(combined_df):,}")
print(f"  Date range: {combined_df['invoice_period'].min().strftime('%Y-%m')} to {combined_df['invoice_period'].max().strftime('%Y-%m')}")

# ================================================================
# STEP 2: Aggregate historical monthly data
# ================================================================
print("\n" + "="*80)
print("📊 [Step 2/6] AGGREGATING HISTORICAL MONTHLY TOTALS")
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

print(f"  ✓ Historical months: {len(monthly_historical)}")
print(f"  First historical month: {monthly_historical['period'].min().strftime('%Y-%m')}")
print(f"  Last historical month: {monthly_historical['period'].max().strftime('%Y-%m')}")

# Display monthly statistics
print("\n  📈 Historical monthly statistics:")
print(f"    Avg monthly undiscounted total: ${monthly_historical['total_undiscounted'].mean():,.2f}")
print(f"    Avg monthly discounted total: ${monthly_historical['total_discounted'].mean():,.2f}")
print(f"    Avg monthly invoice count: {monthly_historical['invoice_count'].mean():.0f}")
print(f"    Avg discount rate: {(monthly_historical['total_discount'].sum() / monthly_historical['total_undiscounted'].sum())*100:.2f}%")

# ================================================================
# DIAGNOSTIC: Check training data
# ================================================================
print("\n" + "="*80)
print("🔍 DIAGNOSTIC: TRAINING DATA INSPECTION")
print("="*80)

print("\n  Monthly historical data:")
for _, row in monthly_historical.iterrows():
    print(f"    {row['period'].strftime('%Y-%m')}: ${row['total_undiscounted']:>12,.2f}  ({row['invoice_count']:>4,.0f} invoices)")

print(f"\n  Data statistics:")
print(f"    Min undiscounted: ${monthly_historical['total_undiscounted'].min():,.2f}")
print(f"    Max undiscounted: ${monthly_historical['total_undiscounted'].max():,.2f}")
print(f"    Mean undiscounted: ${monthly_historical['total_undiscounted'].mean():,.2f}")
print(f"    Trend: {'Increasing' if monthly_historical['total_undiscounted'].iloc[-1] > monthly_historical['total_undiscounted'].iloc[0] else 'Decreasing'}")

# ================================================================
# STEP 3: Build regression models to predict invoice totals
# ================================================================
print("\n" + "="*80)
print("🔧 [Step 3/6] BUILDING REGRESSION MODELS FOR INVOICE TOTALS")
print("="*80)

# Add time features
monthly_historical['month_index'] = range(len(monthly_historical))
monthly_historical['year'] = monthly_historical['period'].dt.year
monthly_historical['month'] = monthly_historical['period'].dt.month

# Option 1: Use 2025+ data only (as suggested by Dave)
monthly_2025 = monthly_historical[monthly_historical['year'] >= 2025].copy()
if len(monthly_2025) > 0:
    monthly_2025['month_index_2025'] = range(len(monthly_2025))

print(f"\n  Data split:")
print(f"    All data: {len(monthly_historical)} months ({monthly_historical['period'].min().strftime('%Y-%m')} to {monthly_historical['period'].max().strftime('%Y-%m')})")
print(f"    2025+ data: {len(monthly_2025)} months")

# Prepare training data
X_all = monthly_historical[['month_index']].values
y_all_undiscounted = monthly_historical['total_undiscounted'].values

# Train models for undiscounted amounts
print("\n  [1/3] Training Linear Regression (All Data)...")
model_linear_all = LinearRegression()
model_linear_all.fit(X_all, y_all_undiscounted)
pred_linear_all = model_linear_all.predict(X_all)
r2_linear_all = r2_score(y_all_undiscounted, pred_linear_all)
mae_linear_all = mean_absolute_error(y_all_undiscounted, pred_linear_all)
print(f"    R² Score: {r2_linear_all:.4f}")
print(f"    MAE: ${mae_linear_all:,.2f}")
print(f"    Slope: ${model_linear_all.coef_[0]:,.2f}/month")
print(f"    Intercept: ${model_linear_all.intercept_:,.2f}")

# Train polynomial model
print("\n  [2/3] Training Polynomial Regression (degree=2)...")
poly_features = PolynomialFeatures(degree=2)
X_poly_all = poly_features.fit_transform(X_all)
model_poly_all = LinearRegression()
model_poly_all.fit(X_poly_all, y_all_undiscounted)
pred_poly_all = model_poly_all.predict(X_poly_all)
r2_poly_all = r2_score(y_all_undiscounted, pred_poly_all)
mae_poly_all = mean_absolute_error(y_all_undiscounted, pred_poly_all)
print(f"    R² Score: {r2_poly_all:.4f}")
print(f"    MAE: ${mae_poly_all:,.2f}")

# Train 2025+ model if enough data
use_2025_model = False
if len(monthly_2025) >= 3:
    print("\n  [3/3] Training Linear Regression (2025+ Only)...")
    X_2025 = monthly_2025[['month_index_2025']].values
    y_2025_undiscounted = monthly_2025['total_undiscounted'].values
    
    model_linear_2025 = LinearRegression()
    model_linear_2025.fit(X_2025, y_2025_undiscounted)
    pred_linear_2025 = model_linear_2025.predict(X_2025)
    r2_linear_2025 = r2_score(y_2025_undiscounted, pred_linear_2025)
    mae_linear_2025 = mean_absolute_error(y_2025_undiscounted, pred_linear_2025)
    print(f"    R² Score: {r2_linear_2025:.4f}")
    print(f"    MAE: ${mae_linear_2025:,.2f}")
    print(f"    Slope: ${model_linear_2025.coef_[0]:,.2f}/month")
    print(f"    Intercept: ${model_linear_2025.intercept_:,.2f}")
    use_2025_model = True
else:
    print("\n  [3/3] ⚠ Not enough 2025 data for separate model")
    r2_linear_2025 = -np.inf  # Ensure it won't be selected

# Select best model based on R² score
model_scores = {
    'Linear (All Data)': (r2_linear_all, model_linear_all, False, 'linear_all'),
    'Polynomial (deg=2)': (r2_poly_all, model_poly_all, False, 'poly'),
    'Linear (2025+)': (r2_linear_2025, model_linear_2025 if use_2025_model else None, True, 'linear_2025')
}

# Find best model
best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k][0])
best_r2, best_model, is_2025_model, model_type = model_scores[best_model_name]

print(f"\n  ✓ Selected Best Model: {best_model_name} (R²={best_r2:.4f})")

# ================================================================
# STEP 4: Forecast future invoice totals
# ================================================================
print("\n" + "="*80)
print("🔮 [Step 4/6] FORECASTING FUTURE INVOICE TOTALS (NEXT 12 MONTHS)")
print("="*80)

last_month_index = monthly_historical['month_index'].max()
last_date = monthly_historical['period'].max()

# Create future dates
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1), 
    periods=FORECAST_MONTHS, 
    freq='MS'
)

# Generate forecasts based on selected model
if is_2025_model:
    # Use 2025 model
    last_2025_index = monthly_2025['month_index_2025'].max()
    future_month_indices = np.arange(last_2025_index + 1, last_2025_index + 1 + FORECAST_MONTHS).reshape(-1, 1)
    forecast_undiscounted = best_model.predict(future_month_indices)
else:
    # Use all-data model
    future_month_indices = np.arange(last_month_index + 1, last_month_index + 1 + FORECAST_MONTHS).reshape(-1, 1)
    
    if model_type == 'poly':
        future_month_indices_poly = poly_features.transform(future_month_indices)
        forecast_undiscounted = best_model.predict(future_month_indices_poly)
    else:
        forecast_undiscounted = best_model.predict(future_month_indices)

# Ensure non-negative forecasts
forecast_undiscounted = np.maximum(forecast_undiscounted, 0)

# Calculate average discount rate from historical data
avg_discount_rate = (monthly_historical['total_discount'].sum() / 
                     monthly_historical['total_undiscounted'].sum())
print(f"\n  Historical average discount rate: {avg_discount_rate*100:.2f}%")

# Calculate discounted amounts
forecast_discounted = forecast_undiscounted * (1 - avg_discount_rate)
forecast_discount = forecast_undiscounted - forecast_discounted

# Estimate invoice counts (use recent 6-month average)
recent_avg_count = monthly_historical.tail(6)['invoice_count'].mean()
forecast_invoice_count = np.full(FORECAST_MONTHS, recent_avg_count)

print(f"  Using recent 6-month average invoice count: {recent_avg_count:.0f} invoices/month")

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'period': future_dates,
    'forecast_undiscounted': forecast_undiscounted,
    'forecast_discounted': forecast_discounted,
    'forecast_discount': forecast_discount,
    'forecast_invoice_count': forecast_invoice_count.astype(int)
})

print("\n  Forecasted Monthly Invoice Totals:")
for _, row in forecast_df.iterrows():
    print(f"    {row['period'].strftime('%Y-%m')}: ${row['forecast_undiscounted']:>12,.2f} (undiscounted), ${row['forecast_discounted']:>12,.2f} (discounted)")

print(f"\n  💰 Total Forecasted (12 months):")
print(f"    Undiscounted: ${forecast_df['forecast_undiscounted'].sum():,.2f}")
print(f"    Discounted: ${forecast_df['forecast_discounted'].sum():,.2f}")
print(f"    Total discount: ${forecast_df['forecast_discount'].sum():,.2f}")
print(f"    Total invoices: {forecast_df['forecast_invoice_count'].sum():,}")

# ================================================================
# STEP 5: Save forecast results
# ================================================================
print("\n" + "="*80)
print("💾 [Step 5/6] SAVING FORECAST RESULTS")
print("="*80)

# Save detailed forecast
output_csv = os.path.join(OUTPUT_PATH, '11_invoice_forecast_12_months.csv')
forecast_df.to_csv(output_csv, index=False)
print(f"  ✓ Saved: {output_csv}")

# Save with historical data for context
output_excel = os.path.join(OUTPUT_PATH, '11_invoice_forecast_with_historical.xlsx')
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
    monthly_historical.to_excel(writer, sheet_name='Historical', index=False)
    
    # Model info sheet
    model_info = pd.DataFrame({
        'Parameter': ['Model Type', 'R² Score', 'Data Period', 'Forecast Period', 
                      'Avg Discount Rate', 'Avg Invoice Count'],
        'Value': [best_model_name, f"{best_r2:.4f}", 
                  f"{monthly_historical['period'].min().strftime('%Y-%m')} to {monthly_historical['period'].max().strftime('%Y-%m')}",
                  f"{forecast_df['period'].min().strftime('%Y-%m')} to {forecast_df['period'].max().strftime('%Y-%m')}",
                  f"{avg_discount_rate*100:.2f}%",
                  f"{recent_avg_count:.0f}"]
    })
    model_info.to_excel(writer, sheet_name='Model_Info', index=False)

print(f"  ✓ Saved: {output_excel}")

# ================================================================
# STEP 6: Create visualizations
# ================================================================
print("\n" + "="*80)
print("🎨 [Step 6/6] CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Historical + Forecasted Invoice Totals
ax1 = axes[0, 0]
ax1.plot(monthly_historical['period'], monthly_historical['total_undiscounted'], 
         marker='o', linewidth=2, label='Historical (Undiscounted)', color='black', alpha=0.7)
ax1.plot(forecast_df['period'], forecast_df['forecast_undiscounted'], 
         marker='s', linewidth=2.5, linestyle='--', label='Forecast (Undiscounted)', 
         color='#4472C4', markersize=8)
ax1.plot(forecast_df['period'], forecast_df['forecast_discounted'], 
         marker='^', linewidth=2.5, linestyle='--', label='Forecast (Discounted)', 
         color='#70AD47', markersize=8)
ax1.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Forecast Start')
ax1.set_title('Monthly Invoice Totals - Historical & Forecast', fontsize=14, fontweight='bold')
ax1.set_xlabel('Period', fontsize=12)
ax1.set_ylabel('Invoice Total ($)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Model Fit Quality
ax2 = axes[0, 1]
if is_2025_model:
    actual = y_2025_undiscounted
    predicted = pred_linear_2025
else:
    actual = y_all_undiscounted
    predicted = pred_poly_all if model_type == 'poly' else pred_linear_all

ax2.scatter(actual, predicted, alpha=0.6, s=100, edgecolors='black', linewidths=1)
# Add perfect prediction line
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_title(f'Model Fit: {best_model_name}\nR² = {best_r2:.4f}', fontsize=14, fontweight='bold')
ax2.set_xlabel('Actual Invoice Total ($)', fontsize=12)
ax2.set_ylabel('Predicted Invoice Total ($)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 3: Invoice Count Forecast
ax3 = axes[1, 0]
ax3.plot(monthly_historical['period'], monthly_historical['invoice_count'], 
         marker='o', linewidth=2, label='Historical', color='black', alpha=0.7)
ax3.plot(forecast_df['period'], forecast_df['forecast_invoice_count'], 
         marker='s', linewidth=2.5, linestyle='--', label='Forecast (Avg)', 
         color='#FFC000', markersize=8)
ax3.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.axhline(y=recent_avg_count, color='#FFC000', linestyle=':', linewidth=2, alpha=0.5, 
            label=f'6-Month Avg: {recent_avg_count:.0f}')
ax3.set_title('Monthly Invoice Count', fontsize=14, fontweight='bold')
ax3.set_xlabel('Period', fontsize=12)
ax3.set_ylabel('Invoice Count', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Discount Amount Trends
ax4 = axes[1, 1]
ax4.plot(monthly_historical['period'], monthly_historical['total_discount'], 
         marker='o', linewidth=2, label='Historical Discount', color='#70AD47', alpha=0.7)
ax4.plot(forecast_df['period'], forecast_df['forecast_discount'], 
         marker='s', linewidth=2.5, linestyle='--', label='Forecast Discount', 
         color='#A9D18E', markersize=8)
ax4.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax4.set_title('Monthly Discount Amount', fontsize=14, fontweight='bold')
ax4.set_xlabel('Period', fontsize=12)
ax4.set_ylabel('Total Discount ($)', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = os.path.join(OUTPUT_PATH, '11_invoice_forecast_visualization.png')
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ INVOICE FORECAST COMPLETE!")
print("="*80)

print(f"\n📊 Model Details:")
print(f"  Model type: {best_model_name}")
print(f"  R² Score: {best_r2:.4f}")
print(f"  Training period: {monthly_historical['period'].min().strftime('%Y-%m')} to {monthly_historical['period'].max().strftime('%Y-%m')}")

print(f"\n💰 Forecast Summary:")
print(f"  Forecast period: {forecast_df['period'].min().strftime('%Y-%m')} to {forecast_df['period'].max().strftime('%Y-%m')}")
print(f"  Total undiscounted: ${forecast_df['forecast_undiscounted'].sum():,.2f}")
print(f"  Total discounted: ${forecast_df['forecast_discounted'].sum():,.2f}")
print(f"  Total invoices: {forecast_df['forecast_invoice_count'].sum():,}")

print(f"\n📁 Output Files:")
print(f"  • 11_invoice_forecast_12_months.csv - Monthly forecast (for use in revenue estimation)")
print(f"  • 11_invoice_forecast_with_historical.xlsx - Complete data with historical context")
print(f"  • 11_invoice_forecast_visualization.png - Visual analysis")

print(f"\n  All files saved to: {OUTPUT_PATH}")

print("\n" + "="*80)
print("NEXT STEP: Run 12_Estimate_revenue_from_forecast.py to calculate credit card revenue")
print("="*80)