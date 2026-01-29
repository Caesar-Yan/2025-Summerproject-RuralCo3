'''
11.5_Forecast_undiscounted_amount_directly - Monthly Undiscounted Price Forecast

This script forecasts monthly undiscounted totals for the next 15 months using
regression models. It first forecasts n_invoices, then uses forecasted invoice 
counts as a feature to predict total_undiscounted_price directly.

This is a companion to 11.3, which forecasts discounted prices. Together, these
two scripts provide independent forecasts of both discounted and undiscounted
amounts, which can be compared to the discount multiplier approach in 11.4.

Inputs:
-------
- visualisations/9.4_monthly_totals_Period_4_Entire.csv

Outputs:
--------
- forecast/11.5_forecast_undiscounted_next_15_months.csv
- forecast/11.5_forecast_undiscounted_with_historical.xlsx
- forecast/11.5_forecast_undiscounted_visualization.png

Author: Chris & Team
Date: January 2026
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========================
# PATH CONFIGURATION
# ========================
BASE_PATH = Path(r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3")

# Input and output paths
INPUT_FILE = BASE_PATH / "visualisations" / "9.4_monthly_totals_Period_4_Entire.csv"
OUTPUT_PATH = BASE_PATH / "forecast"

# Create output directory if it doesn't exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ========================
# DISPLAY PATH INFO
# ========================
print("\n" + "="*80)
print("PATH CONFIGURATION")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"  (Full resolved path: {BASE_PATH.resolve()})")
print(f"Output Folder: {OUTPUT_PATH}")
print(f"  (Full resolved path: {OUTPUT_PATH.resolve()})")
print("\n" + "="*80)
print("PATH CONFIGURATION")
print("="*80)
print(f"Base Directory: {BASE_PATH}")
print(f"Input File: {INPUT_FILE.name}")
print(f"Output Folder: {OUTPUT_PATH}")
print("="*80)

# Check if input file exists
if not INPUT_FILE.exists():
    print(f"\n❌ ERROR: Input file not found!")
    print(f"   Expected: {INPUT_FILE}")
    print(f"\n   Please ensure 9.4_monthly_totals_Period_4_Entire.csv exists.")
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
FORECAST_PERIODS = 15  # Forecast next 15 months

print("\n" + "="*80)
print("🚀 MONTHLY UNDISCOUNTED PRICE FORECAST - NEXT 15 MONTHS")
print("   (with Invoice Count as Predictor Feature)")
print("="*80)

# ================================================================
# STEP 1: Load historical data
# ================================================================
print("\n" + "="*80)
print("📁 [Step 1/6] LOADING HISTORICAL DATA")
print("="*80)

# Load the monthly totals
monthly_historical = pd.read_csv(INPUT_FILE)

print(f"  ✓ Loaded {len(monthly_historical)} months of historical data")

# Parse dates
monthly_historical['invoice_period'] = pd.to_datetime(monthly_historical['invoice_period'])
monthly_historical = monthly_historical.sort_values('invoice_period').reset_index(drop=True)

print(f"  Date range: {monthly_historical['invoice_period'].min().strftime('%Y-%m')} to {monthly_historical['invoice_period'].max().strftime('%Y-%m')}")

# Display summary statistics
print(f"\n  📈 Historical Statistics:")
print(f"    Avg monthly undiscounted total: ${monthly_historical['total_undiscounted_price'].mean():,.2f}")
print(f"    Avg monthly discounted total: ${monthly_historical['total_discounted_price'].mean():,.2f}")
print(f"    Avg monthly invoice count: {monthly_historical['n_invoices'].mean():.0f}")
print(f"    Avg discount amount: ${monthly_historical['discount_amount'].mean():,.2f}")
print(f"    Avg price per invoice (undiscounted): ${(monthly_historical['total_undiscounted_price'] / monthly_historical['n_invoices']).mean():,.2f}")

# ================================================================
# STEP 2: Prepare data for modeling
# ================================================================
print("\n" + "="*80)
print("📊 [Step 2/6] PREPARING DATA FOR MODELING")
print("="*80)

# Add time index
monthly_historical['month_index'] = range(len(monthly_historical))

# Add cyclical features for seasonality
monthly_historical['month'] = monthly_historical['invoice_period'].dt.month
monthly_historical['month_sin'] = np.sin(2 * np.pi * monthly_historical['month'] / 12)
monthly_historical['month_cos'] = np.cos(2 * np.pi * monthly_historical['month'] / 12)

print(f"  ✓ Added time features and cyclical components")

# Display the data
print(f"\n  Historical monthly data preview:")
print(monthly_historical[['invoice_period', 'total_undiscounted_price', 'total_discounted_price', 
                          'n_invoices', 'discount_amount']].head(10).to_string(index=False))

# ================================================================
# STEP 3: Build regression models for INVOICE COUNT (first)
# ================================================================
print("\n" + "="*80)
print("🔧 [Step 3/6] BUILDING REGRESSION MODELS FOR INVOICE COUNT")
print("="*80)

models_invoice_count = {}
X = monthly_historical[['month_index']].values
y_invoice_count = monthly_historical['n_invoices'].values

# Model 1: Simple Linear Regression for invoice count
print("  [1/3] Training Linear Regression...")
model_linear_inv = LinearRegression()
model_linear_inv.fit(X, y_invoice_count)
pred_linear_inv = model_linear_inv.predict(X)
r2_linear_inv = r2_score(y_invoice_count, pred_linear_inv)
mae_linear_inv = mean_absolute_error(y_invoice_count, pred_linear_inv)
print(f"    R² Score: {r2_linear_inv:.4f}")
print(f"    MAE: {mae_linear_inv:.2f} invoices")
models_invoice_count['Linear'] = {
    'model': model_linear_inv,
    'predictions': pred_linear_inv,
    'r2': r2_linear_inv,
    'mae': mae_linear_inv,
    'type': 'linear'
}

# Model 2: Polynomial Regression for invoice count
print("\n  [2/3] Training Polynomial Regression (degree=2)...")
poly_features_inv = PolynomialFeatures(degree=2)
X_poly_inv = poly_features_inv.fit_transform(X)
model_poly_inv = LinearRegression()
model_poly_inv.fit(X_poly_inv, y_invoice_count)
pred_poly_inv = model_poly_inv.predict(X_poly_inv)
r2_poly_inv = r2_score(y_invoice_count, pred_poly_inv)
mae_poly_inv = mean_absolute_error(y_invoice_count, pred_poly_inv)
print(f"    R² Score: {r2_poly_inv:.4f}")
print(f"    MAE: {mae_poly_inv:.2f} invoices")
models_invoice_count['Polynomial'] = {
    'model': model_poly_inv,
    'poly': poly_features_inv,
    'predictions': pred_poly_inv,
    'r2': r2_poly_inv,
    'mae': mae_poly_inv,
    'type': 'polynomial'
}

# Model 3: Seasonal Model for invoice count
print("\n  [3/3] Training Seasonal Model...")
X_seasonal = monthly_historical[['month_index', 'month_sin', 'month_cos']].values
model_seasonal_inv = LinearRegression()
model_seasonal_inv.fit(X_seasonal, y_invoice_count)
pred_seasonal_inv = model_seasonal_inv.predict(X_seasonal)
r2_seasonal_inv = r2_score(y_invoice_count, pred_seasonal_inv)
mae_seasonal_inv = mean_absolute_error(y_invoice_count, pred_seasonal_inv)
print(f"    R² Score: {r2_seasonal_inv:.4f}")
print(f"    MAE: {mae_seasonal_inv:.2f} invoices")
models_invoice_count['Seasonal'] = {
    'model': model_seasonal_inv,
    'predictions': pred_seasonal_inv,
    'r2': r2_seasonal_inv,
    'mae': mae_seasonal_inv,
    'type': 'seasonal'
}

# Select best model for invoice count
best_model_name_invoice = max(models_invoice_count.items(), key=lambda x: x[1]['r2'])[0]
best_model_info_invoice = models_invoice_count[best_model_name_invoice]
print(f"\n  ✓ Selected Best Model (Invoice Count): {best_model_name_invoice} (R²={best_model_info_invoice['r2']:.4f})")

# ================================================================
# STEP 4: Build regression models for UNDISCOUNTED PRICE (using invoice count)
# ================================================================
print("\n" + "="*80)
print("🔧 [Step 4/6] BUILDING REGRESSION MODELS FOR UNDISCOUNTED PRICE")
print("   (Using Invoice Count as Feature)")
print("="*80)

models_undiscounted = {}
y_undiscounted = monthly_historical['total_undiscounted_price'].values

# Model 1: Linear with invoice count
print("  [1/4] Training Linear + Invoice Count...")
X_with_inv = monthly_historical[['month_index', 'n_invoices']].values
model_linear_with_inv = LinearRegression()
model_linear_with_inv.fit(X_with_inv, y_undiscounted)
pred_linear_with_inv = model_linear_with_inv.predict(X_with_inv)
r2_linear_with_inv = r2_score(y_undiscounted, pred_linear_with_inv)
mae_linear_with_inv = mean_absolute_error(y_undiscounted, pred_linear_with_inv)
print(f"    R² Score: {r2_linear_with_inv:.4f}")
print(f"    MAE: ${mae_linear_with_inv:,.2f}")
models_undiscounted['Linear_InvCount'] = {
    'model': model_linear_with_inv,
    'predictions': pred_linear_with_inv,
    'r2': r2_linear_with_inv,
    'mae': mae_linear_with_inv,
    'type': 'linear_inv'
}

# Model 2: Polynomial with invoice count
print("\n  [2/4] Training Polynomial + Invoice Count...")
poly_features_undisc = PolynomialFeatures(degree=2)
X_poly_with_inv = poly_features_undisc.fit_transform(monthly_historical[['month_index', 'n_invoices']].values)
model_poly_with_inv = LinearRegression()
model_poly_with_inv.fit(X_poly_with_inv, y_undiscounted)
pred_poly_with_inv = model_poly_with_inv.predict(X_poly_with_inv)
r2_poly_with_inv = r2_score(y_undiscounted, pred_poly_with_inv)
mae_poly_with_inv = mean_absolute_error(y_undiscounted, pred_poly_with_inv)
print(f"    R² Score: {r2_poly_with_inv:.4f}")
print(f"    MAE: ${mae_poly_with_inv:,.2f}")
models_undiscounted['Poly_InvCount'] = {
    'model': model_poly_with_inv,
    'poly': poly_features_undisc,
    'predictions': pred_poly_with_inv,
    'r2': r2_poly_with_inv,
    'mae': mae_poly_with_inv,
    'type': 'poly_inv'
}

# Model 3: Seasonal with invoice count
print("\n  [3/4] Training Seasonal + Invoice Count...")
X_seasonal_with_inv = monthly_historical[['month_index', 'month_sin', 'month_cos', 'n_invoices']].values
model_seasonal_with_inv = LinearRegression()
model_seasonal_with_inv.fit(X_seasonal_with_inv, y_undiscounted)
pred_seasonal_with_inv = model_seasonal_with_inv.predict(X_seasonal_with_inv)
r2_seasonal_with_inv = r2_score(y_undiscounted, pred_seasonal_with_inv)
mae_seasonal_with_inv = mean_absolute_error(y_undiscounted, pred_seasonal_with_inv)
print(f"    R² Score: {r2_seasonal_with_inv:.4f}")
print(f"    MAE: ${mae_seasonal_with_inv:,.2f}")
models_undiscounted['Seasonal_InvCount'] = {
    'model': model_seasonal_with_inv,
    'predictions': pred_seasonal_with_inv,
    'r2': r2_seasonal_with_inv,
    'mae': mae_seasonal_with_inv,
    'type': 'seasonal_inv'
}

# Model 4: Polynomial + Seasonal + Invoice Count
print("\n  [4/4] Training Poly + Seasonal + Invoice Count...")
X_base_ps = monthly_historical[['month_index']].values
poly_features_ps = PolynomialFeatures(degree=2)
X_poly_ps = poly_features_ps.fit_transform(X_base_ps)
X_poly_seasonal_inv = np.column_stack([
    X_poly_ps,
    monthly_historical['month_sin'].values,
    monthly_historical['month_cos'].values,
    monthly_historical['n_invoices'].values
])
model_poly_seasonal_inv = LinearRegression()
model_poly_seasonal_inv.fit(X_poly_seasonal_inv, y_undiscounted)
pred_poly_seasonal_inv = model_poly_seasonal_inv.predict(X_poly_seasonal_inv)
r2_poly_seasonal_inv = r2_score(y_undiscounted, pred_poly_seasonal_inv)
mae_poly_seasonal_inv = mean_absolute_error(y_undiscounted, pred_poly_seasonal_inv)
print(f"    R² Score: {r2_poly_seasonal_inv:.4f}")
print(f"    MAE: ${mae_poly_seasonal_inv:,.2f}")
models_undiscounted['PolySeasonal_InvCount'] = {
    'model': model_poly_seasonal_inv,
    'poly': poly_features_ps,
    'predictions': pred_poly_seasonal_inv,
    'r2': r2_poly_seasonal_inv,
    'mae': mae_poly_seasonal_inv,
    'type': 'poly_seasonal_inv'
}

# Select best model for undiscounted price
best_model_name_undiscounted = max(models_undiscounted.items(), key=lambda x: x[1]['r2'])[0]
best_model_info_undiscounted = models_undiscounted[best_model_name_undiscounted]
print(f"\n  ✓ Selected Best Model (Undiscounted Price): {best_model_name_undiscounted} (R²={best_model_info_undiscounted['r2']:.4f})")

# ================================================================
# STEP 5: Generate forecast
# ================================================================
print("\n" + "="*80)
print("🔮 [Step 5/6] GENERATING FORECAST (NEXT 15 MONTHS)")
print("="*80)

last_month_index = monthly_historical['month_index'].max()
last_date = monthly_historical['invoice_period'].max()

# Create future dates
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=FORECAST_PERIODS,
    freq='MS'
)

# Generate future indices and features
future_month_indices = np.arange(last_month_index + 1, last_month_index + 1 + FORECAST_PERIODS)
future_months = [d.month for d in future_dates]
future_month_sin = [np.sin(2 * np.pi * m / 12) for m in future_months]
future_month_cos = [np.cos(2 * np.pi * m / 12) for m in future_months]

# === STEP 1: FORECAST INVOICE COUNT ===
print("\n  Step 1: Forecasting invoice count...")
if best_model_info_invoice['type'] == 'linear':
    X_future_inv = future_month_indices.reshape(-1, 1)
    forecast_invoice_count = best_model_info_invoice['model'].predict(X_future_inv)
    
elif best_model_info_invoice['type'] == 'polynomial':
    X_future_inv = future_month_indices.reshape(-1, 1)
    X_future_poly_inv = best_model_info_invoice['poly'].transform(X_future_inv)
    forecast_invoice_count = best_model_info_invoice['model'].predict(X_future_poly_inv)
    
elif best_model_info_invoice['type'] == 'seasonal':
    X_future_inv = np.column_stack([future_month_indices, future_month_sin, future_month_cos])
    forecast_invoice_count = best_model_info_invoice['model'].predict(X_future_inv)

# Ensure non-negative and round to integers
forecast_invoice_count = np.maximum(forecast_invoice_count, 0).round().astype(int)

print(f"  ✓ Forecasted invoice counts: min={forecast_invoice_count.min()}, max={forecast_invoice_count.max()}, avg={forecast_invoice_count.mean():.0f}")

# === STEP 2: FORECAST UNDISCOUNTED PRICE (using forecasted invoice count) ===
print("\n  Step 2: Forecasting undiscounted price using forecasted invoice counts...")
if best_model_info_undiscounted['type'] == 'linear_inv':
    X_future_undisc = np.column_stack([future_month_indices, forecast_invoice_count])
    forecast_undiscounted = best_model_info_undiscounted['model'].predict(X_future_undisc)
    
elif best_model_info_undiscounted['type'] == 'poly_inv':
    X_future_undisc = np.column_stack([future_month_indices, forecast_invoice_count])
    X_future_poly_undisc = best_model_info_undiscounted['poly'].transform(X_future_undisc)
    forecast_undiscounted = best_model_info_undiscounted['model'].predict(X_future_poly_undisc)
    
elif best_model_info_undiscounted['type'] == 'seasonal_inv':
    X_future_undisc = np.column_stack([future_month_indices, future_month_sin, future_month_cos, forecast_invoice_count])
    forecast_undiscounted = best_model_info_undiscounted['model'].predict(X_future_undisc)
    
elif best_model_info_undiscounted['type'] == 'poly_seasonal_inv':
    X_future_base = future_month_indices.reshape(-1, 1)
    X_future_poly = best_model_info_undiscounted['poly'].transform(X_future_base)
    X_future_undisc = np.column_stack([X_future_poly, future_month_sin, future_month_cos, forecast_invoice_count])
    forecast_undiscounted = best_model_info_undiscounted['model'].predict(X_future_undisc)

# Ensure non-negative forecasts
forecast_undiscounted = np.maximum(forecast_undiscounted, 0)

# Calculate average price per invoice (undiscounted)
avg_price_per_invoice_undiscounted = forecast_undiscounted / forecast_invoice_count

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'invoice_period': future_dates,
    'forecast_undiscounted_price': forecast_undiscounted,
    'forecast_invoice_count': forecast_invoice_count,
    'forecast_avg_price_per_invoice_undiscounted': avg_price_per_invoice_undiscounted
})

print("\n  Forecasted Monthly Totals:")
for _, row in forecast_df.iterrows():
    print(f"    {row['invoice_period'].strftime('%Y-%m')}: "
          f"${row['forecast_undiscounted_price']:>12,.2f} (undiscounted), "
          f"{row['forecast_invoice_count']:>5,} invoices, "
          f"${row['forecast_avg_price_per_invoice_undiscounted']:>8,.2f}/invoice")

print(f"\n  💰 Total Forecasted (15 months):")
print(f"    Undiscounted: ${forecast_df['forecast_undiscounted_price'].sum():,.2f}")
print(f"    Total invoices: {forecast_df['forecast_invoice_count'].sum():,}")
print(f"    Avg price per invoice: ${forecast_df['forecast_avg_price_per_invoice_undiscounted'].mean():,.2f}")

# ================================================================
# STEP 6: Save results
# ================================================================
print("\n" + "="*80)
print("💾 [Step 6/6] SAVING FORECAST RESULTS")
print("="*80)

# Save forecast CSV
output_csv = OUTPUT_PATH / '11.5_forecast_undiscounted_next_15_months.csv'
forecast_df.to_csv(output_csv, index=False)
print(f"  ✓ Saved: {output_csv.name}")

# Save comprehensive Excel file
output_excel = OUTPUT_PATH / '11.5_forecast_undiscounted_with_historical.xlsx'
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Forecast sheet
    forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
    
    # Historical sheet
    historical_export = monthly_historical[[
        'invoice_period', 'total_undiscounted_price', 'total_discounted_price',
        'discount_amount', 'n_invoices', 'undiscounted_as_pct'
    ]].copy()
    historical_export.to_excel(writer, sheet_name='Historical', index=False)
    
    # Model info sheet - Invoice Count
    model_info_invoice = pd.DataFrame({
        'Model': list(models_invoice_count.keys()),
        'R²': [m['r2'] for m in models_invoice_count.values()],
        'MAE': [m['mae'] for m in models_invoice_count.values()],
        'Type': [m['type'] for m in models_invoice_count.values()],
        'Selected': ['✓' if k == best_model_name_invoice else '' for k in models_invoice_count.keys()]
    })
    model_info_invoice.to_excel(writer, sheet_name='Models_InvoiceCount', index=False)
    
    # Model info sheet - Undiscounted Price (with invoice count)
    model_info_undiscounted = pd.DataFrame({
        'Model': list(models_undiscounted.keys()),
        'R²': [m['r2'] for m in models_undiscounted.values()],
        'MAE': [m['mae'] for m in models_undiscounted.values()],
        'Type': [m['type'] for m in models_undiscounted.values()],
        'Selected': ['✓' if k == best_model_name_undiscounted else '' for k in models_undiscounted.keys()]
    })
    model_info_undiscounted.to_excel(writer, sheet_name='Models_Undiscounted', index=False)
    
    # Summary info sheet
    summary_info = pd.DataFrame({
        'Parameter': ['Invoice Count Model', 'Undiscounted Price Model', 'Training Period',
                      'Forecast Period', 'Forecast Periods'],
        'Value': [f"{best_model_name_invoice} (R²={best_model_info_invoice['r2']:.4f})",
                  f"{best_model_name_undiscounted} (R²={best_model_info_undiscounted['r2']:.4f})",
                  f"{monthly_historical['invoice_period'].min().strftime('%Y-%m')} to {monthly_historical['invoice_period'].max().strftime('%Y-%m')}",
                  f"{forecast_df['invoice_period'].min().strftime('%Y-%m')} to {forecast_df['invoice_period'].max().strftime('%Y-%m')}",
                  f"{FORECAST_PERIODS}"]
    })
    summary_info.to_excel(writer, sheet_name='Summary', index=False)

print(f"  ✓ Saved: {output_excel.name}")

# ================================================================
# VISUALIZATIONS
# ================================================================
print("\n" + "="*80)
print("🎨 CREATING VISUALIZATIONS")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(monthly_historical['invoice_period'], monthly_historical['total_undiscounted_price'],
         marker='o', linewidth=2, label='Historical (Undiscounted)', color='black', alpha=0.7)
ax.plot(forecast_df['invoice_period'], forecast_df['forecast_undiscounted_price'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast (Undiscounted)',
         color='#ED7D31', markersize=8)
ax.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Forecast Start')
ax.set_title('Monthly Undiscounted Price - Historical & Forecast', fontsize=14, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Undiscounted Price ($)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.5_undiscounted_price_forecast.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz.name}")
plt.close()

# Plot 2: Model Fit Quality - Undiscounted Price
fig, ax = plt.subplots(figsize=(10, 8))
actual = y_undiscounted
predicted = best_model_info_undiscounted['predictions']
ax.scatter(actual, predicted, alpha=0.6, s=100, edgecolors='black', linewidths=1)
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Model Fit (Undiscounted Price): {best_model_name_undiscounted}\nR² = {best_model_info_undiscounted["r2"]:.4f}',
              fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Undiscounted Price ($)', fontsize=12)
ax.set_ylabel('Predicted Undiscounted Price ($)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.5_model_fit_undiscounted_price.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz.name}")
plt.close()

# Plot 3: Invoice Count - Historical + Forecast
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(monthly_historical['invoice_period'], monthly_historical['n_invoices'],
         marker='o', linewidth=2, label='Historical', color='black', alpha=0.7)
ax.plot(forecast_df['invoice_period'], forecast_df['forecast_invoice_count'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast',
         color='#FFC000', markersize=8)
ax.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.set_title(f'Monthly Invoice Count\n(Used as Feature in Undiscounted Price Model)', 
              fontsize=14, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Invoice Count', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.5_invoice_count_forecast.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz.name}")
plt.close()

# Plot 4: Average Price per Invoice (Undiscounted)
fig, ax = plt.subplots(figsize=(14, 7))
historical_avg_price_undiscounted = monthly_historical['total_undiscounted_price'] / monthly_historical['n_invoices']
ax.plot(monthly_historical['invoice_period'], historical_avg_price_undiscounted,
         marker='o', linewidth=2, label='Historical', color='#ED7D31', alpha=0.7)
ax.plot(forecast_df['invoice_period'], forecast_df['forecast_avg_price_per_invoice_undiscounted'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast',
         color='#F4B084', markersize=8)
ax.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
avg_historical_price_undiscounted = historical_avg_price_undiscounted.mean()
ax.axhline(y=avg_historical_price_undiscounted, color='#ED7D31', linestyle=':', linewidth=2, alpha=0.5,
            label=f'Historical Avg: ${avg_historical_price_undiscounted:,.2f}')
ax.set_title('Average Undiscounted Price per Invoice', fontsize=14, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Avg Price per Invoice ($)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.5_average_undiscounted_price_per_invoice.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ UNDISCOUNTED PRICE FORECAST COMPLETE!")
print("="*80)

print(f"\n📊 Model Details:")
print(f"  Invoice Count Model: {best_model_name_invoice} (R²={best_model_info_invoice['r2']:.4f}, MAE={best_model_info_invoice['mae']:.2f})")
print(f"  Undiscounted Price Model: {best_model_name_undiscounted} (R²={best_model_info_undiscounted['r2']:.4f}, MAE=${best_model_info_undiscounted['mae']:,.2f})")
print(f"    → Uses forecasted invoice count as a predictor feature")
print(f"  Training period: {monthly_historical['invoice_period'].min().strftime('%Y-%m')} to {monthly_historical['invoice_period'].max().strftime('%Y-%m')}")

print(f"\n💰 Forecast Summary:")
print(f"  Forecast period: {forecast_df['invoice_period'].min().strftime('%Y-%m')} to {forecast_df['invoice_period'].max().strftime('%Y-%m')}")
print(f"  Total undiscounted: ${forecast_df['forecast_undiscounted_price'].sum():,.2f}")
print(f"  Total invoices: {forecast_df['forecast_invoice_count'].sum():,}")
print(f"  Avg price per invoice: ${forecast_df['forecast_avg_price_per_invoice_undiscounted'].mean():,.2f}")

print(f"\n📁 Output Files:")
print(f"  • 11.5_forecast_undiscounted_next_15_months.csv - Monthly forecast data")
print(f"  • 11.5_forecast_undiscounted_with_historical.xlsx - Complete data with model info")
print(f"  • 11.5_forecast_undiscounted_visualization.png - 4-panel visual analysis")

print(f"\n  All files saved to: {OUTPUT_PATH}")

print("\n" + "="*80)
print("COMPARISON WITH OTHER FORECASTING APPROACHES:")
print("  • 11.3: Forecasts discounted prices, infers undiscounted using constant multiplier")
print("  • 11.4: Forecasts discount rate/multiplier using invoice count")
print("  • 11.5: Forecasts undiscounted prices directly (this script)")
print("\nCombine these approaches to validate and improve forecast accuracy!")
print("="*80)