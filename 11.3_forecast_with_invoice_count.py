'''
11.3_Forecast_with_invoice_count - Monthly Invoice Totals Forecast with Invoice Count as Feature

This script forecasts monthly invoice totals for the next 15 months using
regression models. It first forecasts n_invoices, then uses forecasted invoice 
counts as a feature to predict total_discounted_price. This captures the 
relationship: Total Revenue ≈ f(time, invoice_count, seasonality).

Inputs:
-------
- visualisations/9.4_monthly_totals_Period_4_Entire.csv

Outputs:
--------
- visualisations/11.3_forecast_next_15_months.csv
- visualisations/11.3_forecast_with_historical.xlsx
- visualisations/11.3_forecast_visualization.png

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
OUTPUT_PATH = BASE_PATH / "visualisations"

# Create output directory if it doesn't exist
OUTPUT_PATH.mkdir(exist_ok=True)

# ========================
# DISPLAY PATH INFO
# ========================
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
print("🚀 MONTHLY INVOICE TOTALS FORECAST - NEXT 15 MONTHS")
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
print(f"    Avg monthly discounted total: ${monthly_historical['total_discounted_price'].mean():,.2f}")
print(f"    Avg monthly invoice count: {monthly_historical['n_invoices'].mean():.0f}")
print(f"    Avg undiscounted multiplier: {monthly_historical['undiscounted_as_pct'].mean():.2f}%")
print(f"    Avg price per invoice: ${(monthly_historical['total_discounted_price'] / monthly_historical['n_invoices']).mean():,.2f}")

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
print(monthly_historical[['invoice_period', 'total_discounted_price', 'n_invoices', 'undiscounted_as_pct']].head(10).to_string(index=False))

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
# STEP 4: Build regression models for DISCOUNTED PRICE (using invoice count)
# ================================================================
print("\n" + "="*80)
print("🔧 [Step 4/6] BUILDING REGRESSION MODELS FOR DISCOUNTED PRICE")
print("   (Using Invoice Count as Feature)")
print("="*80)

models_discounted = {}
y_discounted = monthly_historical['total_discounted_price'].values

# Model 1: Linear with invoice count
print("  [1/4] Training Linear + Invoice Count...")
X_with_inv = monthly_historical[['month_index', 'n_invoices']].values
model_linear_with_inv = LinearRegression()
model_linear_with_inv.fit(X_with_inv, y_discounted)
pred_linear_with_inv = model_linear_with_inv.predict(X_with_inv)
r2_linear_with_inv = r2_score(y_discounted, pred_linear_with_inv)
mae_linear_with_inv = mean_absolute_error(y_discounted, pred_linear_with_inv)
print(f"    R² Score: {r2_linear_with_inv:.4f}")
print(f"    MAE: ${mae_linear_with_inv:,.2f}")
models_discounted['Linear_InvCount'] = {
    'model': model_linear_with_inv,
    'predictions': pred_linear_with_inv,
    'r2': r2_linear_with_inv,
    'mae': mae_linear_with_inv,
    'type': 'linear_inv'
}

# Model 2: Polynomial with invoice count
print("\n  [2/4] Training Polynomial + Invoice Count...")
poly_features_disc = PolynomialFeatures(degree=2)
X_poly_with_inv = poly_features_disc.fit_transform(monthly_historical[['month_index', 'n_invoices']].values)
model_poly_with_inv = LinearRegression()
model_poly_with_inv.fit(X_poly_with_inv, y_discounted)
pred_poly_with_inv = model_poly_with_inv.predict(X_poly_with_inv)
r2_poly_with_inv = r2_score(y_discounted, pred_poly_with_inv)
mae_poly_with_inv = mean_absolute_error(y_discounted, pred_poly_with_inv)
print(f"    R² Score: {r2_poly_with_inv:.4f}")
print(f"    MAE: ${mae_poly_with_inv:,.2f}")
models_discounted['Poly_InvCount'] = {
    'model': model_poly_with_inv,
    'poly': poly_features_disc,
    'predictions': pred_poly_with_inv,
    'r2': r2_poly_with_inv,
    'mae': mae_poly_with_inv,
    'type': 'poly_inv'
}

# Model 3: Seasonal with invoice count
print("\n  [3/4] Training Seasonal + Invoice Count...")
X_seasonal_with_inv = monthly_historical[['month_index', 'month_sin', 'month_cos', 'n_invoices']].values
model_seasonal_with_inv = LinearRegression()
model_seasonal_with_inv.fit(X_seasonal_with_inv, y_discounted)
pred_seasonal_with_inv = model_seasonal_with_inv.predict(X_seasonal_with_inv)
r2_seasonal_with_inv = r2_score(y_discounted, pred_seasonal_with_inv)
mae_seasonal_with_inv = mean_absolute_error(y_discounted, pred_seasonal_with_inv)
print(f"    R² Score: {r2_seasonal_with_inv:.4f}")
print(f"    MAE: ${mae_seasonal_with_inv:,.2f}")
models_discounted['Seasonal_InvCount'] = {
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
model_poly_seasonal_inv.fit(X_poly_seasonal_inv, y_discounted)
pred_poly_seasonal_inv = model_poly_seasonal_inv.predict(X_poly_seasonal_inv)
r2_poly_seasonal_inv = r2_score(y_discounted, pred_poly_seasonal_inv)
mae_poly_seasonal_inv = mean_absolute_error(y_discounted, pred_poly_seasonal_inv)
print(f"    R² Score: {r2_poly_seasonal_inv:.4f}")
print(f"    MAE: ${mae_poly_seasonal_inv:,.2f}")
models_discounted['PolySeasonal_InvCount'] = {
    'model': model_poly_seasonal_inv,
    'poly': poly_features_ps,
    'predictions': pred_poly_seasonal_inv,
    'r2': r2_poly_seasonal_inv,
    'mae': mae_poly_seasonal_inv,
    'type': 'poly_seasonal_inv'
}

# Select best model for discounted price
best_model_name_discounted = max(models_discounted.items(), key=lambda x: x[1]['r2'])[0]
best_model_info_discounted = models_discounted[best_model_name_discounted]
print(f"\n  ✓ Selected Best Model (Discounted Price): {best_model_name_discounted} (R²={best_model_info_discounted['r2']:.4f})")

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

# === STEP 2: FORECAST DISCOUNTED PRICE (using forecasted invoice count) ===
print("\n  Step 2: Forecasting discounted price using forecasted invoice counts...")
if best_model_info_discounted['type'] == 'linear_inv':
    X_future_disc = np.column_stack([future_month_indices, forecast_invoice_count])
    forecast_discounted = best_model_info_discounted['model'].predict(X_future_disc)
    
elif best_model_info_discounted['type'] == 'poly_inv':
    X_future_disc = np.column_stack([future_month_indices, forecast_invoice_count])
    X_future_poly_disc = best_model_info_discounted['poly'].transform(X_future_disc)
    forecast_discounted = best_model_info_discounted['model'].predict(X_future_poly_disc)
    
elif best_model_info_discounted['type'] == 'seasonal_inv':
    X_future_disc = np.column_stack([future_month_indices, future_month_sin, future_month_cos, forecast_invoice_count])
    forecast_discounted = best_model_info_discounted['model'].predict(X_future_disc)
    
elif best_model_info_discounted['type'] == 'poly_seasonal_inv':
    X_future_base = future_month_indices.reshape(-1, 1)
    X_future_poly = best_model_info_discounted['poly'].transform(X_future_base)
    X_future_disc = np.column_stack([X_future_poly, future_month_sin, future_month_cos, forecast_invoice_count])
    forecast_discounted = best_model_info_discounted['model'].predict(X_future_disc)

# Ensure non-negative forecasts
forecast_discounted = np.maximum(forecast_discounted, 0)

# Calculate average price per invoice
avg_price_per_invoice = forecast_discounted / forecast_invoice_count

# Calculate average undiscounted multiplier from historical data
avg_undiscounted_pct = monthly_historical['undiscounted_as_pct'].mean()
print(f"\n  Historical average undiscounted multiplier: {avg_undiscounted_pct:.2f}%")

# Infer undiscounted amounts using the multiplier
forecast_undiscounted = forecast_discounted * (avg_undiscounted_pct / 100)
forecast_discount = forecast_undiscounted - forecast_discounted

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'invoice_period': future_dates,
    'forecast_discounted_price': forecast_discounted,
    'forecast_undiscounted_price': forecast_undiscounted,
    'forecast_discount_amount': forecast_discount,
    'forecast_invoice_count': forecast_invoice_count,
    'forecast_avg_price_per_invoice': avg_price_per_invoice,
    'forecast_undiscounted_as_pct': avg_undiscounted_pct
})

print("\n  Forecasted Monthly Totals:")
for _, row in forecast_df.iterrows():
    print(f"    {row['invoice_period'].strftime('%Y-%m')}: "
          f"${row['forecast_discounted_price']:>12,.2f} (discounted), "
          f"{row['forecast_invoice_count']:>5,} invoices, "
          f"${row['forecast_avg_price_per_invoice']:>8,.2f}/invoice")

print(f"\n  💰 Total Forecasted (15 months):")
print(f"    Discounted: ${forecast_df['forecast_discounted_price'].sum():,.2f}")
print(f"    Undiscounted: ${forecast_df['forecast_undiscounted_price'].sum():,.2f}")
print(f"    Total discount: ${forecast_df['forecast_discount_amount'].sum():,.2f}")
print(f"    Total invoices: {forecast_df['forecast_invoice_count'].sum():,}")
print(f"    Avg price per invoice: ${forecast_df['forecast_avg_price_per_invoice'].mean():,.2f}")

# ================================================================
# STEP 6: Save results
# ================================================================
print("\n" + "="*80)
print("💾 [Step 6/6] SAVING FORECAST RESULTS")
print("="*80)

# Save forecast CSV
output_csv = OUTPUT_PATH / '11.3_forecast_next_15_months.csv'
forecast_df.to_csv(output_csv, index=False)
print(f"  ✓ Saved: {output_csv.name}")

# Save comprehensive Excel file
output_excel = OUTPUT_PATH / '11.3_forecast_with_historical.xlsx'
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Forecast sheet
    forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
    
    # Historical sheet
    historical_export = monthly_historical[[
        'invoice_period', 'total_discounted_price', 'total_undiscounted_price',
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
    
    # Model info sheet - Discounted Price (with invoice count)
    model_info_discounted = pd.DataFrame({
        'Model': list(models_discounted.keys()),
        'R²': [m['r2'] for m in models_discounted.values()],
        'MAE': [m['mae'] for m in models_discounted.values()],
        'Type': [m['type'] for m in models_discounted.values()],
        'Selected': ['✓' if k == best_model_name_discounted else '' for k in models_discounted.keys()]
    })
    model_info_discounted.to_excel(writer, sheet_name='Models_Discounted', index=False)
    
    # Summary info sheet
    summary_info = pd.DataFrame({
        'Parameter': ['Invoice Count Model', 'Discounted Price Model', 'Training Period',
                      'Forecast Period', 'Avg Undiscounted Multiplier', 'Forecast Periods'],
        'Value': [f"{best_model_name_invoice} (R²={best_model_info_invoice['r2']:.4f})",
                  f"{best_model_name_discounted} (R²={best_model_info_discounted['r2']:.4f})",
                  f"{monthly_historical['invoice_period'].min().strftime('%Y-%m')} to {monthly_historical['invoice_period'].max().strftime('%Y-%m')}",
                  f"{forecast_df['invoice_period'].min().strftime('%Y-%m')} to {forecast_df['invoice_period'].max().strftime('%Y-%m')}",
                  f"{avg_undiscounted_pct:.2f}%",
                  f"{FORECAST_PERIODS}"]
    })
    summary_info.to_excel(writer, sheet_name='Summary', index=False)

print(f"  ✓ Saved: {output_excel.name}")

# ================================================================
# STEP 7: Create visualizations
# ================================================================
print("\n" + "="*80)
print("🎨 [Step 7/7] CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Historical + Forecast Discounted Price
ax1 = axes[0, 0]
ax1.plot(monthly_historical['invoice_period'], monthly_historical['total_discounted_price'],
         marker='o', linewidth=2, label='Historical (Discounted)', color='black', alpha=0.7)
ax1.plot(forecast_df['invoice_period'], forecast_df['forecast_discounted_price'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast (Discounted)',
         color='#4472C4', markersize=8)
ax1.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Forecast Start')
ax1.set_title('Monthly Discounted Price - Historical & Forecast', fontsize=14, fontweight='bold')
ax1.set_xlabel('Period', fontsize=12)
ax1.set_ylabel('Discounted Price ($)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Model Fit Quality - Discounted Price
ax2 = axes[0, 1]
actual = y_discounted
predicted = best_model_info_discounted['predictions']
ax2.scatter(actual, predicted, alpha=0.6, s=100, edgecolors='black', linewidths=1)
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_title(f'Model Fit (Discounted Price): {best_model_name_discounted}\nR² = {best_model_info_discounted["r2"]:.4f}',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Actual Discounted Price ($)', fontsize=12)
ax2.set_ylabel('Predicted Discounted Price ($)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 3: Invoice Count - Historical + Forecast
ax3 = axes[1, 0]
ax3.plot(monthly_historical['invoice_period'], monthly_historical['n_invoices'],
         marker='o', linewidth=2, label='Historical', color='black', alpha=0.7)
ax3.plot(forecast_df['invoice_period'], forecast_df['forecast_invoice_count'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast',
         color='#FFC000', markersize=8)
ax3.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_title(f'Monthly Invoice Count\n(Used as Feature in Discounted Price Model)', 
              fontsize=14, fontweight='bold')
ax3.set_xlabel('Period', fontsize=12)
ax3.set_ylabel('Invoice Count', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Average Price per Invoice
ax4 = axes[1, 1]
historical_avg_price = monthly_historical['total_discounted_price'] / monthly_historical['n_invoices']
ax4.plot(monthly_historical['invoice_period'], historical_avg_price,
         marker='o', linewidth=2, label='Historical', color='#70AD47', alpha=0.7)
ax4.plot(forecast_df['invoice_period'], forecast_df['forecast_avg_price_per_invoice'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast',
         color='#A9D18E', markersize=8)
ax4.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
avg_historical_price = historical_avg_price.mean()
ax4.axhline(y=avg_historical_price, color='#70AD47', linestyle=':', linewidth=2, alpha=0.5,
            label=f'Historical Avg: ${avg_historical_price:,.2f}')
ax4.set_title('Average Price per Invoice', fontsize=14, fontweight='bold')
ax4.set_xlabel('Period', fontsize=12)
ax4.set_ylabel('Avg Price per Invoice ($)', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.3_forecast_visualization.png'
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ FORECAST COMPLETE!")
print("="*80)

print(f"\n📊 Model Details:")
print(f"  Invoice Count Model: {best_model_name_invoice} (R²={best_model_info_invoice['r2']:.4f}, MAE={best_model_info_invoice['mae']:.2f})")
print(f"  Discounted Price Model: {best_model_name_discounted} (R²={best_model_info_discounted['r2']:.4f}, MAE=${best_model_info_discounted['mae']:,.2f})")
print(f"    → Uses forecasted invoice count as a predictor feature")
print(f"  Training period: {monthly_historical['invoice_period'].min().strftime('%Y-%m')} to {monthly_historical['invoice_period'].max().strftime('%Y-%m')}")

print(f"\n💰 Forecast Summary:")
print(f"  Forecast period: {forecast_df['invoice_period'].min().strftime('%Y-%m')} to {forecast_df['invoice_period'].max().strftime('%Y-%m')}")
print(f"  Total discounted: ${forecast_df['forecast_discounted_price'].sum():,.2f}")
print(f"  Total undiscounted: ${forecast_df['forecast_undiscounted_price'].sum():,.2f}")
print(f"  Total invoices: {forecast_df['forecast_invoice_count'].sum():,}")
print(f"  Avg price per invoice: ${forecast_df['forecast_avg_price_per_invoice'].mean():,.2f}")

print(f"\n📁 Output Files:")
print(f"  • 11.3_forecast_next_15_months.csv - Monthly forecast data")
print(f"  • 11.3_forecast_with_historical.xlsx - Complete data with model info")
print(f"  • 11.3_forecast_visualization.png - 4-panel visual analysis")

print(f"\n  All files saved to: {OUTPUT_PATH}")

print("\n" + "="*80)
print("NEXT STEP: Apply payment profiles to forecasted totals to estimate revenue")
print("="*80)

