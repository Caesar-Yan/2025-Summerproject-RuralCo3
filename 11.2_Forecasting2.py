'''
11.2_Forecasting2 - Monthly Invoice Totals Forecast

This script forecasts monthly invoice totals for the next 15 months using
regression models. It predicts total_discounted_price and infers 
total_undiscounted_price using the historical undiscounted_as_pct multiplier.

Inputs:
-------
- visualisations/9.4_monthly_totals_Period_4_Entire.csv

Outputs:
--------
- visualisations/11.2_forecast_next_15_months.csv
- visualisations/11.2_forecast_with_historical.xlsx
- visualisations/11.2_forecast_visualization.png

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
# STEP 3: Build regression models
# ================================================================
print("\n" + "="*80)
print("🔧 [Step 3/6] BUILDING REGRESSION MODELS")
print("="*80)

models = {}
X = monthly_historical[['month_index']].values
y_discounted = monthly_historical['total_discounted_price'].values

# Model 1: Simple Linear Regression
print("  [1/4] Training Linear Regression...")
model_linear = LinearRegression()
model_linear.fit(X, y_discounted)
pred_linear = model_linear.predict(X)
r2_linear = r2_score(y_discounted, pred_linear)
mae_linear = mean_absolute_error(y_discounted, pred_linear)
print(f"    R² Score: {r2_linear:.4f}")
print(f"    MAE: ${mae_linear:,.2f}")
models['Linear'] = {
    'model': model_linear,
    'predictions': pred_linear,
    'r2': r2_linear,
    'type': 'linear'
}

# Model 2: Polynomial Regression (degree 2)
print("\n  [2/4] Training Polynomial Regression (degree=2)...")
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y_discounted)
pred_poly = model_poly.predict(X_poly)
r2_poly = r2_score(y_discounted, pred_poly)
mae_poly = mean_absolute_error(y_discounted, pred_poly)
print(f"    R² Score: {r2_poly:.4f}")
print(f"    MAE: ${mae_poly:,.2f}")
models['Polynomial'] = {
    'model': model_poly,
    'poly': poly_features,
    'predictions': pred_poly,
    'r2': r2_poly,
    'type': 'polynomial'
}

# Model 3: Seasonal Model
print("\n  [3/4] Training Seasonal Model...")
X_seasonal = monthly_historical[['month_index', 'month_sin', 'month_cos']].values
model_seasonal = LinearRegression()
model_seasonal.fit(X_seasonal, y_discounted)
pred_seasonal = model_seasonal.predict(X_seasonal)
r2_seasonal = r2_score(y_discounted, pred_seasonal)
mae_seasonal = mean_absolute_error(y_discounted, pred_seasonal)
print(f"    R² Score: {r2_seasonal:.4f}")
print(f"    MAE: ${mae_seasonal:,.2f}")
models['Seasonal'] = {
    'model': model_seasonal,
    'predictions': pred_seasonal,
    'r2': r2_seasonal,
    'type': 'seasonal'
}

# Model 4: Polynomial + Seasonal
print("\n  [4/4] Training Polynomial + Seasonal Model...")
X_poly_seasonal = poly_features.fit_transform(monthly_historical[['month_index']].values)
X_poly_seasonal = np.column_stack([
    X_poly_seasonal,
    monthly_historical['month_sin'].values,
    monthly_historical['month_cos'].values
])
model_poly_seasonal = LinearRegression()
model_poly_seasonal.fit(X_poly_seasonal, y_discounted)
pred_poly_seasonal = model_poly_seasonal.predict(X_poly_seasonal)
r2_poly_seasonal = r2_score(y_discounted, pred_poly_seasonal)
mae_poly_seasonal = mean_absolute_error(y_discounted, pred_poly_seasonal)
print(f"    R² Score: {r2_poly_seasonal:.4f}")
print(f"    MAE: ${mae_poly_seasonal:,.2f}")
models['Poly_Seasonal'] = {
    'model': model_poly_seasonal,
    'poly': poly_features,
    'predictions': pred_poly_seasonal,
    'r2': r2_poly_seasonal,
    'type': 'poly_seasonal'
}

# Select best model
best_model_name = max(models.items(), key=lambda x: x[1]['r2'])[0]
best_model_info = models[best_model_name]
print(f"\n  ✓ Selected Best Model: {best_model_name} (R²={best_model_info['r2']:.4f})")

# ================================================================
# STEP 4: Generate forecast
# ================================================================
print("\n" + "="*80)
print("🔮 [Step 4/6] GENERATING FORECAST (NEXT 15 MONTHS)")
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

# Generate forecast based on best model
if best_model_info['type'] == 'linear':
    X_future = future_month_indices.reshape(-1, 1)
    forecast_discounted = best_model_info['model'].predict(X_future)
    
elif best_model_info['type'] == 'polynomial':
    X_future = future_month_indices.reshape(-1, 1)
    X_future_poly = best_model_info['poly'].transform(X_future)
    forecast_discounted = best_model_info['model'].predict(X_future_poly)
    
elif best_model_info['type'] == 'seasonal':
    X_future = np.column_stack([future_month_indices, future_month_sin, future_month_cos])
    forecast_discounted = best_model_info['model'].predict(X_future)
    
elif best_model_info['type'] == 'poly_seasonal':
    X_future_base = future_month_indices.reshape(-1, 1)
    X_future_poly = best_model_info['poly'].transform(X_future_base)
    X_future = np.column_stack([X_future_poly, future_month_sin, future_month_cos])
    forecast_discounted = best_model_info['model'].predict(X_future)

# Ensure non-negative forecasts
forecast_discounted = np.maximum(forecast_discounted, 0)

# Calculate average undiscounted multiplier from historical data
avg_undiscounted_pct = monthly_historical['undiscounted_as_pct'].mean()
print(f"\n  Historical average undiscounted multiplier: {avg_undiscounted_pct:.2f}%")

# Infer undiscounted amounts using the multiplier
forecast_undiscounted = forecast_discounted * (avg_undiscounted_pct / 100)
forecast_discount = forecast_undiscounted - forecast_discounted

# Estimate invoice counts (use recent 6-month average)
recent_avg_count = monthly_historical.tail(6)['n_invoices'].mean()
forecast_invoice_count = np.full(FORECAST_PERIODS, recent_avg_count)
print(f"  Using recent 6-month average invoice count: {recent_avg_count:.0f} invoices/month")

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'invoice_period': future_dates,
    'forecast_discounted_price': forecast_discounted,
    'forecast_undiscounted_price': forecast_undiscounted,
    'forecast_discount_amount': forecast_discount,
    'forecast_invoice_count': forecast_invoice_count.astype(int),
    'forecast_undiscounted_as_pct': avg_undiscounted_pct
})

print("\n  Forecasted Monthly Totals:")
for _, row in forecast_df.iterrows():
    print(f"    {row['invoice_period'].strftime('%Y-%m')}: "
          f"${row['forecast_discounted_price']:>12,.2f} (discounted), "
          f"${row['forecast_undiscounted_price']:>12,.2f} (undiscounted)")

print(f"\n  💰 Total Forecasted (15 months):")
print(f"    Discounted: ${forecast_df['forecast_discounted_price'].sum():,.2f}")
print(f"    Undiscounted: ${forecast_df['forecast_undiscounted_price'].sum():,.2f}")
print(f"    Total discount: ${forecast_df['forecast_discount_amount'].sum():,.2f}")
print(f"    Total invoices: {forecast_df['forecast_invoice_count'].sum():,}")

# ================================================================
# STEP 5: Save results
# ================================================================
print("\n" + "="*80)
print("💾 [Step 5/6] SAVING FORECAST RESULTS")
print("="*80)

# Save forecast CSV
output_csv = OUTPUT_PATH / '11.2_forecast_next_15_months.csv'
forecast_df.to_csv(output_csv, index=False)
print(f"  ✓ Saved: {output_csv.name}")

# Save comprehensive Excel file
output_excel = OUTPUT_PATH / '11.2_forecast_with_historical.xlsx'
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Forecast sheet
    forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
    
    # Historical sheet
    historical_export = monthly_historical[[
        'invoice_period', 'total_discounted_price', 'total_undiscounted_price',
        'discount_amount', 'n_invoices', 'undiscounted_as_pct'
    ]].copy()
    historical_export.to_excel(writer, sheet_name='Historical', index=False)
    
    # Model info sheet
    model_info = pd.DataFrame({
        'Parameter': ['Model Type', 'R² Score', 'MAE', 'Training Period',
                      'Forecast Period', 'Avg Undiscounted Multiplier', 'Avg Invoice Count'],
        'Value': [best_model_name, 
                  f"{best_model_info['r2']:.4f}",
                  f"${mae_linear:,.2f}",
                  f"{monthly_historical['invoice_period'].min().strftime('%Y-%m')} to {monthly_historical['invoice_period'].max().strftime('%Y-%m')}",
                  f"{forecast_df['invoice_period'].min().strftime('%Y-%m')} to {forecast_df['invoice_period'].max().strftime('%Y-%m')}",
                  f"{avg_undiscounted_pct:.2f}%",
                  f"{recent_avg_count:.0f}"]
    })
    model_info.to_excel(writer, sheet_name='Model_Info', index=False)

print(f"  ✓ Saved: {output_excel.name}")

# ================================================================
# STEP 6: Create visualizations
# ================================================================
print("\n" + "="*80)
print("🎨 [Step 6/6] CREATING VISUALIZATIONS")
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

# Plot 2: Model Fit Quality
ax2 = axes[0, 1]
actual = y_discounted
predicted = best_model_info['predictions']
ax2.scatter(actual, predicted, alpha=0.6, s=100, edgecolors='black', linewidths=1)
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_title(f'Model Fit: {best_model_name}\nR² = {best_model_info["r2"]:.4f}',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Actual Discounted Price ($)', fontsize=12)
ax2.set_ylabel('Predicted Discounted Price ($)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 3: Undiscounted vs Discounted Comparison
ax3 = axes[1, 0]
ax3.plot(monthly_historical['invoice_period'], monthly_historical['total_undiscounted_price'],
         marker='o', linewidth=2, label='Historical (Undiscounted)', color='#70AD47', alpha=0.7)
ax3.plot(monthly_historical['invoice_period'], monthly_historical['total_discounted_price'],
         marker='o', linewidth=2, label='Historical (Discounted)', color='black', alpha=0.7)
ax3.plot(forecast_df['invoice_period'], forecast_df['forecast_undiscounted_price'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast (Undiscounted)',
         color='#A9D18E', markersize=8)
ax3.plot(forecast_df['invoice_period'], forecast_df['forecast_discounted_price'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast (Discounted)',
         color='#4472C4', markersize=8)
ax3.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_title('Undiscounted vs Discounted - Historical & Forecast', fontsize=14, fontweight='bold')
ax3.set_xlabel('Period', fontsize=12)
ax3.set_ylabel('Price ($)', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Invoice Count Forecast
ax4 = axes[1, 1]
ax4.plot(monthly_historical['invoice_period'], monthly_historical['n_invoices'],
         marker='o', linewidth=2, label='Historical', color='black', alpha=0.7)
ax4.plot(forecast_df['invoice_period'], forecast_df['forecast_invoice_count'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast (Avg)',
         color='#FFC000', markersize=8)
ax4.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax4.axhline(y=recent_avg_count, color='#FFC000', linestyle=':', linewidth=2, alpha=0.5,
            label=f'6-Month Avg: {recent_avg_count:.0f}')
ax4.set_title('Monthly Invoice Count', fontsize=14, fontweight='bold')
ax4.set_xlabel('Period', fontsize=12)
ax4.set_ylabel('Invoice Count', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.2_forecast_visualization.png'
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
print(f"  Model type: {best_model_name}")
print(f"  R² Score: {best_model_info['r2']:.4f}")
print(f"  Training period: {monthly_historical['invoice_period'].min().strftime('%Y-%m')} to {monthly_historical['invoice_period'].max().strftime('%Y-%m')}")

print(f"\n💰 Forecast Summary:")
print(f"  Forecast period: {forecast_df['invoice_period'].min().strftime('%Y-%m')} to {forecast_df['invoice_period'].max().strftime('%Y-%m')}")
print(f"  Total discounted: ${forecast_df['forecast_discounted_price'].sum():,.2f}")
print(f"  Total undiscounted: ${forecast_df['forecast_undiscounted_price'].sum():,.2f}")
print(f"  Total invoices: {forecast_df['forecast_invoice_count'].sum():,}")

print(f"\n📁 Output Files:")
print(f"  • 11.2_forecast_next_15_months.csv - Monthly forecast data")
print(f"  • 11.2_forecast_with_historical.xlsx - Complete data with historical context")
print(f"  • 11.2_forecast_visualization.png - Visual analysis")

print(f"\n  All files saved to: {OUTPUT_PATH}")

print("\n" + "="*80)
print("NEXT STEP: Apply payment profiles to forecasted totals to estimate revenue")
print("="*80)