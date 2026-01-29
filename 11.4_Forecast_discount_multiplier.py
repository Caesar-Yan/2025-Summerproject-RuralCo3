'''
11.4_Forecast_discount_multiplier - Forecasting Discount Rate with Invoice Count

This script forecasts the discount rate (percentage above 100%) for the next 
15 months using regression models. The discount rate represents how much higher 
the undiscounted price is compared to the discounted price.

For example, if undiscounted_as_pct = 110%, the discount_rate = 10%

All models use invoice count (n_invoices) as a feature to capture the 
relationship between volume and discount behavior.

Inputs:
-------
- visualisations/9.4_monthly_totals_Period_4_Entire.csv

Outputs:
--------
- forecast/11.4_forecast_discount_rate_15_months.csv
- forecast/11.4_discount_rate_forecast_with_historical.xlsx
- forecast/11.4_discount_rate_forecast_visualization.png

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
print("🚀 DISCOUNT RATE FORECAST - NEXT 15 MONTHS")
print("   (All models use Invoice Count as Feature)")
print("   Discount Rate = undiscounted_as_pct - 100")
print("="*80)

# ================================================================
# STEP 1: Load historical data
# ================================================================
print("\n" + "="*80)
print("📁 [Step 1/5] LOADING HISTORICAL DATA")
print("="*80)

# Load the monthly totals
monthly_historical = pd.read_csv(INPUT_FILE)

print(f"  ✓ Loaded {len(monthly_historical)} months of historical data")

# Parse dates
monthly_historical['invoice_period'] = pd.to_datetime(monthly_historical['invoice_period'])
monthly_historical = monthly_historical.sort_values('invoice_period').reset_index(drop=True)

print(f"  Date range: {monthly_historical['invoice_period'].min().strftime('%Y-%m')} to {monthly_historical['invoice_period'].max().strftime('%Y-%m')}")

# Verify undiscounted_as_pct column exists
if 'undiscounted_as_pct' not in monthly_historical.columns:
    print(f"\n❌ ERROR: Column 'undiscounted_as_pct' not found in input file!")
    print(f"   Available columns: {list(monthly_historical.columns)}")
    exit(1)

# Calculate discount rate as percentage above 100%
# If undiscounted_as_pct = 110%, discount_rate = 10%
monthly_historical['discount_rate'] = monthly_historical['undiscounted_as_pct'] - 100

print(f"\n  Calculated discount_rate = undiscounted_as_pct - 100")

# Display summary statistics
print(f"\n  📈 Historical Statistics:")
print(f"    Discount Rate (% above 100):")
print(f"      Mean:   {monthly_historical['discount_rate'].mean():.2f}%")
print(f"      Median: {monthly_historical['discount_rate'].median():.2f}%")
print(f"      Min:    {monthly_historical['discount_rate'].min():.2f}%")
print(f"      Max:    {monthly_historical['discount_rate'].max():.2f}%")
print(f"      Std:    {monthly_historical['discount_rate'].std():.2f}%")
print(f"    Avg monthly invoice count: {monthly_historical['n_invoices'].mean():.0f}")
print(f"    Avg discounted price: ${monthly_historical['total_discounted_price'].mean():,.2f}")

# Check for missing or invalid values
missing_count = monthly_historical['discount_rate'].isna().sum()
if missing_count > 0:
    print(f"\n  ⚠️  WARNING: {missing_count} months have missing discount_rate values")
    print(f"     These will be excluded from modeling")
    monthly_historical = monthly_historical[monthly_historical['discount_rate'].notna()].copy()
    print(f"     Remaining months: {len(monthly_historical)}")

# ================================================================
# STEP 2: Prepare data for modeling
# ================================================================
print("\n" + "="*80)
print("📊 [Step 2/5] PREPARING DATA FOR MODELING")
print("="*80)

# Add time index
monthly_historical['month_index'] = range(len(monthly_historical))

# Add cyclical features for seasonality
monthly_historical['month'] = monthly_historical['invoice_period'].dt.month
monthly_historical['month_sin'] = np.sin(2 * np.pi * monthly_historical['month'] / 12)
monthly_historical['month_cos'] = np.cos(2 * np.pi * monthly_historical['month'] / 12)

print(f"  ✓ Added time features and cyclical components")
print(f"  ✓ All models will use n_invoices as a feature")

# Display the data
print(f"\n  Historical data preview:")
print(monthly_historical[['invoice_period', 'discount_rate', 'undiscounted_as_pct', 
                          'n_invoices', 'total_discounted_price']].head(10).to_string(index=False))

# Check for correlation between features and target
print(f"\n  📊 Correlation Analysis:")
correlation_inv = monthly_historical[['discount_rate', 'n_invoices']].corr().iloc[0, 1]
correlation_time = monthly_historical[['discount_rate', 'month_index']].corr().iloc[0, 1]
print(f"    discount_rate vs n_invoices:   {correlation_inv:.4f}")
print(f"    discount_rate vs month_index:  {correlation_time:.4f}")

# ================================================================
# STEP 3: Build regression models for DISCOUNT RATE
# ================================================================
print("\n" + "="*80)
print("🔧 [Step 3/5] BUILDING REGRESSION MODELS FOR DISCOUNT RATE")
print("   (All models include n_invoices as a feature)")
print("="*80)

models = {}
y_discount_rate = monthly_historical['discount_rate'].values

# Model 1: Linear Regression (time + invoice count)
print("  [1/4] Training Linear Regression (time + invoice count)...")
X_linear = monthly_historical[['month_index', 'n_invoices']].values
model_linear = LinearRegression()
model_linear.fit(X_linear, y_discount_rate)
pred_linear = model_linear.predict(X_linear)
r2_linear = r2_score(y_discount_rate, pred_linear)
mae_linear = mean_absolute_error(y_discount_rate, pred_linear)
print(f"    R² Score: {r2_linear:.4f}")
print(f"    MAE: {mae_linear:.4f}%")
models['Linear'] = {
    'model': model_linear,
    'predictions': pred_linear,
    'r2': r2_linear,
    'mae': mae_linear,
    'type': 'linear'
}

# Model 2: Polynomial Regression (time + invoice count)
print("\n  [2/4] Training Polynomial Regression (degree=2, time + invoice count)...")
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(monthly_historical[['month_index', 'n_invoices']].values)
model_poly = LinearRegression()
model_poly.fit(X_poly, y_discount_rate)
pred_poly = model_poly.predict(X_poly)
r2_poly = r2_score(y_discount_rate, pred_poly)
mae_poly = mean_absolute_error(y_discount_rate, pred_poly)
print(f"    R² Score: {r2_poly:.4f}")
print(f"    MAE: {mae_poly:.4f}%")
models['Polynomial'] = {
    'model': model_poly,
    'poly': poly_features,
    'predictions': pred_poly,
    'r2': r2_poly,
    'mae': mae_poly,
    'type': 'polynomial'
}

# Model 3: Seasonal Model (time + seasonality + invoice count)
print("\n  [3/4] Training Seasonal Model (time + seasonality + invoice count)...")
X_seasonal = monthly_historical[['month_index', 'month_sin', 'month_cos', 'n_invoices']].values
model_seasonal = LinearRegression()
model_seasonal.fit(X_seasonal, y_discount_rate)
pred_seasonal = model_seasonal.predict(X_seasonal)
r2_seasonal = r2_score(y_discount_rate, pred_seasonal)
mae_seasonal = mean_absolute_error(y_discount_rate, pred_seasonal)
print(f"    R² Score: {r2_seasonal:.4f}")
print(f"    MAE: {mae_seasonal:.4f}%")
models['Seasonal'] = {
    'model': model_seasonal,
    'predictions': pred_seasonal,
    'r2': r2_seasonal,
    'mae': mae_seasonal,
    'type': 'seasonal'
}

# Model 4: Polynomial + Seasonal (time^2 + seasonality + invoice count)
print("\n  [4/4] Training Poly + Seasonal (time^2 + seasonality + invoice count)...")
X_base = monthly_historical[['month_index']].values
poly_features_time = PolynomialFeatures(degree=2)
X_poly_time = poly_features_time.fit_transform(X_base)
X_poly_seasonal = np.column_stack([
    X_poly_time,
    monthly_historical['month_sin'].values,
    monthly_historical['month_cos'].values,
    monthly_historical['n_invoices'].values
])
model_poly_seasonal = LinearRegression()
model_poly_seasonal.fit(X_poly_seasonal, y_discount_rate)
pred_poly_seasonal = model_poly_seasonal.predict(X_poly_seasonal)
r2_poly_seasonal = r2_score(y_discount_rate, pred_poly_seasonal)
mae_poly_seasonal = mean_absolute_error(y_discount_rate, pred_poly_seasonal)
print(f"    R² Score: {r2_poly_seasonal:.4f}")
print(f"    MAE: {mae_poly_seasonal:.4f}%")
models['PolySeasonal'] = {
    'model': model_poly_seasonal,
    'poly': poly_features_time,
    'predictions': pred_poly_seasonal,
    'r2': r2_poly_seasonal,
    'mae': mae_poly_seasonal,
    'type': 'poly_seasonal'
}

# Select best model
best_model_name = max(models.items(), key=lambda x: x[1]['r2'])[0]
best_model_info = models[best_model_name]
print(f"\n  ✓ Selected Best Model: {best_model_name} (R²={best_model_info['r2']:.4f}, MAE={best_model_info['mae']:.4f}%)")

# Display model comparison
print(f"\n  📊 Model Comparison:")
print(f"  {'Model':<25} {'R²':<10} {'MAE (pp)':<12} {'Selected':<10}")
print(f"  {'-'*60}")
for name, info in sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True):
    selected = '✓' if name == best_model_name else ''
    print(f"  {name:<25} {info['r2']:<10.4f} {info['mae']:<12.4f} {selected:<10}")

# ================================================================
# STEP 4: Generate forecast
# ================================================================
print("\n" + "="*80)
print("🔮 [Step 4/5] GENERATING FORECAST (NEXT 15 MONTHS)")
print("="*80)

# Load forecasted invoice counts from 11.3 if available
forecast_inv_file = BASE_PATH / "visualisations" / "11.3_forecast_next_15_months.csv"
if forecast_inv_file.exists():
    print(f"  ✓ Loading forecasted invoice counts from 11.3...")
    forecast_11_3 = pd.read_csv(forecast_inv_file)
    forecast_11_3['invoice_period'] = pd.to_datetime(forecast_11_3['invoice_period'])
    forecasted_invoice_counts = forecast_11_3['forecast_invoice_count'].values
    future_dates = forecast_11_3['invoice_period'].values
    print(f"    Loaded {len(forecasted_invoice_counts)} months of forecasted invoice counts")
else:
    print(f"  ⚠️  Warning: 11.3 forecast not found, will use average historical invoice count")
    last_date = monthly_historical['invoice_period'].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=FORECAST_PERIODS,
        freq='MS'
    )
    avg_invoice_count = monthly_historical['n_invoices'].mean()
    forecasted_invoice_counts = np.full(FORECAST_PERIODS, avg_invoice_count)
    print(f"    Using average invoice count: {avg_invoice_count:.0f}")

# Generate future indices and features
last_month_index = monthly_historical['month_index'].max()
future_month_indices = np.arange(last_month_index + 1, last_month_index + 1 + FORECAST_PERIODS)
future_months = [pd.Timestamp(d).month for d in future_dates]
future_month_sin = [np.sin(2 * np.pi * m / 12) for m in future_months]
future_month_cos = [np.cos(2 * np.pi * m / 12) for m in future_months]

# Generate forecast based on selected model
print(f"\n  Generating forecast using {best_model_name} model...")

if best_model_info['type'] == 'linear':
    X_future = np.column_stack([future_month_indices, forecasted_invoice_counts])
    forecast_discount_rate = best_model_info['model'].predict(X_future)
    
elif best_model_info['type'] == 'polynomial':
    X_future = np.column_stack([future_month_indices, forecasted_invoice_counts])
    X_future_poly = best_model_info['poly'].transform(X_future)
    forecast_discount_rate = best_model_info['model'].predict(X_future_poly)
    
elif best_model_info['type'] == 'seasonal':
    X_future = np.column_stack([future_month_indices, future_month_sin, future_month_cos, 
                                forecasted_invoice_counts])
    forecast_discount_rate = best_model_info['model'].predict(X_future)
    
elif best_model_info['type'] == 'poly_seasonal':
    X_future_base = future_month_indices.reshape(-1, 1)
    X_future_poly = best_model_info['poly'].transform(X_future_base)
    X_future = np.column_stack([X_future_poly, future_month_sin, future_month_cos, 
                                forecasted_invoice_counts])
    forecast_discount_rate = best_model_info['model'].predict(X_future)

# Ensure reasonable bounds (discount rate should be positive and reasonable)
historical_min = monthly_historical['discount_rate'].min()
historical_max = monthly_historical['discount_rate'].max()
forecast_discount_rate = np.clip(forecast_discount_rate, 
                                  max(0, historical_min * 0.9), 
                                  historical_max * 1.1)

# Convert back to undiscounted_as_pct format (add 100)
forecast_undiscounted_as_pct = forecast_discount_rate + 100

print(f"  ✓ Forecast generated")
print(f"    Forecasted discount rate range: {forecast_discount_rate.min():.2f}% - {forecast_discount_rate.max():.2f}%")
print(f"    Forecasted discount rate mean: {forecast_discount_rate.mean():.2f}%")
print(f"    (Equivalent to undiscounted_as_pct: {forecast_undiscounted_as_pct.min():.2f}% - {forecast_undiscounted_as_pct.max():.2f}%)")

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'invoice_period': future_dates,
    'forecast_discount_rate': forecast_discount_rate,
    'forecast_undiscounted_as_pct': forecast_undiscounted_as_pct,
    'forecast_invoice_count': forecasted_invoice_counts
})

print("\n  Forecasted Discount Rates:")
for _, row in forecast_df.iterrows():
    print(f"    {pd.Timestamp(row['invoice_period']).strftime('%Y-%m')}: "
          f"{row['forecast_discount_rate']:>6.2f}% "
          f"(undiscounted_as_pct: {row['forecast_undiscounted_as_pct']:>6.2f}%) "
          f"[{row['forecast_invoice_count']:>5,.0f} invoices]")

print(f"\n  📊 Forecast Summary:")
print(f"    Average forecasted discount rate: {forecast_df['forecast_discount_rate'].mean():.2f}%")
print(f"    Historical average discount rate: {monthly_historical['discount_rate'].mean():.2f}%")
print(f"    Difference: {forecast_df['forecast_discount_rate'].mean() - monthly_historical['discount_rate'].mean():.2f} pp")

# ================================================================
# STEP 5: Save results
# ================================================================
print("\n" + "="*80)
print("💾 [Step 5/5] SAVING FORECAST RESULTS")
print("="*80)

# Save forecast CSV
output_csv = OUTPUT_PATH / '11.4_forecast_discount_rate_15_months.csv'
forecast_df.to_csv(output_csv, index=False)
print(f"  ✓ Saved: {output_csv.name}")

# Save comprehensive Excel file
output_excel = OUTPUT_PATH / '11.4_discount_rate_forecast_with_historical.xlsx'
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Forecast sheet
    forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
    
    # Historical sheet
    historical_export = monthly_historical[[
        'invoice_period', 'discount_rate', 'undiscounted_as_pct', 'n_invoices',
        'total_discounted_price', 'total_undiscounted_price', 'discount_amount'
    ]].copy()
    historical_export.to_excel(writer, sheet_name='Historical', index=False)
    
    # Model comparison sheet
    model_comparison = pd.DataFrame({
        'Model': list(models.keys()),
        'R²': [m['r2'] for m in models.values()],
        'MAE': [m['mae'] for m in models.values()],
        'Type': [m['type'] for m in models.values()],
        'Selected': ['✓' if k == best_model_name else '' for k in models.keys()]
    })
    model_comparison = model_comparison.sort_values('R²', ascending=False)
    model_comparison.to_excel(writer, sheet_name='Model_Comparison', index=False)
    
    # Predictions vs Actuals
    predictions_df = pd.DataFrame({
        'invoice_period': monthly_historical['invoice_period'],
        'actual_discount_rate': y_discount_rate,
        'predicted_discount_rate': best_model_info['predictions'],
        'residual': y_discount_rate - best_model_info['predictions'],
        'actual_undiscounted_as_pct': monthly_historical['undiscounted_as_pct'],
        'predicted_undiscounted_as_pct': best_model_info['predictions'] + 100,
        'n_invoices': monthly_historical['n_invoices']
    })
    predictions_df.to_excel(writer, sheet_name='Predictions_vs_Actual', index=False)
    
    # Summary info sheet
    summary_info = pd.DataFrame({
        'Parameter': [
            'Best Model',
            'Model R²',
            'Model MAE (pp)',
            'Training Period',
            'Forecast Period',
            'Historical Avg Discount Rate',
            'Forecast Avg Discount Rate',
            'Difference (pp)',
            'Historical Avg undiscounted_as_pct',
            'Forecast Avg undiscounted_as_pct',
            'Forecast Min Discount Rate',
            'Forecast Max Discount Rate',
            'All Models Use Invoice Count'
        ],
        'Value': [
            best_model_name,
            f"{best_model_info['r2']:.4f}",
            f"{best_model_info['mae']:.4f}%",
            f"{monthly_historical['invoice_period'].min().strftime('%Y-%m')} to {monthly_historical['invoice_period'].max().strftime('%Y-%m')}",
            f"{forecast_df['invoice_period'].min().strftime('%Y-%m')} to {forecast_df['invoice_period'].max().strftime('%Y-%m')}",
            f"{monthly_historical['discount_rate'].mean():.2f}%",
            f"{forecast_df['forecast_discount_rate'].mean():.2f}%",
            f"{forecast_df['forecast_discount_rate'].mean() - monthly_historical['discount_rate'].mean():.2f}",
            f"{monthly_historical['undiscounted_as_pct'].mean():.2f}%",
            f"{forecast_df['forecast_undiscounted_as_pct'].mean():.2f}%",
            f"{forecast_df['forecast_discount_rate'].min():.2f}%",
            f"{forecast_df['forecast_discount_rate'].max():.2f}%",
            'Yes (all 4 models)'
        ]
    })
    summary_info.to_excel(writer, sheet_name='Summary', index=False)

print(f"  ✓ Saved: {output_excel.name}")

# ================================================================
# VISUALIZATIONS
# ================================================================
print("\n" + "="*80)
print("🎨 CREATING VISUALIZATIONS")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(monthly_historical['invoice_period'], monthly_historical['discount_rate'],
         marker='o', linewidth=2, label='Historical', color='black', alpha=0.7)
ax.plot(forecast_df['invoice_period'], forecast_df['forecast_discount_rate'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast',
         color='#4472C4', markersize=8)

# Add historical average line
hist_avg = monthly_historical['discount_rate'].mean()
ax.axhline(y=hist_avg, color='gray', linestyle=':', linewidth=2, alpha=0.5,
            label=f'Historical Avg: {hist_avg:.2f}%')

# Add forecast start line
last_date = monthly_historical['invoice_period'].max()
ax.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5, 
            label='Forecast Start')

ax.set_title('Discount Rate - Historical & Forecast\n(Percentage above 100%)', 
              fontsize=14, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Discount Rate (%)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz_1 = OUTPUT_PATH / '11.4_discount_rate_forecast.png'
plt.savefig(output_viz_1, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz_1.name}")
plt.close()

# Plot 2: Model Fit Quality
fig, ax = plt.subplots(figsize=(10, 8))
actual = y_discount_rate
predicted = best_model_info['predictions']
ax.scatter(actual, predicted, alpha=0.6, s=100, edgecolors='black', linewidths=1)
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Model Fit: {best_model_name}\nR² = {best_model_info["r2"]:.4f}, MAE = {best_model_info["mae"]:.4f}%',
              fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Discount Rate (%)', fontsize=12)
ax.set_ylabel('Predicted Discount Rate (%)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

plt.tight_layout()
output_viz_2 = OUTPUT_PATH / '11.4_model_fit_quality.png'
plt.savefig(output_viz_2, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz_2.name}")
plt.close()

# Plot 3: Residuals over time
fig, ax = plt.subplots(figsize=(14, 6))
residuals = y_discount_rate - best_model_info['predictions']
ax.scatter(monthly_historical['invoice_period'], residuals, alpha=0.6, s=80, 
            edgecolors='black', linewidths=1, color='coral')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.fill_between(monthly_historical['invoice_period'], residuals, 0, alpha=0.3, color='coral')
ax.set_title('Model Residuals Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Residual (Actual - Predicted) (%)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

# Add statistics text
residual_mean = residuals.mean()
residual_std = residuals.std()
ax.text(0.02, 0.98, f'Mean: {residual_mean:.3f}%\nStd: {residual_std:.3f}%',
         transform=ax.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
output_viz_3 = OUTPUT_PATH / '11.4_residuals_over_time.png'
plt.savefig(output_viz_3, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz_3.name}")
plt.close()

# Plot 4: Discount Rate vs Invoice Count
fig, ax = plt.subplots(figsize=(10, 8))
# Show relationship between invoice count and discount rate
ax.scatter(monthly_historical['n_invoices'], monthly_historical['discount_rate'],
            alpha=0.6, s=100, edgecolors='black', linewidths=1, color='#70AD47',
            label='Historical')
ax.scatter(forecast_df['forecast_invoice_count'], forecast_df['forecast_discount_rate'],
            alpha=0.8, s=120, edgecolors='black', linewidths=2, color='#FFC000',
            marker='s', label='Forecast')

# Add trend line
from scipy import stats as scipy_stats
slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
    monthly_historical['n_invoices'], monthly_historical['discount_rate']
)
line_x = np.array([monthly_historical['n_invoices'].min(), 
                   monthly_historical['n_invoices'].max()])
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'r--', linewidth=2, alpha=0.5, 
         label=f'Trend (R={r_value:.3f})')

ax.set_title('Discount Rate vs Invoice Count\n(All models use invoice count)', 
              fontsize=14, fontweight='bold')
ax.set_xlabel('Invoice Count', fontsize=12)
ax.set_ylabel('Discount Rate (%)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

plt.tight_layout()
output_viz_4 = OUTPUT_PATH / '11.4_discount_rate_vs_invoice_count.png'
plt.savefig(output_viz_4, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_viz_4.name}")
plt.close()

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("✅ DISCOUNT RATE FORECAST COMPLETE!")
print("="*80)

print(f"\n📊 Model Performance:")
print(f"  Selected Model: {best_model_name}")
print(f"  R² Score: {best_model_info['r2']:.4f}")
print(f"  MAE: {best_model_info['mae']:.4f} percentage points")
print(f"  All models use invoice count as a feature")

print(f"\n💰 Discount Rate Forecast Summary:")
print(f"  Historical Average: {monthly_historical['discount_rate'].mean():.2f}%")
print(f"  Forecast Average: {forecast_df['forecast_discount_rate'].mean():.2f}%")
print(f"  Difference: {forecast_df['forecast_discount_rate'].mean() - monthly_historical['discount_rate'].mean():.2f} pp")
print(f"  Forecast Range: {forecast_df['forecast_discount_rate'].min():.2f}% - {forecast_df['forecast_discount_rate'].max():.2f}%")

print(f"\n  (In undiscounted_as_pct terms:)")
print(f"  Historical Average: {monthly_historical['undiscounted_as_pct'].mean():.2f}%")
print(f"  Forecast Average: {forecast_df['forecast_undiscounted_as_pct'].mean():.2f}%")

print(f"\n📈 Interpretation:")
if forecast_df['forecast_discount_rate'].mean() > monthly_historical['discount_rate'].mean():
    diff = forecast_df['forecast_discount_rate'].mean() - monthly_historical['discount_rate'].mean()
    print(f"  ↑ Discount gap is expected to INCREASE by {diff:.2f} pp over the next 15 months")
    print(f"    (Higher discount rate = larger gap between undiscounted and discounted prices)")
else:
    diff = monthly_historical['discount_rate'].mean() - forecast_df['forecast_discount_rate'].mean()
    print(f"  ↓ Discount gap is expected to DECREASE by {diff:.2f} pp over the next 15 months")
    print(f"    (Lower discount rate = smaller gap between undiscounted and discounted prices)")

print(f"\n📁 Output Files:")
print(f"  All files saved to: {OUTPUT_PATH}")
print(f"  • 11.4_forecast_discount_rate_15_months.csv - Monthly discount rate forecast")
print(f"  • 11.4_discount_rate_forecast_with_historical.xlsx - Complete data with model info")
print(f"  • 11.4_discount_rate_forecast_visualization.png - 4-panel visual analysis")

print("\n" + "="*80)
print("NEXT STEPS:")
print("  1. Use forecast_undiscounted_as_pct to improve 11.3 undiscounted predictions")
print("  2. Combine with 11.3 discounted forecasts for more accurate revenue estimates")
print("  3. Apply to revenue forecasting in scripts 15.x series")
print("  4. Compare with 11.3's constant multiplier approach")
print("="*80)