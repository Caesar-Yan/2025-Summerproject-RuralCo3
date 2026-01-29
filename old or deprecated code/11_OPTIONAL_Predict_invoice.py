import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# CONFIGURATION
# ================================================================
OUTPUT_DIR = "regression_outputs"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# Load transformed grouped invoice data
# ================================================================
print("="*80)
print("LOADING INVOICE DATA")
print("="*80)

ats_grouped = pd.read_csv('ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv('invoice_grouped_transformed_with_discounts.csv')

# Add customer type identifier
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'

# Combine datasets
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)
print(f"Total invoices loaded: {len(combined_df):,}")

# ================================================================
# Parse invoice_period dates
# ================================================================
def parse_invoice_period(series: pd.Series) -> pd.Series:
    """Robustly parse invoice_period"""
    s = series.copy()
    s_str = s.astype(str).str.strip()
    s_str = s_str.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
    
    # Handle YYYYMM format
    mask_yyyymm = s_str.str.fullmatch(r"\d{6}", na=False)
    out = pd.Series(pd.NaT, index=s.index)
    
    if mask_yyyymm.any():
        out.loc[mask_yyyymm] = pd.to_datetime(s_str.loc[mask_yyyymm], format="%Y%m", errors="coerce")
    
    # Handle other formats
    mask_other = ~mask_yyyymm
    if mask_other.any():
        out.loc[mask_other] = pd.to_datetime(s_str.loc[mask_other], errors="coerce")
    
    return out

combined_df['invoice_period'] = parse_invoice_period(combined_df['invoice_period'])
combined_df = combined_df[combined_df['invoice_period'].notna()].copy()

print(f"Valid invoices after date parsing: {len(combined_df):,}")
print(f"Date range: {combined_df['invoice_period'].min()} to {combined_df['invoice_period'].max()}")

# ================================================================
# Aggregate to monthly totals
# ================================================================
print("\n" + "="*80)
print("AGGREGATING TO MONTHLY TOTALS")
print("="*80)

monthly_totals = combined_df.groupby(
    combined_df['invoice_period'].dt.to_period('M')
).agg({
    'total_undiscounted_price': 'sum',
    'total_discounted_price': 'sum',
    'discount_amount': 'sum',
    'invoice_id': 'count'
}).reset_index()

monthly_totals.columns = ['period', 'total_undiscounted', 'total_discounted', 
                          'total_discount', 'invoice_count']
monthly_totals['period'] = monthly_totals['period'].dt.to_timestamp()

# Sort by date
monthly_totals = monthly_totals.sort_values('period').reset_index(drop=True)

print(f"Monthly periods: {len(monthly_totals)}")
print("\nFirst few months:")
print(monthly_totals.head())
print("\nLast few months:")
print(monthly_totals.tail())

# ================================================================
# Create time-based features
# ================================================================
monthly_totals['month_index'] = range(len(monthly_totals))
monthly_totals['year'] = monthly_totals['period'].dt.year
monthly_totals['month'] = monthly_totals['period'].dt.month
monthly_totals['quarter'] = monthly_totals['period'].dt.quarter

# Identify 2025+ data
monthly_totals['is_2025_onwards'] = (monthly_totals['year'] >= 2025).astype(int)

# Calculate cumulative totals
monthly_totals['cumulative_undiscounted'] = monthly_totals['total_undiscounted'].cumsum()
monthly_totals['cumulative_discounted'] = monthly_totals['total_discounted'].cumsum()

print(f"\nData split: {monthly_totals['is_2025_onwards'].value_counts().to_dict()}")

# ================================================================
# VISUALIZATION: Data exploration
# ================================================================
print("\n" + "="*80)
print("CREATING DATA EXPLORATION PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Plot 1: Monthly invoice totals (undiscounted)
ax1 = axes[0, 0]
ax1.plot(monthly_totals['period'], monthly_totals['total_undiscounted'], 
         marker='o', linewidth=2, color='#4472C4')
ax1.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--', 
            linewidth=2, label='2025 Start', alpha=0.7)
ax1.set_title('Monthly Invoice Totals (Undiscounted)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Period', fontsize=12)
ax1.set_ylabel('Total Amount ($)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: Monthly invoice count
ax2 = axes[0, 1]
ax2.bar(monthly_totals['period'], monthly_totals['invoice_count'], 
        color='#70AD47', alpha=0.7)
ax2.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--', 
            linewidth=2, label='2025 Start', alpha=0.7)
ax2.set_title('Monthly Invoice Count', fontsize=14, fontweight='bold')
ax2.set_xlabel('Period', fontsize=12)
ax2.set_ylabel('Number of Invoices', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Cumulative totals
ax3 = axes[1, 0]
ax3.plot(monthly_totals['period'], monthly_totals['cumulative_undiscounted'], 
         marker='o', linewidth=2, label='Undiscounted', color='#4472C4')
ax3.plot(monthly_totals['period'], monthly_totals['cumulative_discounted'], 
         marker='s', linewidth=2, label='Discounted', color='#70AD47')
ax3.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--', 
            linewidth=2, label='2025 Start', alpha=0.7)
ax3.set_title('Cumulative Invoice Totals', fontsize=14, fontweight='bold')
ax3.set_xlabel('Period', fontsize=12)
ax3.set_ylabel('Cumulative Amount ($)', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 4: Monthly discount amounts
ax4 = axes[1, 1]
ax4.bar(monthly_totals['period'], monthly_totals['total_discount'], 
        color='#FF6B6B', alpha=0.7)
ax4.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--', 
            linewidth=2, label='2025 Start', alpha=0.7)
ax4.set_title('Monthly Discount Amounts', fontsize=14, fontweight='bold')
ax4.set_xlabel('Period', fontsize=12)
ax4.set_ylabel('Discount Amount ($)', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_data_exploration.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/01_data_exploration.png")
plt.close()

# ================================================================
# REGRESSION MODELS
# ================================================================
print("\n" + "="*80)
print("BUILDING REGRESSION MODELS")
print("="*80)

results_summary = []

# ================================================================
# Model 1: Simple Linear Regression (All Data)
# ================================================================
print("\n1. Simple Linear Regression (All Data)")
print("-" * 70)

X_all = monthly_totals[['month_index']].values
y_all = monthly_totals['total_undiscounted'].values

model_linear_all = LinearRegression()
model_linear_all.fit(X_all, y_all)
y_pred_linear_all = model_linear_all.predict(X_all)

r2_linear_all = r2_score(y_all, y_pred_linear_all)
mae_linear_all = mean_absolute_error(y_all, y_pred_linear_all)
rmse_linear_all = np.sqrt(mean_squared_error(y_all, y_pred_linear_all))

print(f"R² Score: {r2_linear_all:.4f}")
print(f"MAE: ${mae_linear_all:,.2f}")
print(f"RMSE: ${rmse_linear_all:,.2f}")
print(f"Coefficient: {model_linear_all.coef_[0]:,.2f}")
print(f"Intercept: ${model_linear_all.intercept_:,.2f}")

results_summary.append({
    'Model': 'Linear Regression (All Data)',
    'R2': r2_linear_all,
    'MAE': mae_linear_all,
    'RMSE': rmse_linear_all,
    'Data_Used': 'All',
    'N_Observations': len(X_all)
})

# ================================================================
# Model 2: Linear Regression (2025 onwards only)
# ================================================================
print("\n2. Linear Regression (2025 Onwards Only)")
print("-" * 70)

monthly_2025 = monthly_totals[monthly_totals['year'] >= 2025].copy()
monthly_2025['month_index_2025'] = range(len(monthly_2025))

X_2025 = monthly_2025[['month_index_2025']].values
y_2025 = monthly_2025['total_undiscounted'].values

if len(X_2025) > 1:
    model_linear_2025 = LinearRegression()
    model_linear_2025.fit(X_2025, y_2025)
    y_pred_linear_2025 = model_linear_2025.predict(X_2025)
    
    r2_linear_2025 = r2_score(y_2025, y_pred_linear_2025)
    mae_linear_2025 = mean_absolute_error(y_2025, y_pred_linear_2025)
    rmse_linear_2025 = np.sqrt(mean_squared_error(y_2025, y_pred_linear_2025))
    
    print(f"R² Score: {r2_linear_2025:.4f}")
    print(f"MAE: ${mae_linear_2025:,.2f}")
    print(f"RMSE: ${rmse_linear_2025:,.2f}")
    print(f"Coefficient: {model_linear_2025.coef_[0]:,.2f}")
    print(f"Intercept: ${model_linear_2025.intercept_:,.2f}")
    
    results_summary.append({
        'Model': 'Linear Regression (2025 Only)',
        'R2': r2_linear_2025,
        'MAE': mae_linear_2025,
        'RMSE': rmse_linear_2025,
        'Data_Used': '2025+',
        'N_Observations': len(X_2025)
    })
else:
    print("⚠ Not enough 2025 data for regression")

# ================================================================
# Model 3: Polynomial Regression (All Data, degree=2)
# ================================================================
print("\n3. Polynomial Regression (All Data, degree=2)")
print("-" * 70)

poly_features = PolynomialFeatures(degree=2)
X_poly_all = poly_features.fit_transform(X_all)

model_poly_all = LinearRegression()
model_poly_all.fit(X_poly_all, y_all)
y_pred_poly_all = model_poly_all.predict(X_poly_all)

r2_poly_all = r2_score(y_all, y_pred_poly_all)
mae_poly_all = mean_absolute_error(y_all, y_pred_poly_all)
rmse_poly_all = np.sqrt(mean_squared_error(y_all, y_pred_poly_all))

print(f"R² Score: {r2_poly_all:.4f}")
print(f"MAE: ${mae_poly_all:,.2f}")
print(f"RMSE: ${rmse_poly_all:,.2f}")

results_summary.append({
    'Model': 'Polynomial Regression (deg=2, All Data)',
    'R2': r2_poly_all,
    'MAE': mae_poly_all,
    'RMSE': rmse_poly_all,
    'Data_Used': 'All',
    'N_Observations': len(X_all)
})

# ================================================================
# Model 4: Polynomial Regression (2025 onwards, degree=2)
# ================================================================
print("\n4. Polynomial Regression (2025 Onwards, degree=2)")
print("-" * 70)

if len(X_2025) > 2:
    X_poly_2025 = poly_features.fit_transform(X_2025)
    
    model_poly_2025 = LinearRegression()
    model_poly_2025.fit(X_poly_2025, y_2025)
    y_pred_poly_2025 = model_poly_2025.predict(X_poly_2025)
    
    r2_poly_2025 = r2_score(y_2025, y_pred_poly_2025)
    mae_poly_2025 = mean_absolute_error(y_2025, y_pred_poly_2025)
    rmse_poly_2025 = np.sqrt(mean_squared_error(y_2025, y_pred_poly_2025))
    
    print(f"R² Score: {r2_poly_2025:.4f}")
    print(f"MAE: ${mae_poly_2025:,.2f}")
    print(f"RMSE: ${rmse_poly_2025:,.2f}")
    
    results_summary.append({
        'Model': 'Polynomial Regression (deg=2, 2025 Only)',
        'R2': r2_poly_2025,
        'MAE': mae_poly_2025,
        'RMSE': rmse_poly_2025,
        'Data_Used': '2025+',
        'N_Observations': len(X_2025)
    })
else:
    print("⚠ Not enough 2025 data for polynomial regression")

# ================================================================
# Model 5: OLS with Seasonal Dummies (All Data)
# ================================================================
print("\n5. OLS with Seasonal Components (All Data)")
print("-" * 70)

# Create seasonal dummies
X_seasonal = pd.get_dummies(monthly_totals[['month_index', 'month', 'quarter']], 
                            columns=['month', 'quarter'], drop_first=True)
X_seasonal_sm = sm.add_constant(X_seasonal)

model_seasonal = sm.OLS(y_all, X_seasonal_sm)
results_seasonal = model_seasonal.fit()

y_pred_seasonal = results_seasonal.predict(X_seasonal_sm)
r2_seasonal = r2_score(y_all, y_pred_seasonal)
mae_seasonal = mean_absolute_error(y_all, y_pred_seasonal)
rmse_seasonal = np.sqrt(mean_squared_error(y_all, y_pred_seasonal))

print(f"R² Score: {r2_seasonal:.4f}")
print(f"Adj. R² Score: {results_seasonal.rsquared_adj:.4f}")
print(f"MAE: ${mae_seasonal:,.2f}")
print(f"RMSE: ${rmse_seasonal:,.2f}")
print(f"\nModel Summary (first few coefficients):")
print(results_seasonal.summary().tables[1].as_text()[:500])

results_summary.append({
    'Model': 'OLS with Seasonal Dummies (All Data)',
    'R2': r2_seasonal,
    'MAE': mae_seasonal,
    'RMSE': rmse_seasonal,
    'Data_Used': 'All',
    'N_Observations': len(X_all)
})

# ================================================================
# Model 6: Exponential Smoothing (All Data)
# ================================================================
print("\n6. Exponential Smoothing (Holt-Winters)")
print("-" * 70)

try:
    # Set period as index for time series
    ts_data = monthly_totals.set_index('period')['total_undiscounted']
    
    # Fit Holt-Winters model (additive seasonality, 12-month period)
    model_hw = ExponentialSmoothing(
        ts_data, 
        seasonal_periods=12, 
        trend='add', 
        seasonal='add'
    ).fit()
    
    y_pred_hw = model_hw.fittedvalues
    
    r2_hw = r2_score(y_all, y_pred_hw)
    mae_hw = mean_absolute_error(y_all, y_pred_hw)
    rmse_hw = np.sqrt(mean_squared_error(y_all, y_pred_hw))
    
    print(f"R² Score: {r2_hw:.4f}")
    print(f"MAE: ${mae_hw:,.2f}")
    print(f"RMSE: ${rmse_hw:,.2f}")
    
    results_summary.append({
        'Model': 'Holt-Winters Exponential Smoothing',
        'R2': r2_hw,
        'MAE': mae_hw,
        'RMSE': rmse_hw,
        'Data_Used': 'All',
        'N_Observations': len(y_all)
    })
except Exception as e:
    print(f"⚠ Exponential Smoothing failed: {str(e)}")

# ================================================================
# Model 7: ARIMA (All Data)
# ================================================================
print("\n7. ARIMA Time Series Model")
print("-" * 70)

try:
    # Fit ARIMA(1,1,1) model
    model_arima = ARIMA(ts_data, order=(1, 1, 1)).fit()
    
    y_pred_arima = model_arima.fittedvalues
    
    # Align predictions with original data (ARIMA may drop first observation)
    y_aligned = y_all[-len(y_pred_arima):]
    
    r2_arima = r2_score(y_aligned, y_pred_arima)
    mae_arima = mean_absolute_error(y_aligned, y_pred_arima)
    rmse_arima = np.sqrt(mean_squared_error(y_aligned, y_pred_arima))
    
    print(f"R² Score: {r2_arima:.4f}")
    print(f"MAE: ${mae_arima:,.2f}")
    print(f"RMSE: ${rmse_arima:,.2f}")
    print(f"AIC: {model_arima.aic:.2f}")
    print(f"BIC: {model_arima.bic:.2f}")
    
    results_summary.append({
        'Model': 'ARIMA(1,1,1)',
        'R2': r2_arima,
        'MAE': mae_arima,
        'RMSE': rmse_arima,
        'Data_Used': 'All',
        'N_Observations': len(y_aligned)
    })
except Exception as e:
    print(f"⚠ ARIMA failed: {str(e)}")

# ================================================================
# Create Results Summary DataFrame
# ================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

results_df = pd.DataFrame(results_summary)
results_df = results_df.sort_values('R2', ascending=False).reset_index(drop=True)

print(results_df.to_string(index=False))

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, 'regression_results_summary.csv'), index=False)
print(f"\n✓ Saved: {OUTPUT_DIR}/regression_results_summary.csv")

# ================================================================
# VISUALIZATION: Model Predictions Comparison
# ================================================================
print("\n" + "="*80)
print("CREATING PREDICTION COMPARISON PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Plot 1: Linear Regression (All Data)
ax1 = axes[0, 0]
ax1.scatter(monthly_totals['period'], y_all, alpha=0.6, label='Actual', color='black')
ax1.plot(monthly_totals['period'], y_pred_linear_all, 
         color='#4472C4', linewidth=2, label='Linear Fit')
ax1.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--', alpha=0.5)
ax1.set_title(f'Linear Regression (All Data)\nR²={r2_linear_all:.4f}', 
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Period')
ax1.set_ylabel('Invoice Total ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: Polynomial Regression (All Data)
ax2 = axes[0, 1]
ax2.scatter(monthly_totals['period'], y_all, alpha=0.6, label='Actual', color='black')
ax2.plot(monthly_totals['period'], y_pred_poly_all, 
         color='#70AD47', linewidth=2, label='Polynomial Fit (deg=2)')
ax2.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--', alpha=0.5)
ax2.set_title(f'Polynomial Regression (All Data)\nR²={r2_poly_all:.4f}', 
              fontsize=12, fontweight='bold')
ax2.set_xlabel('Period')
ax2.set_ylabel('Invoice Total ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 3: OLS with Seasonality
ax3 = axes[1, 0]
ax3.scatter(monthly_totals['period'], y_all, alpha=0.6, label='Actual', color='black')
ax3.plot(monthly_totals['period'], y_pred_seasonal, 
         color='#FF6B6B', linewidth=2, label='OLS + Seasonal')
ax3.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--', alpha=0.5)
ax3.set_title(f'OLS with Seasonal Dummies\nR²={r2_seasonal:.4f}', 
              fontsize=12, fontweight='bold')
ax3.set_xlabel('Period')
ax3.set_ylabel('Invoice Total ($)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 4: Comparison of all models
ax4 = axes[1, 1]
ax4.scatter(monthly_totals['period'], y_all, alpha=0.6, label='Actual', 
            color='black', s=50, zorder=5)
ax4.plot(monthly_totals['period'], y_pred_linear_all, 
         linewidth=2, label=f'Linear (R²={r2_linear_all:.3f})', alpha=0.7)
ax4.plot(monthly_totals['period'], y_pred_poly_all, 
         linewidth=2, label=f'Polynomial (R²={r2_poly_all:.3f})', alpha=0.7)
ax4.plot(monthly_totals['period'], y_pred_seasonal, 
         linewidth=2, label=f'OLS+Seasonal (R²={r2_seasonal:.3f})', alpha=0.7)
ax4.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--', alpha=0.5)
ax4.set_title('All Models Comparison', fontsize=12, fontweight='bold')
ax4.set_xlabel('Period')
ax4.set_ylabel('Invoice Total ($)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_model_predictions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/02_model_predictions.png")
plt.close()

# ================================================================
# VISUALIZATION: 2025 Data Focus
# ================================================================
if len(monthly_2025) > 1:
    print("\n" + "="*80)
    print("CREATING 2025-FOCUSED ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: 2025 Linear Regression
    ax1 = axes[0]
    ax1.scatter(monthly_2025['period'], y_2025, alpha=0.8, label='Actual (2025)', 
                color='#4472C4', s=100)
    ax1.plot(monthly_2025['period'], y_pred_linear_2025, 
             color='#FF6B6B', linewidth=3, label='Linear Fit', linestyle='--')
    ax1.set_title(f'Linear Regression (2025 Data Only)\nR²={r2_linear_2025:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Period', fontsize=12)
    ax1.set_ylabel('Invoice Total ($)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Residuals for 2025 model
    ax2 = axes[1]
    residuals_2025 = y_2025 - y_pred_linear_2025
    ax2.scatter(monthly_2025['period'], residuals_2025, alpha=0.8, 
                color='#70AD47', s=100)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Residuals (2025 Linear Model)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Period', fontsize=12)
    ax2.set_ylabel('Residual ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_2025_focus_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/03_2025_focus_analysis.png")
    plt.close()

# ================================================================
# FORECASTING: Predict next 6 months
# ================================================================
print("\n" + "="*80)
print("FORECASTING NEXT 6 MONTHS")
print("="*80)

last_month_index = monthly_totals['month_index'].max()
future_months = np.arange(last_month_index + 1, last_month_index + 7).reshape(-1, 1)
last_date = monthly_totals['period'].max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')

# Forecast with linear model (all data)
forecast_linear = model_linear_all.predict(future_months)

# Forecast with polynomial model (all data)
future_poly = poly_features.transform(future_months)
forecast_poly = model_poly_all.predict(future_poly)

forecast_df = pd.DataFrame({
    'period': future_dates,
    'month_index': future_months.flatten(),
    'forecast_linear': forecast_linear,
    'forecast_polynomial': forecast_poly
})

print("\nForecasted Invoice Totals (Next 6 Months):")
print(forecast_df.to_string(index=False))

# Save forecast
forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'forecast_next_6_months.csv'), index=False)
print(f"\n✓ Saved: {OUTPUT_DIR}/forecast_next_6_months.csv")

# Visualize forecast
fig, ax = plt.subplots(figsize=(14, 7))

# Historical data
ax.plot(monthly_totals['period'], monthly_totals['total_undiscounted'], 
        marker='o', linewidth=2, label='Historical Data', color='black')

# Forecasts
ax.plot(forecast_df['period'], forecast_df['forecast_linear'], 
        marker='s', linewidth=2, linestyle='--', label='Linear Forecast', color='#4472C4')
ax.plot(forecast_df['period'], forecast_df['forecast_polynomial'], 
        marker='^', linewidth=2, linestyle='--', label='Polynomial Forecast', color='#70AD47')

ax.axvline(x=last_date, color='red', linestyle='--', linewidth=2, 
           label='Forecast Start', alpha=0.7)
ax.set_title('Invoice Total Forecast (Next 6 Months)', fontsize=14, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Invoice Total ($)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_forecast_visualization.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/04_forecast_visualization.png")
plt.close()

# ================================================================
# SAVE MONTHLY DATA WITH PREDICTIONS
# ================================================================
monthly_totals['pred_linear'] = y_pred_linear_all
monthly_totals['pred_polynomial'] = y_pred_poly_all
monthly_totals['pred_seasonal'] = y_pred_seasonal

monthly_totals.to_csv(os.path.join(OUTPUT_DIR, 'monthly_totals_with_predictions.csv'), 
                      index=False)
print(f"✓ Saved: {OUTPUT_DIR}/monthly_totals_with_predictions.csv")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("REGRESSION ANALYSIS COMPLETE")
print("="*80)
print(f"\nBest performing model (by R²): {results_df.iloc[0]['Model']}")
print(f"  R² Score: {results_df.iloc[0]['R2']:.4f}")
print(f"  MAE: ${results_df.iloc[0]['MAE']:,.2f}")
print(f"  RMSE: ${results_df.iloc[0]['RMSE']:,.2f}")

print("\nAll outputs saved to folder: regression_outputs/")
print("Files created:")
print("  1. regression_results_summary.csv - Model comparison")
print("  2. monthly_totals_with_predictions.csv - Data with predictions")
print("  3. forecast_next_6_months.csv - 6-month forecast")
print("  4. 01_data_exploration.png - Data visualization")
print("  5. 02_model_predictions.png - Model comparison plots")
print("  6. 03_2025_focus_analysis.png - 2025-specific analysis")
print("  7. 04_forecast_visualization.png - Forecast chart")
print("="*80)