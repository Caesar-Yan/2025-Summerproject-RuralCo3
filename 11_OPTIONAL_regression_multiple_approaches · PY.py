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
print("INVOICE REGRESSION ANALYSIS - MULTIPLE APPROACHES")
print("="*80)
print("\nAnalysis approaches:")
print("  1. Simple OLS regression")
print("  2. Models based on 2025 data only (curve shape changed)")
print("  3. Models using all historical data for comparison")
print("  4. Time series methods")
print("  5. Model BOTH month-by-month AND cumulatively")
print("="*80)

ats_grouped = pd.read_csv('ats_grouped_transformed_with_discounts.csv')
invoice_grouped = pd.read_csv('invoice_grouped_transformed_with_discounts.csv')

# Add customer type identifier
ats_grouped['customer_type'] = 'ATS'
invoice_grouped['customer_type'] = 'Invoice'

# Combine datasets
combined_df = pd.concat([ats_grouped, invoice_grouped], ignore_index=True)
print(f"\nTotal invoices loaded: {len(combined_df):,}")

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

# Create time-based features
monthly_totals['month_index'] = range(len(monthly_totals))
monthly_totals['year'] = monthly_totals['period'].dt.year
monthly_totals['month'] = monthly_totals['period'].dt.month
monthly_totals['quarter'] = monthly_totals['period'].dt.quarter

# Identify 2025+ data
monthly_totals['is_2025_onwards'] = (monthly_totals['year'] >= 2025).astype(int)

# Calculate cumulative totals
monthly_totals['cumulative_undiscounted'] = monthly_totals['total_undiscounted'].cumsum()
monthly_totals['cumulative_discounted'] = monthly_totals['total_discounted'].cumsum()

print(f"Monthly periods: {len(monthly_totals)}")
print(f"Data split: Pre-2025: {(~monthly_totals['is_2025_onwards'].astype(bool)).sum()}, 2025+: {monthly_totals['is_2025_onwards'].sum()}")

# ================================================================
# PART A: MONTHLY MODELS
# ================================================================
print("\n" + "="*80)
print("PART A: MONTHLY MODELS (Month-by-Month)")
print("="*80)

results_summary = []

# Prepare data
X_all = monthly_totals[['month_index']].values
y_monthly_all = monthly_totals['total_undiscounted'].values

# Split 2025+ data
monthly_2025 = monthly_totals[monthly_totals['year'] >= 2025].copy()
monthly_2025['month_index_2025'] = range(len(monthly_2025))

# ================================================================
# Monthly Model 1: Linear Regression (All Data)
# ================================================================
print("\nA1. Linear Regression - Monthly (All Data)")
print("-" * 70)

model_monthly_linear_all = LinearRegression()
model_monthly_linear_all.fit(X_all, y_monthly_all)
pred_monthly_linear_all = model_monthly_linear_all.predict(X_all)

r2_monthly_linear_all = r2_score(y_monthly_all, pred_monthly_linear_all)
mae_monthly_linear_all = mean_absolute_error(y_monthly_all, pred_monthly_linear_all)
rmse_monthly_linear_all = np.sqrt(mean_squared_error(y_monthly_all, pred_monthly_linear_all))

print(f"R² Score: {r2_monthly_linear_all:.4f}")
print(f"MAE: ${mae_monthly_linear_all:,.2f}")
print(f"Monthly growth: ${model_monthly_linear_all.coef_[0]:,.2f}/month")

results_summary.append({
    'Category': 'Monthly',
    'Model': 'Linear (All Data)',
    'R2': r2_monthly_linear_all,
    'MAE': mae_monthly_linear_all,
    'RMSE': rmse_monthly_linear_all,
    'Data_Used': 'All',
    'N_Obs': len(X_all)
})

# ================================================================
# Monthly Model 2: Linear Regression (2025+ Only)
# ================================================================
if len(monthly_2025) >= 3:
    print("\nA2. Linear Regression - Monthly (2025+ Only)")
    print("-" * 70)
    
    X_2025 = monthly_2025[['month_index_2025']].values
    y_monthly_2025 = monthly_2025['total_undiscounted'].values
    
    model_monthly_linear_2025 = LinearRegression()
    model_monthly_linear_2025.fit(X_2025, y_monthly_2025)
    pred_monthly_linear_2025 = model_monthly_linear_2025.predict(X_2025)
    
    r2_monthly_linear_2025 = r2_score(y_monthly_2025, pred_monthly_linear_2025)
    mae_monthly_linear_2025 = mean_absolute_error(y_monthly_2025, pred_monthly_linear_2025)
    rmse_monthly_linear_2025 = np.sqrt(mean_squared_error(y_monthly_2025, pred_monthly_linear_2025))
    
    print(f"R² Score: {r2_monthly_linear_2025:.4f}")
    print(f"MAE: ${mae_monthly_linear_2025:,.2f}")
    print(f"Monthly growth (2025+): ${model_monthly_linear_2025.coef_[0]:,.2f}/month")
    
    results_summary.append({
        'Category': 'Monthly',
        'Model': 'Linear (2025+ Only)',
        'R2': r2_monthly_linear_2025,
        'MAE': mae_monthly_linear_2025,
        'RMSE': rmse_monthly_linear_2025,
        'Data_Used': '2025+',
        'N_Obs': len(X_2025)
    })

# ================================================================
# Monthly Model 3: Polynomial (degree=2, All Data)
# ================================================================
print("\nA3. Polynomial Regression - Monthly (degree=2, All Data)")
print("-" * 70)

poly_features = PolynomialFeatures(degree=2)
X_poly_all = poly_features.fit_transform(X_all)

model_monthly_poly_all = LinearRegression()
model_monthly_poly_all.fit(X_poly_all, y_monthly_all)
pred_monthly_poly_all = model_monthly_poly_all.predict(X_poly_all)

r2_monthly_poly_all = r2_score(y_monthly_all, pred_monthly_poly_all)
mae_monthly_poly_all = mean_absolute_error(y_monthly_all, pred_monthly_poly_all)
rmse_monthly_poly_all = np.sqrt(mean_squared_error(y_monthly_all, pred_monthly_poly_all))

print(f"R² Score: {r2_monthly_poly_all:.4f}")
print(f"MAE: ${mae_monthly_poly_all:,.2f}")

results_summary.append({
    'Category': 'Monthly',
    'Model': 'Polynomial (deg=2)',
    'R2': r2_monthly_poly_all,
    'MAE': mae_monthly_poly_all,
    'RMSE': rmse_monthly_poly_all,
    'Data_Used': 'All',
    'N_Obs': len(X_all)
})

# ================================================================
# PART B: CUMULATIVE MODELS
# ================================================================
print("\n" + "="*80)
print("PART B: CUMULATIVE MODELS")
print("="*80)
print("Modeling running totals instead of individual months")

y_cumulative_all = monthly_totals['cumulative_undiscounted'].values

# ================================================================
# Cumulative Model 1: Linear Regression (All Data)
# ================================================================
print("\nB1. Linear Regression - Cumulative (All Data)")
print("-" * 70)

model_cumulative_linear_all = LinearRegression()
model_cumulative_linear_all.fit(X_all, y_cumulative_all)
pred_cumulative_linear_all = model_cumulative_linear_all.predict(X_all)

r2_cumulative_linear_all = r2_score(y_cumulative_all, pred_cumulative_linear_all)
mae_cumulative_linear_all = mean_absolute_error(y_cumulative_all, pred_cumulative_linear_all)
rmse_cumulative_linear_all = np.sqrt(mean_squared_error(y_cumulative_all, pred_cumulative_linear_all))

print(f"R² Score: {r2_cumulative_linear_all:.4f}")
print(f"MAE: ${mae_cumulative_linear_all:,.2f}")
print(f"Cumulative growth: ${model_cumulative_linear_all.coef_[0]:,.2f}/month")

results_summary.append({
    'Category': 'Cumulative',
    'Model': 'Linear (All Data)',
    'R2': r2_cumulative_linear_all,
    'MAE': mae_cumulative_linear_all,
    'RMSE': rmse_cumulative_linear_all,
    'Data_Used': 'All',
    'N_Obs': len(X_all)
})

# ================================================================
# Cumulative Model 2: Linear Regression (2025+ Only)
# ================================================================
if len(monthly_2025) >= 3:
    print("\nB2. Linear Regression - Cumulative (2025+ Only)")
    print("-" * 70)
    
    # Recalculate cumulative from 2025 start
    monthly_2025_copy = monthly_2025.copy()
    monthly_2025_copy['cumulative_from_2025'] = monthly_2025_copy['total_undiscounted'].cumsum()
    
    y_cumulative_2025 = monthly_2025_copy['cumulative_from_2025'].values
    
    model_cumulative_linear_2025 = LinearRegression()
    model_cumulative_linear_2025.fit(X_2025, y_cumulative_2025)
    pred_cumulative_linear_2025 = model_cumulative_linear_2025.predict(X_2025)
    
    r2_cumulative_linear_2025 = r2_score(y_cumulative_2025, pred_cumulative_linear_2025)
    mae_cumulative_linear_2025 = mean_absolute_error(y_cumulative_2025, pred_cumulative_linear_2025)
    rmse_cumulative_linear_2025 = np.sqrt(mean_squared_error(y_cumulative_2025, pred_cumulative_linear_2025))
    
    print(f"R² Score: {r2_cumulative_linear_2025:.4f}")
    print(f"MAE: ${mae_cumulative_linear_2025:,.2f}")
    print(f"Cumulative growth (2025+): ${model_cumulative_linear_2025.coef_[0]:,.2f}/month")
    
    results_summary.append({
        'Category': 'Cumulative',
        'Model': 'Linear (2025+ Only)',
        'R2': r2_cumulative_linear_2025,
        'MAE': mae_cumulative_linear_2025,
        'RMSE': rmse_cumulative_linear_2025,
        'Data_Used': '2025+',
        'N_Obs': len(X_2025)
    })

# ================================================================
# Cumulative Model 3: Polynomial (degree=2)
# ================================================================
print("\nB3. Polynomial Regression - Cumulative (degree=2)")
print("-" * 70)

model_cumulative_poly_all = LinearRegression()
model_cumulative_poly_all.fit(X_poly_all, y_cumulative_all)
pred_cumulative_poly_all = model_cumulative_poly_all.predict(X_poly_all)

r2_cumulative_poly_all = r2_score(y_cumulative_all, pred_cumulative_poly_all)
mae_cumulative_poly_all = mean_absolute_error(y_cumulative_all, pred_cumulative_poly_all)
rmse_cumulative_poly_all = np.sqrt(mean_squared_error(y_cumulative_all, pred_cumulative_poly_all))

print(f"R² Score: {r2_cumulative_poly_all:.4f}")
print(f"MAE: ${mae_cumulative_poly_all:,.2f}")

results_summary.append({
    'Category': 'Cumulative',
    'Model': 'Polynomial (deg=2)',
    'R2': r2_cumulative_poly_all,
    'MAE': mae_cumulative_poly_all,
    'RMSE': rmse_cumulative_poly_all,
    'Data_Used': 'All',
    'N_Obs': len(X_all)
})

# ================================================================
# PART C: TIME SERIES MODELS
# ================================================================
print("\n" + "="*80)
print("PART C: TIME SERIES MODELS")
print("="*80)

ts_data = monthly_totals.set_index('period')['total_undiscounted']

# ================================================================
# Time Series Model 1: Holt-Winters Exponential Smoothing
# ================================================================
print("\nC1. Holt-Winters Exponential Smoothing")
print("-" * 70)

try:
    model_hw = ExponentialSmoothing(
        ts_data, 
        seasonal_periods=12, 
        trend='add', 
        seasonal='add'
    ).fit()
    
    pred_hw = model_hw.fittedvalues
    
    r2_hw = r2_score(y_monthly_all, pred_hw)
    mae_hw = mean_absolute_error(y_monthly_all, pred_hw)
    rmse_hw = np.sqrt(mean_squared_error(y_monthly_all, pred_hw))
    
    print(f"R² Score: {r2_hw:.4f}")
    print(f"MAE: ${mae_hw:,.2f}")
    
    results_summary.append({
        'Category': 'Time Series',
        'Model': 'Holt-Winters',
        'R2': r2_hw,
        'MAE': mae_hw,
        'RMSE': rmse_hw,
        'Data_Used': 'All',
        'N_Obs': len(y_monthly_all)
    })
except Exception as e:
    print(f"⚠ Holt-Winters failed: {str(e)}")

# ================================================================
# Time Series Model 2: ARIMA
# ================================================================
print("\nC2. ARIMA(1,1,1) Time Series Model")
print("-" * 70)

try:
    model_arima = ARIMA(ts_data, order=(1, 1, 1)).fit()
    pred_arima = model_arima.fittedvalues
    
    y_aligned = y_monthly_all[-len(pred_arima):]
    
    r2_arima = r2_score(y_aligned, pred_arima)
    mae_arima = mean_absolute_error(y_aligned, pred_arima)
    rmse_arima = np.sqrt(mean_squared_error(y_aligned, pred_arima))
    
    print(f"R² Score: {r2_arima:.4f}")
    print(f"MAE: ${mae_arima:,.2f}")
    print(f"AIC: {model_arima.aic:.2f}")
    
    results_summary.append({
        'Category': 'Time Series',
        'Model': 'ARIMA(1,1,1)',
        'R2': r2_arima,
        'MAE': mae_arima,
        'RMSE': rmse_arima,
        'Data_Used': 'All',
        'N_Obs': len(y_aligned)
    })
except Exception as e:
    print(f"⚠ ARIMA failed: {str(e)}")

# ================================================================
# RESULTS SUMMARY
# ================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

results_df = pd.DataFrame(results_summary)
results_df = results_df.sort_values('R2', ascending=False).reset_index(drop=True)

print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, 'regression_results_summary.csv'), index=False)
print(f"\n✓ Saved: {OUTPUT_DIR}/regression_results_summary.csv")

# ================================================================
# VISUALIZATIONS
# ================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(20, 18))

# Row 1: Monthly Models
ax1 = axes[0, 0]
ax1.scatter(monthly_totals['period'], y_monthly_all, alpha=0.6, label='Actual', 
            color='black', s=30)
ax1.plot(monthly_totals['period'], pred_monthly_linear_all, 
         linewidth=2, label=f'Linear All (R²={r2_monthly_linear_all:.3f})', color='#4472C4')
ax1.plot(monthly_totals['period'], pred_monthly_poly_all, 
         linewidth=2, label=f'Poly All (R²={r2_monthly_poly_all:.3f})', color='#70AD47')
if len(monthly_2025) >= 3:
    # Extend 2025 predictions to full timeline for visualization
    pred_2025_extended = np.full(len(monthly_totals), np.nan)
    pred_2025_extended[-len(pred_monthly_linear_2025):] = pred_monthly_linear_2025
    ax1.plot(monthly_totals['period'], pred_2025_extended, 
             linewidth=2, linestyle='--', label=f'Linear 2025+ (R²={r2_monthly_linear_2025:.3f})', 
             color='#FF6B6B')
ax1.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle=':', alpha=0.5, linewidth=2)
ax1.set_title('Monthly Models: Invoice Totals', fontsize=12, fontweight='bold')
ax1.set_ylabel('Monthly Invoice Total ($)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Row 1, Col 2: Monthly Residuals
ax2 = axes[0, 1]
residuals_monthly = y_monthly_all - pred_monthly_linear_all
ax2.scatter(monthly_totals['period'], residuals_monthly, alpha=0.6, color='#4472C4')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle=':', alpha=0.5, linewidth=2)
ax2.set_title('Monthly Linear Model: Residuals', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residual ($)')
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Row 2: Cumulative Models
ax3 = axes[1, 0]
ax3.scatter(monthly_totals['period'], y_cumulative_all, alpha=0.6, label='Actual Cumulative', 
            color='black', s=30)
ax3.plot(monthly_totals['period'], pred_cumulative_linear_all, 
         linewidth=2, label=f'Linear All (R²={r2_cumulative_linear_all:.3f})', color='#4472C4')
ax3.plot(monthly_totals['period'], pred_cumulative_poly_all, 
         linewidth=2, label=f'Poly All (R²={r2_cumulative_poly_all:.3f})', color='#70AD47')
ax3.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle=':', alpha=0.5, linewidth=2)
ax3.set_title('Cumulative Models: Running Totals', fontsize=12, fontweight='bold')
ax3.set_ylabel('Cumulative Invoice Total ($)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Row 2, Col 2: Cumulative Residuals
ax4 = axes[1, 1]
residuals_cumulative = y_cumulative_all - pred_cumulative_linear_all
ax4.scatter(monthly_totals['period'], residuals_cumulative, alpha=0.6, color='#70AD47')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle=':', alpha=0.5, linewidth=2)
ax4.set_title('Cumulative Linear Model: Residuals', fontsize=12, fontweight='bold')
ax4.set_ylabel('Residual ($)')
ax4.grid(True, alpha=0.3)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Row 3: Comparison
ax5 = axes[2, 0]
categories = results_df['Category'].unique()
colors_map = {'Monthly': '#4472C4', 'Cumulative': '#70AD47', 'Time Series': '#FF6B6B'}
for cat in categories:
    cat_data = results_df[results_df['Category'] == cat]
    ax5.scatter(cat_data['Model'], cat_data['R2'], 
               s=200, alpha=0.7, label=cat, color=colors_map.get(cat, 'gray'))
ax5.set_title('Model Performance Comparison (R² Score)', fontsize=12, fontweight='bold')
ax5.set_ylabel('R² Score')
ax5.set_xlabel('Model')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
ax5.tick_params(axis='x', rotation=45)

# Row 3, Col 2: MAE Comparison
ax6 = axes[2, 1]
for cat in categories:
    cat_data = results_df[results_df['Category'] == cat]
    ax6.scatter(cat_data['Model'], cat_data['MAE'], 
               s=200, alpha=0.7, label=cat, color=colors_map.get(cat, 'gray'))
ax6.set_title('Model Performance Comparison (MAE)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Mean Absolute Error ($)')
ax6.set_xlabel('Model')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')
ax6.tick_params(axis='x', rotation=45)
ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Format x-axes for time plots
for ax in [ax1, ax2, ax3, ax4]:
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = os.path.join(OUTPUT_DIR, '02_comprehensive_model_comparison.png')
plt.savefig(output_viz, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_viz}")
plt.close()

# ================================================================
# SAVE PREDICTIONS
# ================================================================
monthly_totals['pred_monthly_linear'] = pred_monthly_linear_all
monthly_totals['pred_monthly_poly'] = pred_monthly_poly_all
monthly_totals['pred_cumulative_linear'] = pred_cumulative_linear_all
monthly_totals['pred_cumulative_poly'] = pred_cumulative_poly_all

monthly_totals.to_csv(os.path.join(OUTPUT_DIR, 'monthly_data_with_predictions.csv'), index=False)
print(f"✓ Saved: {OUTPUT_DIR}/monthly_data_with_predictions.csv")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE - MULTIPLE APPROACHES")
print("="*80)

print("\n✓ Completed analyses:")
print("  [✓] Simple OLS regression")
print("  [✓] Models using 2025+ data only")
print("  [✓] Models using all data for comparison")
print("  [✓] Time series methods (Holt-Winters, ARIMA)")
print("  [✓] Month-by-month modeling")
print("  [✓] Cumulative modeling")

print(f"\nBest performing model overall:")
best_model = results_df.iloc[0]
print(f"  Model: {best_model['Category']} - {best_model['Model']}")
print(f"  R²: {best_model['R2']:.4f}")
print(f"  MAE: ${best_model['MAE']:,.2f}")

print(f"\nBest Monthly model:")
best_monthly = results_df[results_df['Category'] == 'Monthly'].iloc[0]
print(f"  Model: {best_monthly['Model']}")
print(f"  R²: {best_monthly['R2']:.4f}")

print(f"\nBest Cumulative model:")
best_cumulative = results_df[results_df['Category'] == 'Cumulative'].iloc[0]
print(f"  Model: {best_cumulative['Model']}")
print(f"  R²: {best_cumulative['R2']:.4f}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("="*80)
