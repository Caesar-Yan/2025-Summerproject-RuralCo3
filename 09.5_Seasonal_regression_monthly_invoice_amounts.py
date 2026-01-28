'''
09.5_Seasonal Regression Analysis for Monthly Invoice Amounts

This script performs seasonal regression analysis on monthly credit card invoice totals,
comparing three models: simple linear trend, linear + seasonality, and polynomial + seasonality.

Inputs:
- monthly_totals_Period_4_Entire.csv

Outputs:
- 09.5_regression_comparison_seasonal.png
- 09.5_seasonal_pattern_by_month.png
- 09.5_residuals_seasonal_model.png
- 09.5_year_over_year_comparison.png
- 09.5_regression_results_seasonal.csv
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ================================================================
# Load the monthly totals data
# ================================================================
base_dir = Path("T:/projects/2025/RuralCo/Data provided by RuralCo 20251202/RuralCo3")
visualisations_dir = base_dir / "visualisations"

# Load the entire period monthly totals
monthly_df = pd.read_csv(visualisations_dir / "9.4_monthly_totals_Period_4_Entire.csv")

# Keep only the columns we need
monthly_df = monthly_df[['invoice_period', 'total_undiscounted_price']].copy()

# Parse dates - let pandas infer the format
monthly_df['invoice_period'] = pd.to_datetime(monthly_df['invoice_period'])
monthly_df = monthly_df.sort_values('invoice_period').reset_index(drop=True)

print("Data loaded:")
print(monthly_df)
print(f"\nTotal months: {len(monthly_df)}")
print(f"Date range: {monthly_df['invoice_period'].min()} to {monthly_df['invoice_period'].max()}")

# ================================================================
# Feature Engineering for Seasonality
# ================================================================
# Extract month number (1-12) for seasonal patterns
monthly_df['month'] = monthly_df['invoice_period'].dt.month
monthly_df['month_number'] = range(len(monthly_df))  # Linear time trend (0, 1, 2, ...)
monthly_df['year'] = monthly_df['invoice_period'].dt.year

# Create month dummies for seasonality (exclude one to avoid multicollinearity)
month_dummies = pd.get_dummies(monthly_df['month'], prefix='month', drop_first=True)
monthly_df = pd.concat([monthly_df, month_dummies], axis=1)

# ================================================================
# Model 1: Simple Linear Trend (baseline)
# ================================================================
X_simple = monthly_df[['month_number']].values
y = monthly_df['total_undiscounted_price'].values

model_simple = LinearRegression()
model_simple.fit(X_simple, y)
y_pred_simple = model_simple.predict(X_simple)

r2_simple = r2_score(y, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y, y_pred_simple))

# ================================================================
# Model 2: Linear Trend + Seasonality
# ================================================================
# Features: time trend + month dummies
seasonal_features = ['month_number'] + [col for col in monthly_df.columns if col.startswith('month_')]
X_seasonal = monthly_df[seasonal_features].values

model_seasonal = LinearRegression()
model_seasonal.fit(X_seasonal, y)
y_pred_seasonal = model_seasonal.predict(X_seasonal)

r2_seasonal = r2_score(y, y_pred_seasonal)
rmse_seasonal = np.sqrt(mean_squared_error(y, y_pred_seasonal))

# ================================================================
# Model 3: Polynomial Trend + Seasonality
# ================================================================
monthly_df['month_number_squared'] = monthly_df['month_number'] ** 2
poly_seasonal_features = ['month_number', 'month_number_squared'] + [col for col in monthly_df.columns if col.startswith('month_')]
X_poly_seasonal = monthly_df[poly_seasonal_features].values

model_poly_seasonal = LinearRegression()
model_poly_seasonal.fit(X_poly_seasonal, y)
y_pred_poly_seasonal = model_poly_seasonal.predict(X_poly_seasonal)

r2_poly_seasonal = r2_score(y, y_pred_poly_seasonal)
rmse_poly_seasonal = np.sqrt(mean_squared_error(y, y_pred_poly_seasonal))

# ================================================================
# Print Results
# ================================================================
print("\n" + "="*70)
print("REGRESSION RESULTS COMPARISON")
print("="*70)
print("\nModel 1: Simple Linear Trend")
print(f"  R² Score: {r2_simple:.4f}")
print(f"  RMSE: ${rmse_simple:,.2f}")
print(f"  Monthly trend: ${model_simple.coef_[0]:,.2f}")
print(f"  Intercept: ${model_simple.intercept_:,.2f}")

print("\nModel 2: Linear Trend + Seasonality")
print(f"  R² Score: {r2_seasonal:.4f}")
print(f"  RMSE: ${rmse_seasonal:,.2f}")
print(f"  Monthly trend: ${model_seasonal.coef_[0]:,.2f}")
print(f"  Intercept: ${model_seasonal.intercept_:,.2f}")

print("\nModel 3: Polynomial Trend + Seasonality")
print(f"  R² Score: {r2_poly_seasonal:.4f}")
print(f"  RMSE: ${rmse_poly_seasonal:,.2f}")
print(f"  Linear coefficient: ${model_poly_seasonal.coef_[0]:,.2f}")
print(f"  Quadratic coefficient: ${model_poly_seasonal.coef_[1]:,.2f}")
print(f"  Intercept: ${model_poly_seasonal.intercept_:,.2f}")

# ================================================================
# Seasonal Effects Analysis
# ================================================================
print("\n" + "="*70)
print("SEASONAL EFFECTS (Model 2 - Linear + Seasonal)")
print("="*70)

print("\nFeatures used in seasonal model:")
for i, feature in enumerate(seasonal_features):
    print(f"  {i}: {feature} -> ${model_seasonal.coef_[i]:,.2f}")

print("\nMonth coefficients (relative to January baseline):")
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print(f"  January (baseline): $0")

# Get the month dummy column names - exclude 'month_number' and 'month_number_squared'
month_dummy_features = [f for f in seasonal_features if f.startswith('month_') and f not in ['month_number', 'month_number_squared']]

for month_feature in month_dummy_features:
    # Extract month number from feature name (e.g., 'month_2' -> 2)
    month_num = int(month_feature.replace('month_', ''))
    coef = model_seasonal.coef_[seasonal_features.index(month_feature)]
    print(f"  {month_names[month_num-1]:>3}: ${coef:>12,.2f}")

# ================================================================
# Visualizations
# ================================================================

# Plot 1: All three models comparison
fig, ax = plt.subplots(figsize=(16, 9))

ax.scatter(monthly_df['invoice_period'], y, 
           color='#4472C4', s=120, alpha=0.7,
           label='Actual', edgecolor='black', linewidth=1, zorder=5)

ax.plot(monthly_df['invoice_period'], y_pred_simple,
        color='gray', linewidth=2.5, linestyle=':', 
        label=f'Simple Linear (R²={r2_simple:.3f})', alpha=0.7)

ax.plot(monthly_df['invoice_period'], y_pred_seasonal,
        color='orange', linewidth=2.5, linestyle='--',
        label=f'Linear + Seasonal (R²={r2_seasonal:.3f})')

ax.plot(monthly_df['invoice_period'], y_pred_poly_seasonal,
        color='red', linewidth=3,
        label=f'Polynomial + Seasonal (R²={r2_poly_seasonal:.3f})')

ax.set_title('Monthly Undiscounted Price - Model Comparison\nRuralCo Credit Card Invoice Totals', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=13)
ax.set_ylabel('Total Undiscounted Price ($)', fontsize=13)
ax.grid(True, alpha=0.3, linestyle='--')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
ax.legend(fontsize=12, loc='best', framealpha=0.9)

plt.tight_layout()
save_path = visualisations_dir / "09.5_regression_comparison_seasonal.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison plot saved to: {save_path}")

# Plot 2: Seasonal pattern visualization
fig, ax = plt.subplots(figsize=(12, 7))

# Group by month and calculate average
monthly_avg = monthly_df.groupby('month')['total_undiscounted_price'].agg(['mean', 'std', 'count']).reset_index()
monthly_avg['month_name'] = monthly_avg['month'].map(lambda x: month_names[x-1])

bars = ax.bar(monthly_avg['month_name'], monthly_avg['mean'],
              color='#4472C4', alpha=0.7, edgecolor='black', linewidth=1)

# Add error bars if we have multiple years
if (monthly_avg['count'] > 1).any():
    ax.errorbar(monthly_avg['month_name'], monthly_avg['mean'], 
                yerr=monthly_avg['std'], fmt='none', color='black', 
                capsize=5, alpha=0.5, linewidth=1.5)

# Add value labels
for bar, row in zip(bars, monthly_avg.itertuples()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${height/1e6:.1f}M',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Average Monthly Undiscounted Price by Month\nSeasonal Pattern (with std dev)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Average Undiscounted Price ($)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

plt.tight_layout()
save_path = visualisations_dir / "09.5_seasonal_pattern_by_month.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Seasonal pattern plot saved to: {save_path}")

# Plot 3: Residuals for best model (highest R²)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Use the seasonal model residuals
residuals_seasonal = y - y_pred_seasonal

# Residuals over time
ax1.scatter(monthly_df['invoice_period'], residuals_seasonal,
            color='#4472C4', s=100, alpha=0.6, edgecolor='black', linewidth=0.5)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.set_title('Residuals Over Time (Linear + Seasonal)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Residual ($)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

# Residuals histogram
ax2.hist(residuals_seasonal, bins=12, color='#4472C4', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Residual ($)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

plt.tight_layout()
save_path = visualisations_dir / "09.5_residuals_seasonal_model.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Residuals plot saved to: {save_path}")

# Plot 4: Year-over-year comparison
fig, ax = plt.subplots(figsize=(12, 7))

for year in monthly_df['year'].unique():
    year_data = monthly_df[monthly_df['year'] == year].copy()
    year_data = year_data.sort_values('month')
    
    ax.plot(year_data['month'], year_data['total_undiscounted_price'],
            marker='o', linewidth=2, markersize=8, label=str(year), alpha=0.8)

ax.set_title('Year-over-Year Comparison\nMonthly Undiscounted Price', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Total Undiscounted Price ($)', fontsize=12)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names, rotation=45)
ax.grid(True, alpha=0.3, linestyle='--')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
ax.legend(fontsize=11, title='Year')

plt.tight_layout()
save_path = visualisations_dir / "09.5_year_over_year_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Year-over-year comparison saved to: {save_path}")

# ================================================================
# Save detailed results
# ================================================================
results_df = monthly_df[['invoice_period', 'month', 'year', 'month_number', 
                          'total_undiscounted_price']].copy()
results_df['pred_simple'] = y_pred_simple
results_df['pred_seasonal'] = y_pred_seasonal
results_df['pred_poly_seasonal'] = y_pred_poly_seasonal
results_df['residual_simple'] = y - y_pred_simple
results_df['residual_seasonal'] = y - y_pred_seasonal
results_df['residual_poly_seasonal'] = y - y_pred_poly_seasonal

results_path = visualisations_dir / "09.5_regression_results_seasonal.csv"
results_df.to_csv(results_path, index=False)
print(f"✓ Detailed results saved to: {results_path}")

# ================================================================
# Summary Statistics
# ================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nTotal undiscounted price across all months: ${y.sum():,.2f}")
print(f"Average monthly undiscounted price: ${y.mean():,.2f}")
print(f"Std dev: ${y.std():,.2f}")
print(f"Min month: ${y.min():,.2f} ({monthly_df.loc[y.argmin(), 'invoice_period'].strftime('%b %Y')})")
print(f"Max month: ${y.max():,.2f} ({monthly_df.loc[y.argmax(), 'invoice_period'].strftime('%b %Y')})")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  1. 09.5_regression_comparison_seasonal.png")
print("  2. 09.5_seasonal_pattern_by_month.png")
print("  3. 09.5_residuals_seasonal_model.png")
print("  4. 09.5_year_over_year_comparison.png")
print("  5. 09.5_regression_results_seasonal.csv")
print("="*70)