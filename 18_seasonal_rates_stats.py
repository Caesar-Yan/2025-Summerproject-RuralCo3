"""
Script: 18_seasonal_rates_stats.py

Purpose:
    Generate a comprehensive table of seasonal proportionality metrics from scripts 9.5, 9.6, and 9.8.
    Consolidates seasonal coefficients, spending ratios, late payment rates, and adjustment factors.

Inputs:
    - 09.5_regression_results_seasonal.csv (from 9.5 - if available)
    - 09.6_reconstructed_late_payment_rates.csv (from 9.6)
    - 9.8_seasonal_adjustment_summary.csv (from 9.8)
    - 9.4_monthly_totals_Period_4_Entire.csv (for backup seasonal analysis)

Outputs:
    - 18_consolidated_seasonal_metrics.csv
    - 18_seasonal_proportionality_analysis.csv
    - Console output with detailed seasonal analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Define base directories
base_dir = Path("T:/projects/2025/RuralCo")
ruralco3_dir = base_dir / "Data provided by RuralCo 20251202/RuralCo3"
visualisations_dir = ruralco3_dir / "visualisations"
data_cleaning_dir = ruralco3_dir / "data_cleaning"

# Create output directory
visualisations_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SEASONAL PROPORTIONALITY METRICS ANALYSIS")
print("="*80)

# ================================================================
# LOAD DATA FROM SCRIPT 9.6 (Reconstructed Late Payment Rates)
# ================================================================
print(f"\n{'='*80}")
print(f"LOADING DATA FROM SCRIPT 9.6 (RECONSTRUCTED RATES)")
print(f"{'='*80}")

# Load reconstructed late payment rates
rates_96_file = visualisations_dir / "09.6_reconstructed_late_payment_rates.csv"
if rates_96_file.exists():
    rates_96 = pd.read_csv(rates_96_file)
    rates_96['invoice_period'] = pd.to_datetime(rates_96['invoice_period'])
    rates_96['month'] = rates_96['invoice_period'].dt.month
    rates_96['month_name'] = rates_96['invoice_period'].dt.strftime('%b')
    
    print(f"✓ Loaded 9.6 data: {len(rates_96)} months")
    print(f"Date range: {rates_96['invoice_period'].min()} to {rates_96['invoice_period'].max()}")
else:
    print(f"❌ Error: {rates_96_file} not found")
    rates_96 = None

# ================================================================
# LOAD DATA FROM SCRIPT 9.8 (Seasonal Adjustment Summary)
# ================================================================
print(f"\n{'='*80}")
print(f"LOADING DATA FROM SCRIPT 9.8 (ADJUSTMENT FACTORS)")
print(f"{'='*80}")

# Load seasonal adjustment summary
summary_98_file = visualisations_dir / "9.8_seasonal_adjustment_summary.csv"
if summary_98_file.exists():
    summary_98 = pd.read_csv(summary_98_file)
    summary_98['year_month'] = pd.to_datetime(summary_98['year_month'])
    summary_98['month'] = summary_98['year_month'].dt.month
    summary_98['month_name'] = summary_98['year_month'].dt.strftime('%b')
    
    print(f"✓ Loaded 9.8 data: {len(summary_98)} months")
else:
    print(f"❌ Warning: {summary_98_file} not found")
    summary_98 = None

# ================================================================
# RECREATE SCRIPT 9.5 SEASONAL ANALYSIS (if results not saved)
# ================================================================
print(f"\n{'='*80}")
print(f"RECREATING SCRIPT 9.5 SEASONAL ANALYSIS")
print(f"{'='*80}")

# Load monthly totals for seasonal regression
monthly_file = visualisations_dir / "9.4_monthly_totals_Period_4_Entire.csv"
seasonal_coeffs = None

if monthly_file.exists():
    print(f"Loading monthly totals data...")
    monthly_df = pd.read_csv(monthly_file)
    monthly_df['invoice_period'] = pd.to_datetime(monthly_df['invoice_period'])
    monthly_df = monthly_df.sort_values('invoice_period').reset_index(drop=True)
    
    # Feature engineering for seasonality
    monthly_df['month'] = monthly_df['invoice_period'].dt.month
    monthly_df['month_number'] = range(len(monthly_df))
    
    # Create month dummies (exclude January to avoid multicollinearity)
    month_dummies = pd.get_dummies(monthly_df['month'], prefix='month', drop_first=True)
    monthly_df = pd.concat([monthly_df, month_dummies], axis=1)
    
    # Linear + Seasonal regression
    seasonal_features = ['month_number'] + [col for col in monthly_df.columns if col.startswith('month_')]
    X_seasonal = monthly_df[seasonal_features].values
    y = monthly_df['total_undiscounted_price'].values
    
    model_seasonal = LinearRegression()
    model_seasonal.fit(X_seasonal, y)
    
    # Extract seasonal coefficients
    seasonal_coeffs = pd.DataFrame({
        'month': range(1, 13),
        'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    })
    
    # January is baseline (0), others are from model coefficients
    seasonal_coeffs['seasonal_coeff_spending'] = 0.0
    
    # Get only the month dummy features (exclude month_number)
    month_dummy_features = [f for f in seasonal_features if f.startswith('month_') and f != 'month_number']
    
    for i, feature in enumerate(month_dummy_features):
        # Find the coefficient index in the original feature list
        coeff_index = seasonal_features.index(feature)
        month_num = int(feature.replace('month_', ''))
        seasonal_coeffs.loc[seasonal_coeffs['month'] == month_num, 'seasonal_coeff_spending'] = model_seasonal.coef_[coeff_index]
    
    print(f"✓ Calculated seasonal coefficients for {len(seasonal_coeffs)} months")
    
    # Calculate proportional effects
    baseline_spending = monthly_df['total_undiscounted_price'].mean()
    seasonal_coeffs['seasonal_effect_pct'] = (seasonal_coeffs['seasonal_coeff_spending'] / baseline_spending) * 100
    
else:
    print(f"❌ Warning: {monthly_file} not found")

# ================================================================
# CREATE CONSOLIDATED SEASONAL METRICS TABLE
# ================================================================
print(f"\n{'='*80}")
print(f"CREATING CONSOLIDATED SEASONAL METRICS TABLE")
print(f"{'='*80}")

# Start with month framework
months_df = pd.DataFrame({
    'month': range(1, 13),
    'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
})

# Merge 9.5 data (seasonal coefficients)
if seasonal_coeffs is not None:
    months_df = months_df.merge(
        seasonal_coeffs[['month', 'seasonal_coeff_spending', 'seasonal_effect_pct']], 
        on='month', how='left'
    )

# Merge 9.6 data (spending ratios and reconstructed rates)
if rates_96 is not None:
    # Aggregate by month across years
    rates_96_monthly = rates_96.groupby('month').agg({
        'spending_ratio_to_november': 'mean',
        'reconstructed_late_rate_pct': 'mean',
        'spending_amount': 'mean'
    }).reset_index()
    
    months_df = months_df.merge(rates_96_monthly, on='month', how='left')

# Merge 9.8 data (adjustment factors)
if summary_98 is not None:
    # Aggregate by month across years
    summary_98_monthly = summary_98.groupby('month').agg({
        'avg_adjustment_factor': 'mean',
        'seasonal_rate_pct': 'mean',
        'avg_baseline_rate_pct': 'mean',
        'spending_ratio': 'mean'
    }).reset_index()
    
    months_df = months_df.merge(summary_98_monthly, on='month', how='left')

# ================================================================
# CALCULATE ADDITIONAL PROPORTIONALITY METRICS
# ================================================================
print(f"\n{'='*80}")
print(f"CALCULATING PROPORTIONALITY METRICS")
print(f"{'='*80}")

# Calculate relative metrics (November as baseline = 1.0)
november_row = months_df[months_df['month'] == 11].iloc[0] if len(months_df[months_df['month'] == 11]) > 0 else None

if november_row is not None:
    # Spending proportions relative to November
    if 'spending_amount' in months_df.columns:
        nov_spending = november_row['spending_amount']
        months_df['spending_proportion_to_nov'] = months_df['spending_amount'] / nov_spending
    
    # Late rate proportions relative to November
    if 'reconstructed_late_rate_pct' in months_df.columns:
        nov_late_rate = november_row['reconstructed_late_rate_pct']
        months_df['late_rate_proportion_to_nov'] = months_df['reconstructed_late_rate_pct'] / nov_late_rate

# Calculate inverse relationship strength
if 'spending_ratio_to_november' in months_df.columns and 'reconstructed_late_rate_pct' in months_df.columns:
    # Correlation between spending ratio and late rate (should be negative)
    correlation = months_df[['spending_ratio_to_november', 'reconstructed_late_rate_pct']].corr().iloc[0,1]
    print(f"Spending vs Late Rate Correlation: {correlation:.4f}")

# Calculate seasonal volatility
volatility_metrics = {}
for col in ['spending_amount', 'reconstructed_late_rate_pct', 'avg_adjustment_factor']:
    if col in months_df.columns and months_df[col].notna().any():
        volatility_metrics[f'{col}_cv'] = months_df[col].std() / months_df[col].mean()
        volatility_metrics[f'{col}_range'] = months_df[col].max() - months_df[col].min()

# ================================================================
# CREATE DETAILED ANALYSIS TABLE
# ================================================================
print(f"\n{'='*80}")
print(f"CREATING DETAILED SEASONAL ANALYSIS")
print(f"{'='*80}")

# Display the consolidated table
print(f"\n📊 CONSOLIDATED SEASONAL METRICS BY MONTH:")
print("="*120)

# Select key columns for display
display_columns = ['month_name']
if 'seasonal_effect_pct' in months_df.columns:
    display_columns.append('seasonal_effect_pct')
if 'spending_ratio_to_november' in months_df.columns:
    display_columns.append('spending_ratio_to_november')
if 'reconstructed_late_rate_pct' in months_df.columns:
    display_columns.append('reconstructed_late_rate_pct')
if 'avg_adjustment_factor' in months_df.columns:
    display_columns.append('avg_adjustment_factor')

if len(display_columns) > 1:
    print(months_df[display_columns].to_string(index=False, float_format='%.3f'))

# ================================================================
# GENERATE PROPORTIONALITY INSIGHTS
# ================================================================
print(f"\n{'='*80}")
print(f"PROPORTIONALITY INSIGHTS")
print(f"{'='*80}")

insights = []

# 1. Highest/Lowest spending months
if 'spending_amount' in months_df.columns:
    highest_spending = months_df.loc[months_df['spending_amount'].idxmax()]
    lowest_spending = months_df.loc[months_df['spending_amount'].idxmin()]
    
    insights.append({
        'category': 'Spending Extremes',
        'highest_month': highest_spending['month_name'],
        'highest_value': highest_spending['spending_amount'],
        'lowest_month': lowest_spending['month_name'], 
        'lowest_value': lowest_spending['spending_amount'],
        'ratio': highest_spending['spending_amount'] / lowest_spending['spending_amount']
    })

# 2. Highest/Lowest late payment rate months
if 'reconstructed_late_rate_pct' in months_df.columns:
    highest_late = months_df.loc[months_df['reconstructed_late_rate_pct'].idxmax()]
    lowest_late = months_df.loc[months_df['reconstructed_late_rate_pct'].idxmin()]
    
    insights.append({
        'category': 'Late Rate Extremes',
        'highest_month': highest_late['month_name'],
        'highest_value': highest_late['reconstructed_late_rate_pct'],
        'lowest_month': lowest_late['month_name'],
        'lowest_value': lowest_late['reconstructed_late_rate_pct'],
        'ratio': highest_late['reconstructed_late_rate_pct'] / lowest_late['reconstructed_late_rate_pct']
    })

# 3. Strongest seasonal adjustment factors
if 'avg_adjustment_factor' in months_df.columns:
    strongest_up = months_df.loc[months_df['avg_adjustment_factor'].idxmax()]
    strongest_down = months_df.loc[months_df['avg_adjustment_factor'].idxmin()]
    
    insights.append({
        'category': 'Adjustment Factor Extremes',
        'highest_month': strongest_up['month_name'],
        'highest_value': strongest_up['avg_adjustment_factor'],
        'lowest_month': strongest_down['month_name'],
        'lowest_value': strongest_down['avg_adjustment_factor'],
        'ratio': strongest_up['avg_adjustment_factor'] / strongest_down['avg_adjustment_factor']
    })

# Display insights
for insight in insights:
    print(f"\n🎯 {insight['category']}:")
    print(f"  Highest: {insight['highest_month']} ({insight['highest_value']:.2f})")
    print(f"  Lowest:  {insight['lowest_month']} ({insight['lowest_value']:.2f})")
    print(f"  Ratio:   {insight['ratio']:.2f}x difference")

# ================================================================
# SAVE RESULTS
# ================================================================
print(f"\n{'='*80}")
print(f"SAVING RESULTS")
print(f"{'='*80}")

# Save consolidated metrics
output_file1 = visualisations_dir / '18_consolidated_seasonal_metrics.csv'
months_df.to_csv(output_file1, index=False)
print(f"✓ Saved consolidated metrics: {output_file1}")

# Save proportionality analysis
proportionality_df = pd.DataFrame(insights)
if len(proportionality_df) > 0:
    output_file2 = visualisations_dir / '18_seasonal_proportionality_analysis.csv'
    proportionality_df.to_csv(output_file2, index=False)
    print(f"✓ Saved proportionality analysis: {output_file2}")

# Save volatility metrics
volatility_df = pd.DataFrame([volatility_metrics])
if len(volatility_df.columns) > 0:
    output_file3 = visualisations_dir / '18_seasonal_volatility_metrics.csv'
    volatility_df.to_csv(output_file3, index=False)
    print(f"✓ Saved volatility metrics: {output_file3}")

# ================================================================
# SUMMARY STATISTICS
# ================================================================
print(f"\n{'='*80}")
print(f"SUMMARY STATISTICS")
print(f"{'='*80}")

print(f"📊 Data Sources Successfully Loaded:")
print(f"  • Script 9.5 (Seasonal Coefficients): {'✓' if seasonal_coeffs is not None else '❌'}")
print(f"  • Script 9.6 (Reconstructed Rates): {'✓' if rates_96 is not None else '❌'}")  
print(f"  • Script 9.8 (Adjustment Factors): {'✓' if summary_98 is not None else '❌'}")

if len(volatility_metrics) > 0:
    print(f"\n📈 Seasonal Volatility:")
    for metric, value in volatility_metrics.items():
        if 'cv' in metric:
            print(f"  • {metric.replace('_cv', '')} Coefficient of Variation: {value:.3f}")

print(f"\n📁 Output Files Generated:")
print(f"  • {output_file1.name}")
if len(proportionality_df) > 0:
    print(f"  • {output_file2.name}")
if len(volatility_df.columns) > 0:
    print(f"  • {output_file3.name}")

print(f"\n{'='*80}")
print(f"SEASONAL PROPORTIONALITY ANALYSIS COMPLETE!")
print(f"{'='*80}")
