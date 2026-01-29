"""
11_Forecast.py - Discount Cancellation Impact Prediction

Objective: Predict revenue changes from discount cancellation
Data Source: datetime_parsed_invoice_line_item_df.csv
Author: RuralCo Analysis Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

# ========================
# PATH CONFIGURATION
# ========================
# IMPORTANT: Update BASE_PATH to match your project location
# Example Windows: r"C:\Users\YourName\Projects\RuralCo3"
# Example Mac/Linux: "/Users/yourname/Projects\RuralCo3"

BASE_PATH = r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3"

# Input and output paths (derived from BASE_PATH)
INPUT_DATA_PATH = os.path.join(BASE_PATH, "data_cleaning", "datetime_parsed_invoice_line_item_df.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "visualisations")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ========================
# DISPLAY PATH INFO
# ========================
print("\n" + "="*70)
print("PATH CONFIGURATION")
print("="*70)
print(f"Base Directory: {BASE_PATH}")
print(f"Input File: {INPUT_DATA_PATH}")
print(f"Output Folder: {OUTPUT_PATH}")
print("="*70)

# Check if input file exists
if not os.path.exists(INPUT_DATA_PATH):
    print(f"\n❌ ERROR: Input file not found!")
    print(f"   Expected: {INPUT_DATA_PATH}")
    print(f"\n   Please update BASE_PATH in the script to match your project location.")
    exit(1)

# ========================
# PLOTTING CONFIGURATION
# ========================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "="*70)
print("🚀 DISCOUNT CANCELLATION IMPACT ANALYSIS")
print("="*70)

# ========================
# FORECAST DATA CLEANING (minimal fix)
# ========================
# These address forecast distortion caused by partial / abnormal months (e.g., 2025-12),
# and missing months (gaps) in the monthly timeline.
MIN_MONTHLY_REVENUE = 100_000          # drop incomplete months
TRAIN_START = "2024-01-01"             # safest training window start
TRAIN_END   = "2025-11-01"             # exclude partial 2025-12


# ========================
# 1. DATA LOADING AND PREPROCESSING
# ========================

def load_invoice_data():
    """Load raw invoice data from CSV file"""
    print("\n📁 [Step 1/7] Loading invoice data...")

    print(f"  Reading: {os.path.basename(INPUT_DATA_PATH)}")
    print(f"  File size: {os.path.getsize(INPUT_DATA_PATH)/1024/1024:.1f} MB")
    print(f"  This may take 30-60 seconds...")

    # Read only necessary columns to save memory and time
    cols_to_read = [
        'invoice_period',           # Invoice date
        'line_gross_amt_derived',   # Gross amount (total)
        'line_net_amt_derived',     # Net amount
        'line_discount_derived',    # Discount amount
        'invoice_id',               # Invoice identifier
    ]

    df = pd.read_csv(INPUT_DATA_PATH, usecols=cols_to_read)

    print(f"  ✓ Read complete: {len(df):,} records")

    # Convert dates to datetime format
    df['invoice_period'] = pd.to_datetime(df['invoice_period'])
    df['year_month'] = df['invoice_period'].dt.to_period('M')

    # Fill missing discount values with 0
    df['line_discount_derived'] = df['line_discount_derived'].fillna(0)

    return df


def calculate_monthly_metrics(df):
    """Calculate monthly revenue and discount metrics"""
    print("\n📊 [Step 2/7] Calculating monthly metrics...")

    # Aggregate data by month
    monthly = df.groupby('year_month').agg({
        'line_gross_amt_derived': 'sum',     # Actual revenue (with discounts)
        'line_net_amt_derived': 'sum',       # Net revenue
        'line_discount_derived': 'sum',      # Total discount amount
        'invoice_id': 'nunique',             # Number of unique invoices
    }).reset_index()

    # Rename columns for clarity
    monthly.columns = [
        'month',
        'discounted_revenue',      # Revenue with discounts applied
        'net_revenue',             # Net revenue
        'total_discount',          # Total discount given
        'invoice_count'            # Count of invoices
    ]

    # Calculate what revenue would be without discounts
    monthly['undiscounted_revenue'] = monthly['discounted_revenue'] + monthly['total_discount']

    # Calculate discount rate as percentage
    monthly['discount_rate'] = (monthly['total_discount'] / monthly['undiscounted_revenue'] * 100).round(2)

    # Convert period back to timestamp
    monthly['month'] = monthly['month'].dt.to_timestamp()

    # Filter out anomalous data (e.g., year 1970)
    monthly = monthly[monthly['month'].dt.year >= 2023].copy()

    # Sort by date
    monthly = monthly.sort_values('month').reset_index(drop=True)

    # ------------------------
    # MINIMAL FIX: clean training months (drop anomalies + handle gaps)
    # ------------------------
    monthly_clean = monthly.copy()

    # Enforce recommended training window (exclude partial 2025-12 etc.)
    monthly_clean = monthly_clean[
        (monthly_clean["month"] >= pd.to_datetime(TRAIN_START)) &
        (monthly_clean["month"] <= pd.to_datetime(TRAIN_END))
    ].copy()

    # Drop incomplete/abnormal months by revenue threshold
    monthly_clean = monthly_clean[monthly_clean["undiscounted_revenue"] >= MIN_MONTHLY_REVENUE].copy()

    # Ensure continuous monthly timeline; fill missing months by interpolation
    # (If using 2024-01 ~ 2025-11, this is typically already continuous, but safe to keep.)
    all_months = pd.date_range(monthly_clean["month"].min(), monthly_clean["month"].max(), freq="MS")
    monthly_clean = (
        monthly_clean.set_index("month")
        .reindex(all_months)
        .rename_axis("month")
        .reset_index()
    )

    # Interpolate numeric columns for any missing months created by reindex
    num_cols = [
        "discounted_revenue", "net_revenue", "total_discount",
        "undiscounted_revenue", "discount_rate", "invoice_count"
    ]
    for c in num_cols:
        if c in monthly_clean.columns:
            monthly_clean[c] = pd.to_numeric(monthly_clean[c], errors="coerce")

    monthly_clean["invoice_count"] = monthly_clean["invoice_count"].interpolate(limit_direction="both").round()
    monthly_clean["discounted_revenue"] = monthly_clean["discounted_revenue"].interpolate(limit_direction="both")
    monthly_clean["net_revenue"] = monthly_clean["net_revenue"].interpolate(limit_direction="both")
    monthly_clean["total_discount"] = monthly_clean["total_discount"].interpolate(limit_direction="both")
    monthly_clean["undiscounted_revenue"] = monthly_clean["undiscounted_revenue"].interpolate(limit_direction="both")

    # Recompute discount_rate safely after interpolation
    monthly_clean["undiscounted_revenue"] = monthly_clean["undiscounted_revenue"].clip(lower=1e-6)
    monthly_clean["discount_rate"] = (
        monthly_clean["total_discount"] / monthly_clean["undiscounted_revenue"] * 100
    ).round(2)

    # Display summary statistics (use cleaned)
    print(f"  ✓ Calculation complete: {len(monthly_clean)} months of data (CLEANED for forecast)")
    print(f"  Date range: {monthly_clean['month'].min().date()} to {monthly_clean['month'].max().date()}")
    print(f"\n  📈 Overall Statistics (Cleaned):")
    print(f"    Total Actual Revenue: ${monthly_clean['discounted_revenue'].sum():,.2f}")
    print(f"    Total Discount Given: ${monthly_clean['total_discount'].sum():,.2f}")
    print(f"    Potential Revenue (No Discount): ${monthly_clean['undiscounted_revenue'].sum():,.2f}")
    print(f"    Average Discount Rate: {monthly_clean['discount_rate'].mean():.2f}%")

    return monthly_clean


def add_time_features(df):
    """Add time-based features for modeling"""
    df = df.copy()
    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['quarter'] = df['month'].dt.quarter
    df['time_index'] = range(len(df))

    # Cyclical features using sine/cosine transformation
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    return df


# ========================
# 2. FORECASTING MODELS
# ========================

def build_forecast_models(df, target_col='undiscounted_revenue'):
    """Build multiple forecasting models and select the best one"""
    print(f"\n🔧 [Step 3/7] Building forecast models (target: {target_col})...")

    models = {}
    X = df[['time_index']].values
    y = df[target_col].values

    # Model 1: Simple Linear OLS
    print("  [1/4] Simple Linear OLS...")
    model_linear = LinearRegression()
    model_linear.fit(X, y)
    pred_linear = model_linear.predict(X)
    r2_linear = r2_score(y, pred_linear)
    models['Linear'] = {
        'model': model_linear,
        'predictions': pred_linear,
        'r2': r2_linear,
        'type': 'linear'
    }
    print(f"    R² = {r2_linear:.4f}")

    # Model 2: Polynomial Regression (degree 2)
    print("  [2/4] Polynomial Regression (degree=2)...")
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)
    pred_poly = model_poly.predict(X_poly)
    r2_poly = r2_score(y, pred_poly)
    models['Polynomial'] = {
        'model': model_poly,
        'poly': poly,
        'predictions': pred_poly,
        'r2': r2_poly,
        'type': 'polynomial'
    }
    print(f"    R² = {r2_poly:.4f}")

    # Model 3: Seasonal Model
    print("  [3/4] Seasonal Model...")
    X_seasonal = df[['time_index', 'month_sin', 'month_cos']].values
    model_seasonal = LinearRegression()
    model_seasonal.fit(X_seasonal, y)
    pred_seasonal = model_seasonal.predict(X_seasonal)
    r2_seasonal = r2_score(y, pred_seasonal)
    models['Seasonal'] = {
        'model': model_seasonal,
        'predictions': pred_seasonal,
        'r2': r2_seasonal,
        'type': 'seasonal'
    }
    print(f"    R² = {r2_seasonal:.4f}")

    # Model 4: Trend + Seasonal
    print("  [4/4] Trend + Seasonal Model...")
    X_full = df[['time_index', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']].values
    model_full = LinearRegression()
    model_full.fit(X_full, y)
    pred_full = model_full.predict(X_full)
    r2_full = r2_score(y, pred_full)
    models['Trend_Seasonal'] = {
        'model': model_full,
        'predictions': pred_full,
        'r2': r2_full,
        'type': 'trend_seasonal'
    }
    print(f"    R² = {r2_full:.4f}")

    # Select best model based on R²
    best_model_name = max(models.items(), key=lambda x: x[1]['r2'])[0]
    print(f"\n  ✓ Best model: {best_model_name} (R² = {models[best_model_name]['r2']:.4f})")

    return models, best_model_name


def forecast_future(model_dict, df, periods=12):
    """Generate forecasts for future periods"""
    model = model_dict['model']
    model_type = model_dict['type']

    last_date = df['month'].max()
    last_time_index = df['time_index'].max()

    # Generate future dates
    future_dates = [last_date + relativedelta(months=i) for i in range(1, periods+1)]
    future_months = [d.month for d in future_dates]
    future_quarters = [d.quarter for d in future_dates]
    future_indices = list(range(last_time_index + 1, last_time_index + periods + 1))

    # Generate forecasts based on model type
    if model_type == 'linear':
        X_future = np.array(future_indices).reshape(-1, 1)
        forecasts = model.predict(X_future)

    elif model_type == 'polynomial':
        X_future = np.array(future_indices).reshape(-1, 1)
        X_future_poly = model_dict['poly'].transform(X_future)
        forecasts = model.predict(X_future_poly)

    elif model_type == 'seasonal':
        month_sin = [np.sin(2 * np.pi * m / 12) for m in future_months]
        month_cos = [np.cos(2 * np.pi * m / 12) for m in future_months]
        X_future = np.column_stack([future_indices, month_sin, month_cos])
        forecasts = model.predict(X_future)

    elif model_type == 'trend_seasonal':
        month_sin = [np.sin(2 * np.pi * m / 12) for m in future_months]
        month_cos = [np.cos(2 * np.pi * m / 12) for m in future_months]
        quarter_sin = [np.sin(2 * np.pi * q / 4) for q in future_quarters]
        quarter_cos = [np.cos(2 * np.pi * q / 4) for q in future_quarters]
        X_future = np.column_stack([future_indices, month_sin, month_cos, quarter_sin, quarter_cos])
        forecasts = model.predict(X_future)

    # Ensure non-negative forecasts
    forecasts = [max(0, f) for f in forecasts]

    forecast_df = pd.DataFrame({
        'month': future_dates,
        'forecast': forecasts
    })

    return forecast_df


# ========================
# 3. DISCOUNT SCENARIO ANALYSIS
# ========================

def analyze_discount_scenarios(monthly_df, forecast_undiscounted, best_model_name):
    """Analyze different discount scenarios"""
    print("\n💰 [Step 4/7] Analyzing discount scenarios...")

    # Calculate historical average discount rate
    avg_discount_rate = monthly_df['discount_rate'].mean() / 100

    print(f"  Historical average discount rate: {avg_discount_rate*100:.2f}%")

    # Define scenarios
    scenarios = {
        'current': {
            'name': 'Current State (Keep Discounts)',
            'discount_rate': avg_discount_rate,
            'color': '#E63946'
        },
        'no_discount': {
            'name': 'Full Cancellation (0% Discount)',
            'discount_rate': 0.0,
            'color': '#06D6A0'
        },
        'half_discount': {
            'name': '50% Discount',
            'discount_rate': avg_discount_rate * 0.5,
            'color': '#F1A208'
        },
        'quarter_discount': {
            'name': '25% Discount',
            'discount_rate': avg_discount_rate * 0.25,
            'color': '#118AB2'
        }
    }

    # Calculate forecasts for each scenario
    scenario_results = {}

    for scenario_key, scenario_info in scenarios.items():
        discount_rate = scenario_info['discount_rate']

        # Forecasted undiscounted revenue
        undiscounted = forecast_undiscounted['forecast'].values

        # Apply discount rate to get actual revenue
        discounted = undiscounted * (1 - discount_rate)

        # Discount amount
        discount_amount = undiscounted * discount_rate

        scenario_results[scenario_key] = {
            'name': scenario_info['name'],
            'discount_rate': discount_rate,
            'color': scenario_info['color'],
            'undiscounted_revenue': undiscounted,
            'discounted_revenue': discounted,
            'discount_amount': discount_amount,
            'total_12m_undiscounted': undiscounted.sum(),
            'total_12m_discounted': discounted.sum(),
            'total_12m_discount': discount_amount.sum()
        }

    # Print comparison table
    print(f"\n  📊 Next 12 Months Forecast Comparison (using {best_model_name} model):")
    print(f"  {'-'*78}")
    print(f"  {'Scenario':<35} {'Actual Revenue':<18} {'Discount':<18} {'Rate':<8}")
    print(f"  {'-'*78}")

    for scenario_key, result in scenario_results.items():
        print(f"  {result['name']:<35} ${result['total_12m_discounted']:>15,.0f} ${result['total_12m_discount']:>15,.0f} {result['discount_rate']*100:>6.1f}%")

    # Calculate revenue increase compared to current state
    baseline = scenario_results['current']['total_12m_discounted']
    print(f"\n  💡 Revenue Increase vs. Current State:")
    print(f"  {'-'*78}")

    for scenario_key in ['no_discount', 'quarter_discount', 'half_discount']:
        result = scenario_results[scenario_key]
        increase = result['total_12m_discounted'] - baseline
        increase_pct = (increase / baseline) * 100
        print(f"  {result['name']:<35} +${increase:>15,.0f}  (+{increase_pct:>5.1f}%)")

    return scenario_results, scenarios


# ========================
# 4. VISUALIZATION
# ========================

def plot_historical_analysis(monthly_df, save_path=OUTPUT_PATH):
    """Create historical analysis visualizations"""
    print("\n🎨 [Step 5/7] Creating historical analysis charts...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Chart 1: Revenue Comparison
    ax1 = axes[0, 0]
    ax1.plot(monthly_df['month'], monthly_df['undiscounted_revenue']/1e6,
            'o-', label='Undiscounted Revenue', linewidth=2, markersize=6, color='#06D6A0')
    ax1.plot(monthly_df['month'], monthly_df['discounted_revenue']/1e6,
            's-', label='Actual Revenue (with discount)', linewidth=2, markersize=6, color='#E63946')
    ax1.fill_between(monthly_df['month'],
                     monthly_df['discounted_revenue']/1e6,
                     monthly_df['undiscounted_revenue']/1e6,
                     alpha=0.3, color='orange', label='Discount Amount')
    ax1.set_title('Monthly Revenue Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Revenue (Million $)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Chart 2: Discount Rate Trend
    ax2 = axes[0, 1]
    ax2.plot(monthly_df['month'], monthly_df['discount_rate'],
            'o-', linewidth=2, markersize=6, color='#F1A208')
    ax2.axhline(y=monthly_df['discount_rate'].mean(), color='red',
               linestyle='--', linewidth=2, label=f'Average: {monthly_df["discount_rate"].mean():.1f}%')
    ax2.set_title('Discount Rate Trend', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Discount Rate (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Chart 3: Monthly Discount Amount
    ax3 = axes[1, 0]
    ax3.bar(monthly_df['month'], monthly_df['total_discount']/1e6,
            color='#A23B72', alpha=0.7)
    ax3.set_title('Monthly Discount Amount', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Discount Amount (Million $)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)

    # Chart 4: Cumulative Discount Loss
    ax4 = axes[1, 1]
    cumulative_discount = monthly_df['total_discount'].cumsum() / 1e6
    ax4.plot(monthly_df['month'], cumulative_discount,
            linewidth=3, color='#E63946')
    ax4.fill_between(monthly_df['month'], 0, cumulative_discount,
                     alpha=0.3, color='#E63946')
    ax4.set_title('Cumulative Discount Loss', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Cumulative Discount (Million $)')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    # Add total text
    total_discount = monthly_df['total_discount'].sum() / 1e6
    ax4.text(0.5, 0.95, f'Total: ${total_discount:.2f}M',
            transform=ax4.transAxes, ha='center', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_file = os.path.join(save_path, '11_historical_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"  ✓ Saved: {output_file}")


def plot_forecast_scenarios(monthly_df, forecast_df, scenario_results, save_path=OUTPUT_PATH):
    """Create forecast scenario comparison charts"""
    print("  Creating forecast scenario comparison charts...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Chart 1: Main Forecast
    ax1 = axes[0]

    # Historical data
    ax1.plot(monthly_df['month'], monthly_df['discounted_revenue']/1e6,
            'o-', label='Historical Actual Revenue', linewidth=2, markersize=6,
            color='gray', alpha=0.7)

    # Forecast scenarios
    for scenario_key, result in scenario_results.items():
        ax1.plot(forecast_df['month'], result['discounted_revenue']/1e6,
                's--', label=result['name'], linewidth=2, markersize=7,
                color=result['color'], alpha=0.9)

    # Separator line
    last_date = monthly_df['month'].max()
    ax1.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(last_date, ax1.get_ylim()[1]*0.95, 'Forecast Start →',
            ha='right', va='top', fontsize=11, color='red', fontweight='bold')

    ax1.set_title('Discount Scenario Forecast Comparison - Next 12 Months',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Monthly Revenue (Million $)', fontsize=12)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Chart 2: Cumulative Revenue Increase
    ax2 = axes[1]

    baseline_cumsum = scenario_results['current']['discounted_revenue'].cumsum() / 1e6

    for scenario_key in ['no_discount', 'quarter_discount', 'half_discount']:
        result = scenario_results[scenario_key]
        cumsum = result['discounted_revenue'].cumsum() / 1e6
        increase = cumsum - baseline_cumsum

        ax2.plot(forecast_df['month'], increase,
                linewidth=3, label=result['name'], color=result['color'])
        ax2.fill_between(forecast_df['month'], 0, increase,
                        alpha=0.3, color=result['color'])

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Cumulative Revenue Increase (vs. Current State)',
                  fontsize=15, fontweight='bold', pad=20)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Cumulative Increase (Million $)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_file = os.path.join(save_path, '11_forecast_scenarios.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"  ✓ Saved: {output_file}")


def plot_model_comparison(monthly_df, models, save_path=OUTPUT_PATH):
    """Create model comparison charts"""
    print("  Creating model comparison charts...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    model_names = ['Linear', 'Polynomial', 'Seasonal', 'Trend_Seasonal']

    for idx, model_name in enumerate(model_names):
        ax = axes[idx]

        if model_name in models:
            model = models[model_name]

            # Actual values
            ax.plot(monthly_df['month'], monthly_df['undiscounted_revenue']/1e6,
                   'o-', label='Actual', alpha=0.7, markersize=5, color='steelblue')

            # Fitted values
            ax.plot(monthly_df['month'], model['predictions']/1e6,
                   's-', label='Fitted', alpha=0.8, markersize=6, color='coral')

            ax.set_title(f"{model_name}\nR² = {model['r2']:.4f}",
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Revenue (Million $)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_file = os.path.join(save_path, '11_model_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"  ✓ Saved: {output_file}")


# ========================
# 5. EXPORT RESULTS
# ========================

def export_results(monthly_df, forecast_df, scenario_results, models, best_model_name,
                   save_path=OUTPUT_PATH):
    """Export detailed results to Excel"""
    print("\n💾 [Step 6/7] Exporting results...")

    filename = os.path.join(save_path, '11_forecast_results.xlsx')

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:

        # Sheet 1: Scenario Comparison Summary
        summary_data = []
        for scenario_key, result in scenario_results.items():
            summary_data.append({
                'Scenario': result['name'],
                'Discount_Rate_%': result['discount_rate'] * 100,
                '12M_Actual_Revenue': result['total_12m_discounted'],
                '12M_Discount_Amount': result['total_12m_discount'],
                'Increase_vs_Current': result['total_12m_discounted'] - scenario_results['current']['total_12m_discounted'],
                'Increase_%': ((result['total_12m_discounted'] / scenario_results['current']['total_12m_discounted']) - 1) * 100
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Scenario_Summary', index=False)

        # Sheet 2: Historical Data
        historical = monthly_df[['month', 'discounted_revenue', 'undiscounted_revenue',
                                'total_discount', 'discount_rate', 'invoice_count']].copy()
        historical.columns = ['Month', 'Actual_Revenue', 'Undiscounted_Revenue',
                             'Discount_Amount', 'Discount_Rate_%', 'Invoice_Count']
        historical.to_excel(writer, sheet_name='Historical_Data', index=False)

        # Sheet 3-6: Detailed Forecast for Each Scenario
        for scenario_key, result in scenario_results.items():
            scenario_detail = pd.DataFrame({
                'Month': forecast_df['month'],
                'Undiscounted_Revenue': result['undiscounted_revenue'],
                'Actual_Revenue': result['discounted_revenue'],
                'Discount_Amount': result['discount_amount'],
                'Discount_Rate_%': result['discount_rate'] * 100,
                'Cumulative_Actual_Revenue': result['discounted_revenue'].cumsum(),
                'Cumulative_Discount': result['discount_amount'].cumsum()
            })

            sheet_name = result['name'].replace(' ', '_')[:31]  # Excel limit
            scenario_detail.to_excel(writer, sheet_name=sheet_name, index=False)

        # Sheet 7: Model Performance
        model_performance = []
        for name, model in models.items():
            model_performance.append({
                'Model': name,
                'R_Squared': model['r2'],
                'RMSE': np.sqrt(mean_squared_error(
                    monthly_df['undiscounted_revenue'],
                    model['predictions']
                ))
            })

        perf_df = pd.DataFrame(model_performance)
        perf_df = perf_df.sort_values('R_Squared', ascending=False)
        perf_df.to_excel(writer, sheet_name='Model_Performance', index=False)

    print(f"  ✓ Saved: {filename}")


# ========================
# 6. MAIN EXECUTION
# ========================

def main():
    """Main execution function"""

    # Step 1: Load data
    raw_df = load_invoice_data()

    # Step 2: Calculate monthly metrics (cleaned) + add time features
    monthly_df = calculate_monthly_metrics(raw_df)
    monthly_df = add_time_features(monthly_df)

    # Step 3: Build forecast models
    models, best_model_name = build_forecast_models(monthly_df, target_col='undiscounted_revenue')

    # Step 4: Forecast next 12 months
    forecast_undiscounted = forecast_future(models[best_model_name], monthly_df, periods=12)

    # Step 5: Analyze discount scenarios
    scenario_results, scenarios = analyze_discount_scenarios(
        monthly_df, forecast_undiscounted, best_model_name
    )

    # Step 6: Create visualizations
    plot_historical_analysis(monthly_df)
    plot_model_comparison(monthly_df, models)
    plot_forecast_scenarios(monthly_df, forecast_undiscounted, scenario_results)

    # Step 7: Export results
    export_results(monthly_df, forecast_undiscounted, scenario_results, models, best_model_name)

    # Final summary
    print("\n" + "="*70)
    print("✅ [Step 7/7] Analysis Complete!")
    print("="*70)

    print("\n📊 Key Findings:")
    print(f"  • Total Historical Discount: ${monthly_df['total_discount'].sum():,.2f}")
    print(f"  • Average Discount Rate: {monthly_df['discount_rate'].mean():.2f}%")
    print(f"  • Best Forecast Model: {best_model_name} (R² = {models[best_model_name]['r2']:.4f})")

    print("\n💰 Next 12 Months Forecast:")
    baseline = scenario_results['current']['total_12m_discounted']
    no_disc = scenario_results['no_discount']['total_12m_discounted']
    increase = no_disc - baseline

    print(f"  • Current State Forecast: ${baseline:,.0f}")
    print(f"  • Full Cancellation Forecast: ${no_disc:,.0f}")
    print(f"  • Revenue Increase: ${increase:,.0f} (+{(increase/baseline)*100:.1f}%)")

    print("\n📁 Output Files:")
    print(f"  • 11_historical_analysis.png - Historical analysis charts")
    print(f"  • 11_model_comparison.png - Model comparison charts")
    print(f"  • 11_forecast_scenarios.png - Forecast scenario comparison")
    print(f"  • 11_forecast_results.xlsx - Detailed results data")

    print(f"\n  All files saved to: {OUTPUT_PATH}")

    print("\n" + "="*70)

    return monthly_df, models, scenario_results


if __name__ == "__main__":
    monthly_data, forecast_models, scenarios = main()
