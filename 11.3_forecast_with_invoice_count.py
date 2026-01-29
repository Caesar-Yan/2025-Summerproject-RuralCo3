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
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import statsmodels.api as sm
except Exception:
    ARIMA = None
    SARIMAX = None
    ExponentialSmoothing = None
    sm = None

# ========================
# PATH CONFIGURATION
# ========================
BASE_PATH = Path(r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3")

# Input and output paths
INPUT_FILE = BASE_PATH / "visualisations" / "9.4_monthly_totals_Period_4_Entire.csv"
OUTPUT_PATH = BASE_PATH / "visualisations"

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
print(f"Input File: {INPUT_FILE.name}")
print(f"Output Folder: {OUTPUT_PATH}")
print(f"  (Full resolved path: {OUTPUT_PATH.resolve()})")
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

# ----------------------------------------------------------------
# Additional models: Time-series approaches (ARIMA / SARIMA / ExpSmoothing)
# Try each model with and without `n_invoices` as an exogenous feature
# ----------------------------------------------------------------
print("\n  [5/5] Training Time-Series Models (ARIMA / SARIMA / ExpSmoothing)...")
if ARIMA is not None and SARIMAX is not None and ExponentialSmoothing is not None:
    ts_endog = monthly_historical['total_discounted_price']
    exog = monthly_historical[['n_invoices']]

    # ARIMA (no exog)
    try:
        arima_mod = ARIMA(ts_endog, order=(1, 1, 1)).fit()
        pred_arima = arima_mod.predict(start=0, end=len(ts_endog) - 1)
        r2_arima = r2_score(ts_endog, pred_arima)
        mae_arima = mean_absolute_error(ts_endog, pred_arima)
        models_discounted['ARIMA'] = {
            'model': arima_mod,
            'predictions': pred_arima,
            'r2': r2_arima,
            'mae': mae_arima,
            'type': 'arima'
        }
        print(f"    ARIMA: R²={r2_arima:.4f}, MAE=${mae_arima:,.2f}")
    except Exception as e:
        print(f"    ARIMA failed: {e}")

    # ARIMA with exogenous invoice count
    try:
        arima_exog_mod = ARIMA(ts_endog, order=(1, 1, 1), exog=exog).fit()
        pred_arima_exog = arima_exog_mod.predict(start=0, end=len(ts_endog) - 1, exog=exog)
        r2_arima_exog = r2_score(ts_endog, pred_arima_exog)
        mae_arima_exog = mean_absolute_error(ts_endog, pred_arima_exog)
        models_discounted['ARIMA_Exog'] = {
            'model': arima_exog_mod,
            'predictions': pred_arima_exog,
            'r2': r2_arima_exog,
            'mae': mae_arima_exog,
            'type': 'arima_exog'
        }
        print(f"    ARIMA+Exog: R²={r2_arima_exog:.4f}, MAE=${mae_arima_exog:,.2f}")
    except Exception as e:
        print(f"    ARIMA+Exog failed: {e}")

    # SARIMA (no exog)
    try:
        sarima_mod = SARIMAX(ts_endog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                             enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred_sarima = sarima_mod.predict(start=0, end=len(ts_endog) - 1)
        r2_sarima = r2_score(ts_endog, pred_sarima)
        mae_sarima = mean_absolute_error(ts_endog, pred_sarima)
        models_discounted['SARIMA'] = {
            'model': sarima_mod,
            'predictions': pred_sarima,
            'r2': r2_sarima,
            'mae': mae_sarima,
            'type': 'sarima'
        }
        print(f"    SARIMA: R²={r2_sarima:.4f}, MAE=${mae_sarima:,.2f}")
    except Exception as e:
        print(f"    SARIMA failed: {e}")

    # SARIMA with exogenous invoice count
    try:
        sarima_exog_mod = SARIMAX(ts_endog, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred_sarima_exog = sarima_exog_mod.predict(start=0, end=len(ts_endog) - 1, exog=exog)
        r2_sarima_exog = r2_score(ts_endog, pred_sarima_exog)
        mae_sarima_exog = mean_absolute_error(ts_endog, pred_sarima_exog)
        models_discounted['SARIMA_Exog'] = {
            'model': sarima_exog_mod,
            'predictions': pred_sarima_exog,
            'r2': r2_sarima_exog,
            'mae': mae_sarima_exog,
            'type': 'sarima_exog'
        }
        print(f"    SARIMA+Exog: R²={r2_sarima_exog:.4f}, MAE=${mae_sarima_exog:,.2f}")
    except Exception as e:
        print(f"    SARIMA+Exog failed: {e}")

    # Exponential Smoothing (Holt-Winters) - no exog
    try:
        hw_mod = ExponentialSmoothing(ts_endog, trend='add', seasonal='add', seasonal_periods=12).fit()
        # fittedvalues sometimes available; fallback to predict
        if hasattr(hw_mod, 'fittedvalues'):
            pred_hw = hw_mod.fittedvalues
        else:
            pred_hw = hw_mod.predict(start=0, end=len(ts_endog) - 1)
        r2_hw = r2_score(ts_endog, pred_hw)
        mae_hw = mean_absolute_error(ts_endog, pred_hw)
        models_discounted['ExpSmoothing'] = {
            'model': hw_mod,
            'predictions': pred_hw,
            'r2': r2_hw,
            'mae': mae_hw,
            'type': 'exp_smoothing'
        }
        print(f"    ExpSmoothing: R²={r2_hw:.4f}, MAE=${mae_hw:,.2f}")
    except Exception as e:
        print(f"    ExpSmoothing failed: {e}")

    # Exponential Smoothing + InvoiceCount as hybrid: regress on invoice count, smooth residuals
    try:
        lr_exog = LinearRegression()
        X_inv = monthly_historical[['n_invoices']].values
        lr_exog.fit(X_inv, ts_endog)
        pred_lr = lr_exog.predict(X_inv)
        resid = ts_endog - pred_lr
        hw_resid_mod = ExponentialSmoothing(resid, trend='add', seasonal='add', seasonal_periods=12).fit()
        if hasattr(hw_resid_mod, 'fittedvalues'):
            pred_resid = hw_resid_mod.fittedvalues
        else:
            pred_resid = hw_resid_mod.predict(start=0, end=len(resid) - 1)
        combined_pred = pred_lr + pred_resid
        r2_hw_exog = r2_score(ts_endog, combined_pred)
        mae_hw_exog = mean_absolute_error(ts_endog, combined_pred)
        models_discounted['ExpSmoothing_ExogHybrid'] = {
            'model': (lr_exog, hw_resid_mod),
            'predictions': combined_pred,
            'r2': r2_hw_exog,
            'mae': mae_hw_exog,
            'type': 'exp_smoothing_exog_hybrid'
        }
        print(f"    ExpSmoothing+ExogHybrid: R²={r2_hw_exog:.4f}, MAE=${mae_hw_exog:,.2f}")
    except Exception as e:
        print(f"    ExpSmoothing+ExogHybrid failed: {e}")
else:
    print("    statsmodels not available; skipping ARIMA/SARIMA/ExpSmoothing models.")

# Select best model for discounted price
best_model_name_discounted = max(models_discounted.items(), key=lambda x: x[1]['r2'])[0]
best_model_info_discounted = models_discounted[best_model_name_discounted]
print(f"\n  ✓ Selected Best Model (Discounted Price): {best_model_name_discounted} (R²={best_model_info_discounted['r2']:.4f})")

# ----------------------------------------------------------------
# Save model fit metrics for all models to CSV
# ----------------------------------------------------------------
metrics = []
# Invoice count model metrics
for name, info in models_invoice_count.items():
    metrics.append({
        'target': 'invoice_count',
        'model': name,
        'type': info.get('type', ''),
        'r_squared': info.get('r2', np.nan),
        'mae': info.get('mae', np.nan),
        'selected': 'yes' if name == best_model_name_invoice else ''
    })

# Discounted price model metrics
for name, info in models_discounted.items():
    metrics.append({
        'target': 'discounted_price',
        'model': name,
        'type': info.get('type', ''),
        'r_squared': info.get('r2', np.nan),
        'mae': info.get('mae', np.nan),
        'selected': 'yes' if name == best_model_name_discounted else ''
    })

metrics_df = pd.DataFrame(metrics)
metrics_csv = OUTPUT_PATH / '11.3_model_fit_metrics.csv'
try:
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  ✓ Saved model fit metrics: {metrics_csv.name}")
except Exception as e:
    fallback_metrics = Path.cwd() / '11.3_model_fit_metrics.csv'
    try:
        metrics_df.to_csv(fallback_metrics, index=False)
        metrics_csv = fallback_metrics
        print(f"  ⚠️ Could not save to OUTPUT_PATH; saved model fit metrics locally as: {fallback_metrics.name}")
    except Exception as e2:
        print(f"  ⚠️ Failed to save model fit metrics to both OUTPUT_PATH and local cwd: {e}; {e2}")

# ----------------------------------------------------------------
# Create an aesthetic table image (PNG) for discounted_price model metrics
# and save as 11.3_model_fit_metrics_discounted.png for presentation use
# ----------------------------------------------------------------
try:
    disc_metrics = metrics_df[metrics_df['target'] == 'discounted_price'].copy()
    if not disc_metrics.empty:
        disc_metrics = disc_metrics[['model', 'r_squared']].copy()
        disc_metrics['r_squared'] = disc_metrics['r_squared'].round(4)

        # Human-friendly model labels: map components to Polynomial, Seasonal, N_invoices, Linear
        def pretty_label(name: str) -> str:
            n = name or ''
            parts = []
            if 'Poly' in n or 'poly' in n:
                parts.append('Polynomial')
            if 'Seasonal' in n or 'seasonal' in n:
                parts.append('Seasonal')
            if 'Inv' in n or 'inv' in n or 'Exog' in n or 'exog' in n or 'n_invoices' in n:
                parts.append('N_invoices')
            if 'Linear' in n or n.lower().startswith('linear'):
                parts.append('Linear')
            if not parts:
                if n.startswith('ARIMA'):
                    return 'ARIMA'
                if n.startswith('SARIMA'):
                    return 'SARIMA'
                if 'ExpSmoothing' in n or 'Exp' in n:
                    return 'ExpSmoothing'
            return ' + '.join(parts) if parts else n

        disc_metrics['model'] = disc_metrics['model'].apply(pretty_label)
        disc_metrics = disc_metrics.sort_values('r_squared', ascending=False).reset_index(drop=True)

        # Estimate figure width to fit long model labels
        max_model_len = disc_metrics['model'].map(lambda x: len(str(x))).max() if not disc_metrics.empty else 10
        fig_w = max(6, 0.18 * int(max_model_len) + 4)
        fig_h = max(1.5, 0.6 * (len(disc_metrics) + 1))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')
        table = ax.table(cellText=disc_metrics.values,
                         colLabels=disc_metrics.columns.str.replace('_', ' ').str.title(),
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        try:
            table.auto_set_column_width(col=list(range(len(disc_metrics.columns))))
        except Exception:
            table.scale(1.2, 1.4)

        # Slightly increase body row height to avoid snug text
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#2F4F4F')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F7F7F9' if row % 2 == 0 else 'white')
                try:
                    # scale height by ~15%
                    cell.set_height(cell.get_height() * 1.15)
                except Exception:
                    pass

        plt.tight_layout()
        table_png = OUTPUT_PATH / '11.3_model_fit_metrics_discounted.png'
        try:
            plt.savefig(table_png, dpi=200, bbox_inches='tight')
            print(f"  ✓ Saved discounted-model table image: {table_png.name}")
        except Exception:
            fallback = Path.cwd() / '11.3_model_fit_metrics_discounted.png'
            plt.savefig(fallback, dpi=200, bbox_inches='tight')
            print(f"  ⚠️ Could not save to OUTPUT_PATH; saved locally as: {fallback.name}")
        plt.close()
    else:
        print("  ⚠️ No discounted_price model metrics available to render table.")
except Exception as e:
    print(f"  ⚠️ Failed creating discounted metrics table image: {e}")

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

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(monthly_historical['invoice_period'], monthly_historical['total_discounted_price'],
         marker='o', linewidth=2, label='Historical (Discounted)', color='black', alpha=0.7)
ax.plot(forecast_df['invoice_period'], forecast_df['forecast_discounted_price'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast (Discounted)',
         color='#4472C4', markersize=8)
ax.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Forecast Start')
ax.set_title('Monthly Discounted Price - Historical & Forecast', fontsize=14, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Discounted Price ($)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.3_discounted_price_forecast.png'
try:
    plt.savefig(output_viz, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_viz.name}")
except Exception as e:
    fallback_path = Path.cwd() / '11.3_discounted_price_forecast.png'
    try:
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        print(f"  ⚠️ Failed saving to {output_viz}; saved visualization locally as: {fallback_path.name}")
    except Exception as e2:
        print(f"  ⚠️ Failed to save visualization to both OUTPUT_PATH and local cwd: {e}; {e2}")
plt.close()

# Plot 2: Model Fit Quality - Discounted Price
fig, ax = plt.subplots(figsize=(10, 8))
actual = y_discounted
predicted = best_model_info_discounted['predictions']
ax.scatter(actual, predicted, alpha=0.6, s=100, edgecolors='black', linewidths=1)
min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Model Fit (Discounted Price): {best_model_name_discounted}\nR² = {best_model_info_discounted["r2"]:.4f}',
              fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Discounted Price ($)', fontsize=12)
ax.set_ylabel('Predicted Discounted Price ($)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.3_model_fit_discounted_price.png'
try:
    plt.savefig(output_viz, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_viz.name}")
except Exception as e:
    fallback_path = Path.cwd() / '11.3_model_fit_discounted_price.png'
    plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
    print(f"  ⚠️ Saved to local cwd instead: {fallback_path.name}")
plt.close()

# Plot 3: Invoice Count - Historical + Forecast
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(monthly_historical['invoice_period'], monthly_historical['n_invoices'],
         marker='o', linewidth=2, label='Historical', color='black', alpha=0.7)
ax.plot(forecast_df['invoice_period'], forecast_df['forecast_invoice_count'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast',
         color='#FFC000', markersize=8)
ax.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.set_title(f'Monthly Invoice Count\n(Used as Feature in Discounted Price Model)', 
              fontsize=14, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Invoice Count', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.3_invoice_count_forecast.png'
try:
    plt.savefig(output_viz, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_viz.name}")
except Exception as e:
    fallback_path = Path.cwd() / '11.3_invoice_count_forecast.png'
    plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
    print(f"  ⚠️ Saved to local cwd instead: {fallback_path.name}")
plt.close()

# Plot 4: Average Price per Invoice
fig, ax = plt.subplots(figsize=(14, 7))
historical_avg_price = monthly_historical['total_discounted_price'] / monthly_historical['n_invoices']
ax.plot(monthly_historical['invoice_period'], historical_avg_price,
         marker='o', linewidth=2, label='Historical', color='#70AD47', alpha=0.7)
ax.plot(forecast_df['invoice_period'], forecast_df['forecast_avg_price_per_invoice'],
         marker='s', linewidth=2.5, linestyle='--', label='Forecast',
         color='#A9D18E', markersize=8)
ax.axvline(x=last_date, color='red', linestyle='--', linewidth=2, alpha=0.5)
avg_historical_price = historical_avg_price.mean()
ax.axhline(y=avg_historical_price, color='#70AD47', linestyle=':', linewidth=2, alpha=0.5,
            label=f'Historical Avg: ${avg_historical_price:,.2f}')
ax.set_title('Average Price per Invoice', fontsize=14, fontweight='bold')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Avg Price per Invoice ($)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_viz = OUTPUT_PATH / '11.3_average_price_per_invoice.png'
try:
    plt.savefig(output_viz, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_viz.name}")
except Exception as e:
    fallback_path = Path.cwd() / '11.3_average_price_per_invoice.png'
    plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
    print(f"  ⚠️ Saved to local cwd instead: {fallback_path.name}")
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

