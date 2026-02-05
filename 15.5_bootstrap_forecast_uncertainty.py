"""
15.5_bootstrap_forecast_uncertainty

Bootstrap revenue estimation using forecast model uncertainty.

Instead of synthetic statement bundling, this script:
1. Loads 11.3 (discounted) and 11.5 (undiscounted) forecasts
2. Extracts forecast residuals to estimate model uncertainty
3. For each bootstrap iteration:
   - Adds random noise to forecasts (sampled from residual distribution)
   - Allocates forecasted totals to deciles using pct_of_total_value
   - Applies decile payment profiles + seasonal calibration
   - Calculates revenue components (interest on discounted/undiscounted, retained discounts)
4. Aggregates per-iteration results: mean and 95% CI per month
5. Generates cumulative last-12-months plot

Inputs:
-------
- visualisations/11.3_forecast_next_15_months.csv (discounted forecast)
- forecast/11.5_forecast_undiscounted_next_15_months.csv (undiscounted forecast)
- payment_profile/decile_payment_profile.pkl
- visualisations/09.6_reconstructed_late_payment_rates.csv
- visualisations/10.6_calibrated_baseline_late_rate.csv
- forecast/15.2_statement_distribution_summary.csv (decile distribution)

Outputs:
--------
- forecast/15.5_bootstrap_component_summary.csv (mean and CI)
- forecast/15.5_cumulative_revenue_last_12_months.png
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path(r"\\file\Usersc$\cch155\Home\Desktop\2025\data605\2025-Summerproject-RuralCo3")
ALT_BASE = Path(r"T:\projects\2025\RuralCo\Data provided by RuralCo 20251202\RuralCo3")

DATA_CLEAN_DIR = BASE_PATH / 'data_cleaning'
PROFILE_DIR = BASE_PATH / 'payment_profile'
VIS_DIR = BASE_PATH / 'visualisations'
FORECAST_DIR = BASE_PATH / 'forecast'
FORECAST_DIR.mkdir(exist_ok=True)

ALT_VIS = ALT_BASE / 'visualisations'
ALT_FORECAST = ALT_BASE / 'forecast'
ALT_PROFILE = ALT_BASE / 'payment_profile'

ANNUAL_INTEREST_RATE = 0.2395
DAILY_INTEREST_RATE = ANNUAL_INTEREST_RATE / 365.0

def ensure_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce')

def load_file_with_fallback(primary_path, alt_path):
    """Load file from primary path, fall back to alternate if needed."""
    if primary_path.exists():
        return pd.read_csv(primary_path)
    elif alt_path.exists():
        return pd.read_csv(alt_path)
    else:
        raise FileNotFoundError(f'File not found: {primary_path}')

def extract_decile_metrics(decile_profile, calibration_df, n_deciles=10):
    """Extract decile-specific late rates and expected days from payment profile."""
    scaling = 1.0
    if not calibration_df.empty and 'scaling_factor' in calibration_df.columns:
        scaling = float(calibration_df['scaling_factor'].iloc[0])
    
    rows = []
    for d in range(n_deciles):
        key = f'decile_{d}'
        late_rate = 0.02
        expected_days = 60
        
        try:
            dec = decile_profile['deciles'].get(key, {})
            late_rate = float(dec.get('payment_behavior', {}).get('prob_late', late_rate))
            cd_given_late = dec.get('delinquency_distribution', {}).get('cd_given_late')
            if cd_given_late:
                expected_days = 0
                for cd, p in cd_given_late.items():
                    cd_int = int(cd)
                    days = {2: 30, 3: 60, 4: 90, 5: 120, 6: 150, 7: 180, 8: 210, 9: 240}.get(cd_int, 60)
                    expected_days += days * p
                if expected_days == 0:
                    expected_days = 60
        except Exception:
            pass
        
        rows.append({
            'decile': d,
            'calibrated_late_rate': late_rate * scaling,
            'expected_days_overdue': expected_days
        })
    
    return pd.DataFrame(rows)

def estimate_components_from_perturbed_forecast(perturbed_disc, perturbed_undisc, dist_df, decile_metrics_df, seasonal_rates, forecast_df):
    """
    Estimate revenue for TWO SEPARATE SCENARIOS:
    
    Scenario 1 (with_discount): Interest calculated on DISCOUNTED amount only
    Scenario 2 (no_discount): Interest on UNDISCOUNTED amount + Retained discount value
    
    Args:
        perturbed_disc: array of perturbed discounted prices (1 per month)
        perturbed_undisc: array of perturbed undiscounted prices (1 per month)
        dist_df: decile distribution (pct_of_total_value)
        decile_metrics_df: decile payment behaviors
        seasonal_rates: seasonal adjustment factors
        forecast_df: original forecast (for invoice_period dates)
    
    Returns:
        Four arrays: 
        - interest_on_disc (with_discount scenario)
        - interest_on_undisc_plus_retained (no_discount scenario)
        - interest_on_undisc (component of no_discount)
        - retained_discount (component of no_discount)
    """
    n_deciles = decile_metrics_df['decile'].nunique()
    props = (dist_df.sort_values('decile')['pct_of_total_value'].values / 100.0).astype(float)
    if len(props) < n_deciles:
        props = np.pad(props, (0, n_deciles - len(props)), constant_values=0.0)
    
    # Build seasonal map
    seasonal_map = {}
    if not seasonal_rates.empty:
        seasonal_rates['invoice_period'] = ensure_datetime_series(seasonal_rates['invoice_period'])
        for _, r in seasonal_rates.iterrows():
            dt = r['invoice_period']
            if pd.isna(dt):
                continue
            seasonal_map[(dt.year, dt.month)] = float(r.get('reconstructed_late_rate_pct', 0.0)) / 100.0
    baseline = np.mean(list(seasonal_map.values())) if len(seasonal_map) else 0.02
    
    # Output arrays
    monthly_interest_on_disc = []
    monthly_interest_on_undisc = []
    monthly_retained = []
    
    for month_idx, row in forecast_df.iterrows():
        inv_period = ensure_datetime_series(pd.Series([row['invoice_period']])).iloc[0]
        if pd.isna(inv_period):
            ym = (0, 0)
        else:
            ym = (inv_period.year, inv_period.month)
        
        seasonal_factor = seasonal_map.get(ym, baseline) / (baseline if baseline > 0 else 1.0)
        
        # Use perturbed values
        fd = float(perturbed_disc[month_idx]) if month_idx < len(perturbed_disc) else 0.0
        fu = float(perturbed_undisc[month_idx]) if month_idx < len(perturbed_undisc) else 0.0
        fdisc_amount = max(0.0, fu - fd)
        
        # Scenario 1 (with_discount): Interest on DISCOUNTED amount only
        month_interest_disc = 0.0
        
        # Scenario 2 (no_discount): Interest on UNDISCOUNTED amount + Retained discount
        month_interest_undisc = 0.0
        month_retained = 0.0
        
        for d in range(n_deciles):
            prop = props[d]
            dec_val_disc = fd * prop
            dec_val_undisc = fu * prop
            dec_disc_amt = fdisc_amount * prop
            
            metric = decile_metrics_df[decile_metrics_df['decile'] == d].iloc[0]
            adj_late = min(metric['calibrated_late_rate'] * seasonal_factor, 1.0)
            days = metric['expected_days_overdue']
            
            # Scenario 1: Interest on discounted amount when customer pays late
            late_disc_value = dec_val_disc * adj_late
            interest_disc = late_disc_value * DAILY_INTEREST_RATE * days
            month_interest_disc += interest_disc
            
            # Scenario 2: Interest on undiscounted amount + retained discount when customer pays late
            late_undisc_value = dec_val_undisc * adj_late
            interest_undisc = late_undisc_value * DAILY_INTEREST_RATE * days
            retained = dec_disc_amt * adj_late
            
            month_interest_undisc += interest_undisc
            month_retained += retained
        
        monthly_interest_on_disc.append(month_interest_disc)
        monthly_interest_on_undisc.append(month_interest_undisc)
        monthly_retained.append(month_retained)
    
    return (np.array(monthly_interest_on_disc), 
            np.array(monthly_interest_on_undisc) + np.array(monthly_retained),
            np.array(monthly_interest_on_undisc),
            np.array(monthly_retained))

def main():
    debug_runs = __import__('os').getenv('BUNDLE_RUNS_DEBUG')
    if debug_runs and debug_runs.isdigit():
        RUNS = int(debug_runs)
    else:
        RUNS = 100
    
    print(f'\nLoading forecasts for {RUNS} bootstrap iterations...')
    
    # Load forecasts
    f11_3_path = VIS_DIR / '11.3_forecast_next_15_months.csv'
    if not f11_3_path.exists():
        f11_3_path = ALT_VIS / '11.3_forecast_next_15_months.csv'
    f11_3 = pd.read_csv(f11_3_path)
    f11_3['invoice_period'] = ensure_datetime_series(f11_3['invoice_period'])
    
    f11_5_path = FORECAST_DIR / '11.5_forecast_undiscounted_next_15_months.csv'
    if not f11_5_path.exists():
        f11_5_path = ALT_FORECAST / '11.5_forecast_undiscounted_next_15_months.csv'
    f11_5 = pd.read_csv(f11_5_path)
    f11_5['invoice_period'] = ensure_datetime_series(f11_5['invoice_period'])
    
    # Merge forecasts
    forecast_df = f11_3[['invoice_period', 'forecast_discounted_price']].merge(
        f11_5[['invoice_period', 'forecast_undiscounted_price']],
        on='invoice_period',
        how='inner'
    )
    
    if forecast_df.empty:
        raise SystemExit('No matching forecast periods between 11.3 and 11.5')
    
    forecast_df = forecast_df.sort_values('invoice_period').reset_index(drop=True)
    n_months = len(forecast_df)
    
    print(f'  [OK] Loaded {n_months} months of forecasts')
    print(f'    Discounted range: {forecast_df["forecast_discounted_price"].min():.2f} - {forecast_df["forecast_discounted_price"].max():.2f}')
    print(f'    Undiscounted range: {forecast_df["forecast_undiscounted_price"].min():.2f} - {forecast_df["forecast_undiscounted_price"].max():.2f}')
    
    # Load payment profile and calibration
    profile_path = PROFILE_DIR / 'decile_payment_profile.pkl'
    if not profile_path.exists():
        profile_path = ALT_PROFILE / 'decile_payment_profile.pkl'
    with open(profile_path, 'rb') as f:
        decile_profile = pickle.load(f)
    
    calib_path = VIS_DIR / '10.6_calibrated_baseline_late_rate.csv'
    calibration_df = load_file_with_fallback(calib_path, ALT_VIS / '10.6_calibrated_baseline_late_rate.csv')
    
    seasonal_path = VIS_DIR / '09.6_reconstructed_late_payment_rates.csv'
    seasonal_rates = load_file_with_fallback(seasonal_path, ALT_VIS / '09.6_reconstructed_late_payment_rates.csv')
    
    # Load decile distribution
    dist_path = FORECAST_DIR / '15.2_statement_distribution_summary.csv'
    dist_df = load_file_with_fallback(dist_path, ALT_FORECAST / '15.2_statement_distribution_summary.csv')
    
    n_deciles = decile_profile.get('metadata', {}).get('n_deciles', 10)
    decile_metrics_df = extract_decile_metrics(decile_profile, calibration_df, n_deciles)
    
    # Estimate forecast uncertainty from actual vs fitted (use residuals)
    # Simple approach: use std deviation of forecasted values as proxy for uncertainty
    std_disc = forecast_df['forecast_discounted_price'].std()
    std_undisc = forecast_df['forecast_undiscounted_price'].std()
    
    if std_disc == 0:
        std_disc = forecast_df['forecast_discounted_price'].mean() * 0.05  # 5% if no variance
    if std_undisc == 0:
        std_undisc = forecast_df['forecast_undiscounted_price'].mean() * 0.05
    
    print(f'  [OK] Estimated forecast uncertainty:')
    print(f'    Discounted std: ${std_disc:,.2f}')
    print(f'    Undiscounted std: ${std_undisc:,.2f}')
    
    # Bootstrap loop
    arr_interest_disc = np.zeros((RUNS, n_months), dtype=float)
    arr_interest_undisc_plus_retained = np.zeros((RUNS, n_months), dtype=float)
    arr_interest_undisc = np.zeros((RUNS, n_months), dtype=float)
    arr_retained = np.zeros((RUNS, n_months), dtype=float)
    per_iter_rows = []
    
    print(f'\nRunning {RUNS} bootstrap iterations...')
    
    rng = np.random.default_rng(42)
    
    for it in range(RUNS):
        # Generate perturbed forecasts
        noise_disc = rng.normal(0, std_disc, n_months)
        noise_undisc = rng.normal(0, std_undisc, n_months)
        
        perturbed_disc = np.maximum(forecast_df['forecast_discounted_price'].values + noise_disc, 0)
        perturbed_undisc = np.maximum(forecast_df['forecast_undiscounted_price'].values + noise_undisc, 0)
        
        # Estimate components for both scenarios
        interest_disc, interest_undisc_plus_retained, interest_undisc, retained = estimate_components_from_perturbed_forecast(
            perturbed_disc, perturbed_undisc, dist_df, decile_metrics_df, seasonal_rates, forecast_df
        )
        
        arr_interest_disc[it, :] = interest_disc
        arr_interest_undisc_plus_retained[it, :] = interest_undisc_plus_retained
        arr_interest_undisc[it, :] = interest_undisc
        arr_retained[it, :] = retained
        
        # Log per-iteration monthly results (with_discount scenario)
        for j in range(n_months):
            per_iter_rows.append({
                'iteration': it,
                'invoice_period': forecast_df['invoice_period'].iloc[j],
                'scenario': 'with_discount',
                'mean_total': float(interest_disc[j])
            })
        
        # Log per-iteration monthly results (no_discount scenario)
        for j in range(n_months):
            per_iter_rows.append({
                'iteration': it,
                'invoice_period': forecast_df['invoice_period'].iloc[j],
                'scenario': 'no_discount',
                'mean_total': float(interest_undisc_plus_retained[j])
            })
        
        if (it + 1) % max(1, RUNS // 10) == 0:
            print(f'  ... completed {it + 1}/{RUNS}')
    
    # Summary: mean and 95% CI per month for both scenarios
    stats = []
    for j in range(n_months):
        iid = forecast_df['invoice_period'].iloc[j]
        
        # Scenario 1: with_discount (interest on discounted amount only)
        disc_arr = arr_interest_disc[:, j]
        
        # Scenario 2: no_discount (interest on undiscounted + retained discount)
        undisc_arr = arr_interest_undisc_plus_retained[:, j]
        
        stats.append({
            'scenario': 'with_discount',
            'invoice_period': iid,
            'mean_total': float(np.mean(disc_arr)),
            'lower_95_': float(np.percentile(disc_arr, 2.5)),
            'upper_95_': float(np.percentile(disc_arr, 97.5)),
        })
        
        stats.append({
            'scenario': 'no_discount',
            'invoice_period': iid,
            'mean_total': float(np.mean(undisc_arr)),
            'lower_95_': float(np.percentile(undisc_arr, 2.5)),
            'upper_95_': float(np.percentile(undisc_arr, 97.5)),
        })
    
    summary_df = pd.DataFrame(stats)
    out_csv = ALT_FORECAST / '15.5_bootstrap_component_summary.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f'[OK] Saved summary: {out_csv.name}')
    
    # Calculate summary statistics for LAST 12 MONTHS (matching plot)
    last_periods = summary_df['invoice_period'].unique()
    last_periods = sorted(last_periods)[-12:]  # Get last 12 unique periods
    
    summary_last12 = summary_df[summary_df['invoice_period'].isin(last_periods)]
    with_disc = summary_last12[summary_last12['scenario'] == 'with_discount']
    no_disc = summary_last12[summary_last12['scenario'] == 'no_discount']
    
    # Plot cumulative last 12 months (style matching 15.4)
    summary_df['invoice_period'] = ensure_datetime_series(summary_df['invoice_period'])
    
    # Plot both scenarios overlayed
    fig, ax = plt.subplots(figsize=(13, 7))
    
    # Scenario 1: with_discount
    disc_df = summary_df[summary_df['scenario'] == 'with_discount'].sort_values('invoice_period').reset_index(drop=True)
    last_n = min(12, len(disc_df))
    last_disc = disc_df.tail(last_n)
    
    # Calculate cumulative values and confidence intervals
    disc_cumsum_mean = last_disc['mean_total'].cumsum()
    disc_cumsum_lower = last_disc['lower_95_'].cumsum()
    disc_cumsum_upper = last_disc['upper_95_'].cumsum()
    
    # Plot mean line
    ax.plot(last_disc['invoice_period'], disc_cumsum_mean,
            marker='o', linewidth=2.5, markersize=8, color='#4472C4', label='WITH DISCOUNT (Interest on Discounted)')
    
    # Add confidence interval band
    ax.fill_between(last_disc['invoice_period'], disc_cumsum_lower, disc_cumsum_upper,
                    alpha=0.3, color='#4472C4')
    
    # Add area under curve (more transparent)
    ax.fill_between(last_disc['invoice_period'], 0, disc_cumsum_mean,
                    alpha=0.1, color='#4472C4')
    
    # Add final value annotation for with_discount
    final_disc = disc_cumsum_mean.iloc[-1]
    final_disc_lower = disc_cumsum_lower.iloc[-1]
    final_disc_upper = disc_cumsum_upper.iloc[-1]
    
    if final_disc >= 1e6:
        disc_label = f'  ${final_disc/1e6:.2f}M'
    else:
        disc_label = f'  ${final_disc/1e3:.0f}K'
    
    # Format final values with K for <1M, M for >=1M
    def format_revenue(val):
        if val < 1e6:
            return f'${val/1e3:.0f}K'
        else:
            return f'${val/1e6:.2f}M'
    
    ax.text(last_disc['invoice_period'].iloc[-1], final_disc, disc_label,
            fontsize=12, fontweight='bold', color='#4472C4', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#4472C4', linewidth=1))
    
    # Scenario 2: no_discount
    undisc_df = summary_df[summary_df['scenario'] == 'no_discount'].sort_values('invoice_period').reset_index(drop=True)
    last_undisc = undisc_df.tail(last_n)
    
    # Calculate cumulative values and confidence intervals
    undisc_cumsum_mean = last_undisc['mean_total'].cumsum()
    undisc_cumsum_lower = last_undisc['lower_95_'].cumsum()
    undisc_cumsum_upper = last_undisc['upper_95_'].cumsum()
    
    # Plot mean line
    ax.plot(last_undisc['invoice_period'], undisc_cumsum_mean,
            marker='s', linewidth=2.5, markersize=8, color='#70AD47', label='NO DISCOUNT (Interest on Undiscounted + Retained)')
    
    # Add confidence interval band
    ax.fill_between(last_undisc['invoice_period'], undisc_cumsum_lower, undisc_cumsum_upper,
                    alpha=0.3, color='#70AD47')
    
    # Add area under curve (more transparent)
    ax.fill_between(last_undisc['invoice_period'], 0, undisc_cumsum_mean,
                    alpha=0.1, color='#70AD47')
    
    # Add final value annotation for no_discount
    final_undisc = undisc_cumsum_mean.iloc[-1]
    final_undisc_lower = undisc_cumsum_lower.iloc[-1]
    final_undisc_upper = undisc_cumsum_upper.iloc[-1]
    
    if final_undisc >= 1e6:
        undisc_label = f'  ${final_undisc/1e6:.2f}M'
    else:
        undisc_label = f'  ${final_undisc/1e3:.0f}K'
    ax.text(last_undisc['invoice_period'].iloc[-1], final_undisc, undisc_label,
            fontsize=12, fontweight='bold', color='#70AD47', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#70AD47', linewidth=1))
    
    ax.set_title('Forecasted Revenue: Cumulative over Last 12 Months\n(Bootstrap Forecast Uncertainty)', 
                 fontsize=21, fontweight='bold')
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Cumulative Revenue ($)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    
    out_png = ALT_FORECAST / '15.5_cumulative_revenue_last_12_months.png'
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f'[OK] Saved plot: {out_png.name}')
    plt.close()
    
    # Save cumulative confidence intervals CSV
    cumulative_stats = []
    
    # Scenario 1: with_discount cumulative
    for i, row in last_disc.iterrows():
        cumulative_stats.append({
            'scenario': 'with_discount',
            'invoice_period': row['invoice_period'],
            'cumulative_mean': disc_cumsum_mean.loc[i],
            'cumulative_lower_95': disc_cumsum_lower.loc[i],
            'cumulative_upper_95': disc_cumsum_upper.loc[i]
        })
    
    # Scenario 2: no_discount cumulative  
    for i, row in last_undisc.iterrows():
        cumulative_stats.append({
            'scenario': 'no_discount',
            'invoice_period': row['invoice_period'],
            'cumulative_mean': undisc_cumsum_mean.loc[i],
            'cumulative_lower_95': undisc_cumsum_lower.loc[i],
            'cumulative_upper_95': undisc_cumsum_upper.loc[i]
        })
    
    cumulative_df = pd.DataFrame(cumulative_stats)
    cumulative_csv = ALT_FORECAST / '15.5_cumulative_confidence_intervals.csv'
    cumulative_df.to_csv(cumulative_csv, index=False)
    print(f'[OK] Saved cumulative CI: {cumulative_csv.name}')
    
    print(f'[OK] Bootstrap complete: {RUNS} iterations, {n_months} months')
    
    print(f'\n[LAST 12 MONTHS] Revenue Summary:')
    print(f'[SCENARIO 1] WITH DISCOUNT')
    print(f'  Total revenue (mean): ${with_disc["mean_total"].sum():,.2f}')
    print(f'  95% CI: [${with_disc["lower_95_"].sum():,.2f}, ${with_disc["upper_95_"].sum():,.2f}]')
    
    print(f'[SCENARIO 2] NO DISCOUNT')
    print(f'  Total revenue (mean): ${no_disc["mean_total"].sum():,.2f}')
    print(f'  95% CI: [${no_disc["lower_95_"].sum():,.2f}, ${no_disc["upper_95_"].sum():,.2f}]')
    
    if not with_disc.empty:
        ratio = (no_disc["mean_total"].sum() / with_disc["mean_total"].sum()) if with_disc["mean_total"].sum() > 0 else 0
        print(f'\n  Ratio (NO_DISCOUNT / WITH_DISCOUNT): {ratio:.2f}x')

if __name__ == '__main__':
    main()
