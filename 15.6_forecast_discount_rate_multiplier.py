"""
15.6_bootstrap_with_discount_rate_forecast

Bootstrap revenue estimation using discount rate forecast from 11.4.

Instead of using 11.5's constant multiplier approach, this script:
1. Loads 11.3 (discounted) and 11.4 (discount_rate) forecasts
2. Calculates undiscounted as: undiscounted = discounted × (1 + discount_rate/100)
3. For each bootstrap iteration:
   - Adds random noise to discounted forecast (from 11.3 residuals)
   - Adds random noise to discount_rate forecast (from 11.4 residuals)
   - Recalculates undiscounted amounts
   - Allocates forecasted totals to deciles using pct_of_total_value
   - Applies decile payment profiles + seasonal calibration
   - Calculates revenue components (interest on discounted/undiscounted, retained discounts)
4. Aggregates per-iteration results: mean and 95% CI per month
5. Generates cumulative last-12-months plot

This approach should give more realistic undiscounted values than 11.5's constant multiplier.

Inputs:
-------
- visualisations/11.3_forecast_next_15_months.csv (discounted forecast)
- forecast/11.4_forecast_discount_rate_15_months.csv (discount rate forecast)
- payment_profile/decile_payment_profile.pkl
- visualisations/09.6_reconstructed_late_payment_rates.csv
- visualisations/10.6_calibrated_baseline_late_rate.csv
- forecast/15.2_statement_distribution_summary.csv (decile distribution)

Outputs:
--------
- forecast/15.6_bootstrap_component_summary.csv (mean and CI)
- forecast/15.6_cumulative_revenue_last_12_months.png
- forecast/15.6_discount_rate_comparison.png (comparing approaches)
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

DATA_CLEAN_DIR = ALT_BASE / 'data_cleaning'
PROFILE_DIR = ALT_BASE / 'payment_profile'
VIS_DIR = ALT_BASE / 'visualisations'
FORECAST_DIR = ALT_BASE / 'forecast'
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
    
    print('\n' + '='*80)
    print('15.6 BOOTSTRAP WITH DISCOUNT RATE FORECAST')
    print('='*80)
    print(f'\nUsing discount rate forecast from 11.4 (not constant multiplier)')
    print(f'Running {RUNS} bootstrap iterations...\n')
    
    # ================================================================
    # STEP 1: Load forecasts
    # ================================================================
    print('='*80)
    print('STEP 1: LOADING FORECASTS')
    print('='*80)
    
    # Load 11.3 discounted forecast
    f11_3_path = VIS_DIR / '11.3_forecast_next_15_months.csv'
    if not f11_3_path.exists():
        f11_3_path = ALT_VIS / '11.3_forecast_next_15_months.csv'
    f11_3 = pd.read_csv(f11_3_path)
    f11_3['invoice_period'] = ensure_datetime_series(f11_3['invoice_period'])
    
    print(f'✓ Loaded 11.3 discounted forecast: {len(f11_3)} months')
    
    # Load 11.4 discount rate forecast
    f11_4_path = FORECAST_DIR / '11.4_forecast_discount_rate_15_months.csv'
    if not f11_4_path.exists():
        f11_4_path = ALT_FORECAST / '11.4_forecast_discount_rate_15_months.csv'
    f11_4 = pd.read_csv(f11_4_path)
    f11_4['invoice_period'] = ensure_datetime_series(f11_4['invoice_period'])
    
    print(f'✓ Loaded 11.4 discount rate forecast: {len(f11_4)} months')
    print(f'  Discount rate range: {f11_4["forecast_discount_rate"].min():.2f}% - {f11_4["forecast_discount_rate"].max():.2f}%')
    print(f'  Discount rate mean: {f11_4["forecast_discount_rate"].mean():.2f}%')
    
    # Merge forecasts
    forecast_df = f11_3[['invoice_period', 'forecast_discounted_price']].merge(
        f11_4[['invoice_period', 'forecast_discount_rate']],
        on='invoice_period',
        how='inner'
    )
    
    if forecast_df.empty:
        raise SystemExit('No matching forecast periods between 11.3 and 11.4')
    
    forecast_df = forecast_df.sort_values('invoice_period').reset_index(drop=True)
    
    # Calculate undiscounted using discount rate
    # undiscounted = discounted × (1 + discount_rate/100)
    # or equivalently: undiscounted = discounted × (undiscounted_as_pct/100)
    forecast_df['forecast_undiscounted_price'] = forecast_df['forecast_discounted_price'] * (
        1 + forecast_df['forecast_discount_rate'] / 100
    )
    
    n_months = len(forecast_df)
    
    print(f'\n✓ Merged forecasts: {n_months} months')
    print(f'\nCalculated undiscounted prices using discount rate:')
    print(f'  Discounted range: ${forecast_df["forecast_discounted_price"].min():,.0f} - ${forecast_df["forecast_discounted_price"].max():,.0f}')
    print(f'  Undiscounted range: ${forecast_df["forecast_undiscounted_price"].min():,.0f} - ${forecast_df["forecast_undiscounted_price"].max():,.0f}')
    print(f'  Implied multiplier range: {(forecast_df["forecast_undiscounted_price"]/forecast_df["forecast_discounted_price"]).min():.3f} - {(forecast_df["forecast_undiscounted_price"]/forecast_df["forecast_discounted_price"]).max():.3f}')
    
    # ================================================================
    # STEP 2: Load payment profile and calibration
    # ================================================================
    print('\n' + '='*80)
    print('STEP 2: LOADING PAYMENT PROFILE & CALIBRATION')
    print('='*80)
    
    profile_path = PROFILE_DIR / 'decile_payment_profile.pkl'
    if not profile_path.exists():
        profile_path = ALT_PROFILE / 'decile_payment_profile.pkl'
    with open(profile_path, 'rb') as f:
        decile_profile = pickle.load(f)
    
    print('✓ Loaded payment profile')
    
    calib_path = VIS_DIR / '10.6_calibrated_baseline_late_rate.csv'
    calibration_df = load_file_with_fallback(calib_path, ALT_VIS / '10.6_calibrated_baseline_late_rate.csv')
    
    print('✓ Loaded calibration data')
    
    seasonal_path = VIS_DIR / '09.6_reconstructed_late_payment_rates.csv'
    seasonal_rates = load_file_with_fallback(seasonal_path, ALT_VIS / '09.6_reconstructed_late_payment_rates.csv')
    
    print('✓ Loaded seasonal adjustment rates')
    
    # Load decile distribution
    dist_path = FORECAST_DIR / '15.2_statement_distribution_summary.csv'
    dist_df = load_file_with_fallback(dist_path, ALT_FORECAST / '15.2_statement_distribution_summary.csv')
    
    print('✓ Loaded decile distribution')
    
    n_deciles = decile_profile.get('metadata', {}).get('n_deciles', 10)
    decile_metrics_df = extract_decile_metrics(decile_profile, calibration_df, n_deciles)
    
    print(f'✓ Extracted {n_deciles} decile metrics')
    
    # ================================================================
    # STEP 3: Estimate forecast uncertainty
    # ================================================================
    print('\n' + '='*80)
    print('STEP 3: ESTIMATING FORECAST UNCERTAINTY')
    print('='*80)
    
    # For discounted: use std deviation as proxy for uncertainty
    std_disc = forecast_df['forecast_discounted_price'].std()
    if std_disc == 0:
        std_disc = forecast_df['forecast_discounted_price'].mean() * 0.05
    
    # For discount rate: use std deviation of forecast
    std_rate = forecast_df['forecast_discount_rate'].std()
    if std_rate == 0:
        std_rate = forecast_df['forecast_discount_rate'].mean() * 0.05
    
    print(f'Estimated forecast uncertainty:')
    print(f'  Discounted price std: ${std_disc:,.0f}')
    print(f'  Discount rate std: {std_rate:.2f} percentage points')
    print(f'\nThis will be used to generate bootstrap perturbations')
    
    # ================================================================
    # STEP 4: Bootstrap loop
    # ================================================================
    print('\n' + '='*80)
    print('STEP 4: RUNNING BOOTSTRAP ITERATIONS')
    print('='*80)
    
    arr_interest_disc = np.zeros((RUNS, n_months), dtype=float)
    arr_interest_undisc_plus_retained = np.zeros((RUNS, n_months), dtype=float)
    arr_interest_undisc = np.zeros((RUNS, n_months), dtype=float)
    arr_retained = np.zeros((RUNS, n_months), dtype=float)
    
    # Also track the perturbed multipliers for comparison
    arr_multipliers = np.zeros((RUNS, n_months), dtype=float)
    
    rng = np.random.default_rng(42)
    
    for it in range(RUNS):
        # Generate perturbed forecasts
        noise_disc = rng.normal(0, std_disc, n_months)
        noise_rate = rng.normal(0, std_rate, n_months)
        
        perturbed_disc = np.maximum(forecast_df['forecast_discounted_price'].values + noise_disc, 0)
        perturbed_rate = np.maximum(forecast_df['forecast_discount_rate'].values + noise_rate, 0)
        
        # Calculate perturbed undiscounted using perturbed rate
        perturbed_undisc = perturbed_disc * (1 + perturbed_rate / 100)
        
        # Store multipliers for analysis
        arr_multipliers[it, :] = perturbed_undisc / np.maximum(perturbed_disc, 1)
        
        # Estimate components for both scenarios
        interest_disc, interest_undisc_plus_retained, interest_undisc, retained = estimate_components_from_perturbed_forecast(
            perturbed_disc, perturbed_undisc, dist_df, decile_metrics_df, seasonal_rates, forecast_df
        )
        
        arr_interest_disc[it, :] = interest_disc
        arr_interest_undisc_plus_retained[it, :] = interest_undisc_plus_retained
        arr_interest_undisc[it, :] = interest_undisc
        arr_retained[it, :] = retained
        
        if (it + 1) % max(1, RUNS // 10) == 0:
            print(f'  ... completed {it + 1}/{RUNS}')
    
    print(f'\n✓ Bootstrap complete: {RUNS} iterations')
    
    # ================================================================
    # STEP 5: Summarize results
    # ================================================================
    print('\n' + '='*80)
    print('STEP 5: SUMMARIZING RESULTS')
    print('='*80)
    
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
    
    # Save summary
    out_csv = ALT_VIS / '15.6_bootstrap_component_summary.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f'✓ Saved summary: {out_csv.name}')
    
    # Save detailed component breakdown with confidence intervals
    detailed_stats = []
    for j in range(n_months):
        iid = forecast_df['invoice_period'].iloc[j]
        
        # Interest on discounted amount
        interest_disc_arr = arr_interest_disc[:, j]
        
        # Interest on undiscounted amount 
        interest_undisc_arr = arr_interest_undisc[:, j]
        
        # Retained discount value
        retained_arr = arr_retained[:, j]
        
        # Combined interest + retained (total for no_discount scenario)
        total_no_discount_arr = arr_interest_undisc_plus_retained[:, j]
        
        detailed_stats.append({
            'invoice_period': iid,
            'component': 'interest_on_discounted',
            'mean_value': float(np.mean(interest_disc_arr)),
            'lower_95': float(np.percentile(interest_disc_arr, 2.5)),
            'upper_95': float(np.percentile(interest_disc_arr, 97.5)),
            'std_dev': float(np.std(interest_disc_arr)),
            'scenario': 'with_discount'
        })
        
        detailed_stats.append({
            'invoice_period': iid,
            'component': 'interest_on_undiscounted',
            'mean_value': float(np.mean(interest_undisc_arr)),
            'lower_95': float(np.percentile(interest_undisc_arr, 2.5)),
            'upper_95': float(np.percentile(interest_undisc_arr, 97.5)),
            'std_dev': float(np.std(interest_undisc_arr)),
            'scenario': 'no_discount'
        })
        
        detailed_stats.append({
            'invoice_period': iid,
            'component': 'retained_discount_value',
            'mean_value': float(np.mean(retained_arr)),
            'lower_95': float(np.percentile(retained_arr, 2.5)),
            'upper_95': float(np.percentile(retained_arr, 97.5)),
            'std_dev': float(np.std(retained_arr)),
            'scenario': 'no_discount'
        })
        
        detailed_stats.append({
            'invoice_period': iid,
            'component': 'total_no_discount_revenue',
            'mean_value': float(np.mean(total_no_discount_arr)),
            'lower_95': float(np.percentile(total_no_discount_arr, 2.5)),
            'upper_95': float(np.percentile(total_no_discount_arr, 97.5)),
            'std_dev': float(np.std(total_no_discount_arr)),
            'scenario': 'no_discount'
        })
    
    detailed_df = pd.DataFrame(detailed_stats)
    detailed_csv = ALT_VIS / '15.6_detailed_component_breakdown.csv'
    detailed_df.to_csv(detailed_csv, index=False)
    print(f'✓ Saved detailed component breakdown: {detailed_csv.name}')
    
    # Create final revenue summary with confidence intervals
    # Calculate cumulative totals for last 12 months
    last_n = min(12, n_months)
    
    # Sum across last 12 months for each bootstrap iteration
    final_interest_disc = arr_interest_disc[:, -last_n:].sum(axis=1)
    final_interest_undisc = arr_interest_undisc[:, -last_n:].sum(axis=1)
    final_retained = arr_retained[:, -last_n:].sum(axis=1)
    final_total_no_discount = arr_interest_undisc_plus_retained[:, -last_n:].sum(axis=1)
    
    final_summary = [
        {
            'scenario': 'WITH_DISCOUNT',
            'component': 'Total Revenue',
            'mean_value': float(np.mean(final_interest_disc)),
            'lower_95': float(np.percentile(final_interest_disc, 2.5)),
            'upper_95': float(np.percentile(final_interest_disc, 97.5)),
            'std_dev': float(np.std(final_interest_disc)),
            'description': 'Interest earned on discounted invoice amounts only'
        },
        {
            'scenario': 'WITH_DISCOUNT',
            'component': 'Interest on Discounted Amount',
            'mean_value': float(np.mean(final_interest_disc)),
            'lower_95': float(np.percentile(final_interest_disc, 2.5)),
            'upper_95': float(np.percentile(final_interest_disc, 97.5)),
            'std_dev': float(np.std(final_interest_disc)),
            'description': 'Interest component for with_discount scenario'
        },
        {
            'scenario': 'NO_DISCOUNT',
            'component': 'Total Revenue',
            'mean_value': float(np.mean(final_total_no_discount)),
            'lower_95': float(np.percentile(final_total_no_discount, 2.5)),
            'upper_95': float(np.percentile(final_total_no_discount, 97.5)),
            'std_dev': float(np.std(final_total_no_discount)),
            'description': 'Combined interest on undiscounted + retained discount value'
        },
        {
            'scenario': 'NO_DISCOUNT',
            'component': 'Interest on Undiscounted Amount',
            'mean_value': float(np.mean(final_interest_undisc)),
            'lower_95': float(np.percentile(final_interest_undisc, 2.5)),
            'upper_95': float(np.percentile(final_interest_undisc, 97.5)),
            'std_dev': float(np.std(final_interest_undisc)),
            'description': 'Interest earned on full undiscounted invoice amounts'
        },
        {
            'scenario': 'NO_DISCOUNT',
            'component': 'Retained Discount Value',
            'mean_value': float(np.mean(final_retained)),
            'lower_95': float(np.percentile(final_retained, 2.5)),
            'upper_95': float(np.percentile(final_retained, 97.5)),
            'std_dev': float(np.std(final_retained)),
            'description': 'Discount amounts retained when customers pay late'
        }
    ]
    
    # Add comparison metrics
    revenue_ratio = np.mean(final_total_no_discount) / np.mean(final_interest_disc)
    final_summary.append({
        'scenario': 'COMPARISON',
        'component': 'Revenue Multiplier',
        'mean_value': float(revenue_ratio),
        'lower_95': np.nan,
        'upper_95': np.nan,
        'std_dev': np.nan,
        'description': f'NO_DISCOUNT advantage over WITH_DISCOUNT: {revenue_ratio:.2f}x'
    })
    
    final_summary_df = pd.DataFrame(final_summary)
    final_summary_csv = ALT_VIS / '15.6_final_revenue_summary_with_ci.csv'
    final_summary_df.to_csv(final_summary_csv, index=False)
    print(f'✓ Saved final revenue summary: {final_summary_csv.name}')
    
    # ================================================================
    # STEP 6: Calculate last 12 months statistics
    # ================================================================
    print('\n' + '='*80)
    print('STEP 6: LAST 12 MONTHS SUMMARY')
    print('='*80)
    
    last_periods = summary_df['invoice_period'].unique()
    last_periods = sorted(last_periods)[-12:]
    
    summary_last12 = summary_df[summary_df['invoice_period'].isin(last_periods)]
    with_disc = summary_last12[summary_last12['scenario'] == 'with_discount']
    no_disc = summary_last12[summary_last12['scenario'] == 'no_discount']
    
    print(f'\n[LAST 12 MONTHS] Revenue Summary:')
    print(f'\n[SCENARIO 1] WITH DISCOUNT (Interest on Discounted Amount)')
    print(f'  Total revenue (mean): ${with_disc["mean_total"].sum():,.2f}')
    print(f'  95% CI: [${with_disc["lower_95_"].sum():,.2f}, ${with_disc["upper_95_"].sum():,.2f}]')
    
    print(f'\n[SCENARIO 2] NO DISCOUNT (Interest on Undiscounted Amount + Retained Discount)')
    print(f'  Total revenue (mean): ${no_disc["mean_total"].sum():,.2f}')
    print(f'  95% CI: [${no_disc["lower_95_"].sum():,.2f}, ${no_disc["upper_95_"].sum():,.2f}]')
    
    if not with_disc.empty and with_disc["mean_total"].sum() > 0:
        ratio = no_disc["mean_total"].sum() / with_disc["mean_total"].sum()
        print(f'\n  Ratio (NO_DISCOUNT / WITH_DISCOUNT): {ratio:.2f}x')
    
    # ================================================================
    # STEP 7: Create visualizations
    # ================================================================
    print('\n' + '='*80)
    print('STEP 7: CREATING VISUALIZATIONS')
    print('='*80)
    
    summary_df['invoice_period'] = ensure_datetime_series(summary_df['invoice_period'])
    
    # ====================
    # Plot 1: Cumulative revenue (last 12 months)
    # ====================
    fig, ax = plt.subplots(figsize=(13, 7))
    
    # Scenario 1: with_discount
    disc_df = summary_df[summary_df['scenario'] == 'with_discount'].sort_values('invoice_period').reset_index(drop=True)
    last_n = min(12, len(disc_df))
    last_disc = disc_df.tail(last_n)
    
    ax.plot(last_disc['invoice_period'], last_disc['mean_total'].cumsum(),
            marker='o', linewidth=2.5, markersize=8, color='#4472C4', label='WITH DISCOUNT (Interest on Discounted)')
    ax.fill_between(last_disc['invoice_period'], 0, last_disc['mean_total'].cumsum(),
                    alpha=0.2, color='#4472C4')
    
    # Add final value annotation for with_discount
    disc_cumsum = last_disc['mean_total'].cumsum()
    final_disc = disc_cumsum.iloc[-1]
    ax.text(last_disc['invoice_period'].iloc[-1], final_disc, f'  ${final_disc/1e3:.0f}K',
            fontsize=10, fontweight='bold', color='#4472C4', va='center')
    
    # Scenario 2: no_discount
    undisc_df = summary_df[summary_df['scenario'] == 'no_discount'].sort_values('invoice_period').reset_index(drop=True)
    last_undisc = undisc_df.tail(last_n)
    
    ax.plot(last_undisc['invoice_period'], last_undisc['mean_total'].cumsum(),
            marker='s', linewidth=2.5, markersize=8, color='#70AD47', label='NO DISCOUNT (Interest on Undiscounted + Retained)')
    ax.fill_between(last_undisc['invoice_period'], 0, last_undisc['mean_total'].cumsum(),
                    alpha=0.2, color='#70AD47')
    
    # Add final value annotation for no_discount
    undisc_cumsum = last_undisc['mean_total'].cumsum()
    final_undisc = undisc_cumsum.iloc[-1]
    ax.text(last_undisc['invoice_period'].iloc[-1], final_undisc, f'  ${final_undisc/1e6:.2f}M',
            fontsize=10, fontweight='bold', color='#70AD47', va='center')
    
    ax.set_title('Forecasted Revenue: Jan 2026 - Jan 2027\n(Using Discount Rate Forecast)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Cumulative Revenue ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    
    # # Add summary statistics box
    # summary_text = (
    #     f'WITH DISCOUNT:\n'
    #     f'  Mean: ${with_disc["mean_total"].sum():,.0f}\n'
    #     f'  95% CI: [${with_disc["lower_95_"].sum():,.0f}, ${with_disc["upper_95_"].sum():,.0f}]\n\n'
    #     f'NO DISCOUNT:\n'
    #     f'  Mean: ${no_disc["mean_total"].sum():,.0f}\n'
    #     f'  95% CI: [${no_disc["lower_95_"].sum():,.0f}, ${no_disc["upper_95_"].sum():,.0f}]\n\n'
    #     f'Ratio: {ratio:.2f}x\n\n'
    #     f'Using discount rate from 11.4\n'
    #     f'(not constant multiplier)'
    # )
    # ax.text(0.98, 0.40, summary_text, transform=ax.transAxes, fontsize=9,
    #         verticalalignment='top', horizontalalignment='right',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    #         family='monospace')
    
    out_png = ALT_VIS / '15.6_cumulative_revenue_last_12_months.png'
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f'✓ Saved plot: {out_png.name}')
    plt.close()
    
    # ====================
    # Plot 1b: Cumulative revenue with confidence intervals (large fonts)
    # ====================
    fig, ax = plt.subplots(figsize=(13, 7))
    
    # Scenario 1: with_discount (with CI)
    disc_df = summary_df[summary_df['scenario'] == 'with_discount'].sort_values('invoice_period').reset_index(drop=True)
    last_n = min(12, len(disc_df))
    last_disc = disc_df.tail(last_n).reset_index(drop=True)
    
    disc_cumsum_mean = last_disc['mean_total'].cumsum()
    disc_cumsum_lower = last_disc['lower_95_'].cumsum()
    disc_cumsum_upper = last_disc['upper_95_'].cumsum()
    
    ax.plot(last_disc['invoice_period'], disc_cumsum_mean,
            marker='o', linewidth=3.5, markersize=11, color='#4472C4', label='WITH DISCOUNT', zorder=3)
    ax.fill_between(last_disc['invoice_period'], disc_cumsum_lower, disc_cumsum_upper,
                    alpha=0.2, color='#4472C4', zorder=1)
    
    # Scenario 2: no_discount (with CI)
    undisc_df = summary_df[summary_df['scenario'] == 'no_discount'].sort_values('invoice_period').reset_index(drop=True)
    last_undisc = undisc_df.tail(last_n).reset_index(drop=True)
    
    undisc_cumsum_mean = last_undisc['mean_total'].cumsum()
    undisc_cumsum_lower = last_undisc['lower_95_'].cumsum()
    undisc_cumsum_upper = last_undisc['upper_95_'].cumsum()
    
    ax.plot(last_undisc['invoice_period'], undisc_cumsum_mean,
            marker='s', linewidth=3.5, markersize=11, color='#70AD47', label='NO DISCOUNT', zorder=3)
    ax.fill_between(last_undisc['invoice_period'], undisc_cumsum_lower, undisc_cumsum_upper,
                    alpha=0.2, color='#70AD47', zorder=1, label='_nolegend_')
    
    # Add final value annotations with revenue figures along the lines
    final_disc = disc_cumsum_mean.iloc[-1]
    final_disc_lower = disc_cumsum_lower.iloc[-1]
    final_disc_upper = disc_cumsum_upper.iloc[-1]
    
    # Format final values with K for <1M, M for >=1M
    def format_revenue(val):
        if val < 1e6:
            return f'${val/1e3:.0f}K'
        else:
            return f'${val/1e6:.2f}M'
    
    def format_ci(val):
        if val < 1e6:
            return f'${val/1e3:.0f}K'
        else:
            return f'${val/1e6:.2f}M'
    
    ax.text(last_disc['invoice_period'].iloc[-1], final_disc, 
            f'  {format_revenue(final_disc)}',
            fontsize=18, fontweight='bold', color='#4472C4', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#4472C4'))
    
    final_undisc = undisc_cumsum_mean.iloc[-1]
    final_undisc_lower = undisc_cumsum_lower.iloc[-1]
    final_undisc_upper = undisc_cumsum_upper.iloc[-1]
    ax.text(last_undisc['invoice_period'].iloc[-1], final_undisc, 
            f'  {format_revenue(final_undisc)}',
            fontsize=18, fontweight='bold', color='#70AD47', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#70AD47'))
    
    ax.set_title('Forecasted Revenue: Jan 2026 - Jan 2027 (with 95% Confidence Intervals)\n(Using Discount Rate Forecast)', 
                 fontsize=21, fontweight='bold')
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Cumulative Revenue ($)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax.legend(fontsize=17, loc='upper left', framealpha=0.95)
    
    out_png_ci = ALT_VIS / '15.6_cumulative_revenue_with_ci_last_12_months.png'
    out_png_ci.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png_ci, dpi=300)
    print(f'✓ Saved plot with CI: {out_png_ci.name}')
    plt.close()
    
    # ====================
    # Plot 2: Multiplier distribution comparison
    # ====================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top panel: Multiplier distribution over iterations (all months)
    ax1 = axes[0]
    
    multiplier_means = arr_multipliers.mean(axis=0)
    multiplier_lower = np.percentile(arr_multipliers, 2.5, axis=0)
    multiplier_upper = np.percentile(arr_multipliers, 97.5, axis=0)
    
    ax1.plot(forecast_df['invoice_period'], multiplier_means,
             marker='o', linewidth=2.5, markersize=8, color='#4472C4', label='Mean Multiplier')
    ax1.fill_between(forecast_df['invoice_period'], multiplier_lower, multiplier_upper,
                     alpha=0.3, color='#4472C4', label='95% CI')
    
    # Add historical average line for reference
    hist_path = VIS_DIR / '9.4_monthly_totals_Period_4_Entire.csv'
    if not hist_path.exists():
        hist_path = ALT_VIS / '9.4_monthly_totals_Period_4_Entire.csv'
    
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        if 'undiscounted_as_pct' in hist_df.columns:
            hist_avg = hist_df['undiscounted_as_pct'].mean() / 100
            ax1.axhline(y=hist_avg, color='red', linestyle='--', linewidth=2, alpha=0.5,
                       label=f'Historical Avg: {hist_avg:.3f}')
    
    ax1.set_title('Undiscounted/Discounted Multiplier\n(Derived from 11.4 Discount Rate Forecast)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Multiplier', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(fontsize=10)
    
    # Bottom panel: Discount rate over time
    ax2 = axes[1]
    
    ax2.plot(forecast_df['invoice_period'], forecast_df['forecast_discount_rate'],
             marker='o', linewidth=2.5, markersize=8, color='#70AD47', label='Forecast Discount Rate')
    
    # Add bootstrap uncertainty
    rate_means = np.zeros(n_months)
    rate_lower = np.zeros(n_months)
    rate_upper = np.zeros(n_months)
    
    for j in range(n_months):
        # Reconstruct discount rate from multipliers
        rates_iter = (arr_multipliers[:, j] - 1) * 100
        rate_means[j] = rates_iter.mean()
        rate_lower[j] = np.percentile(rates_iter, 2.5)
        rate_upper[j] = np.percentile(rates_iter, 97.5)
    
    ax2.fill_between(forecast_df['invoice_period'], rate_lower, rate_upper,
                     alpha=0.3, color='#70AD47', label='95% CI (Bootstrap)')
    
    ax2.set_title('Discount Rate Forecast (from 11.4)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Discount Rate (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    out_png2 = ALT_VIS / '15.6_discount_rate_comparison.png'
    plt.savefig(out_png2, dpi=300)
    print(f'✓ Saved comparison plot: {out_png2.name}')
    plt.close()
    
    # ====================
    # Plot 3: Revenue Component Breakdown (last 12 months)
    # ====================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate mean values for last 12 months
    last_n = min(12, n_months)
    last_months = forecast_df['invoice_period'].tail(last_n).reset_index(drop=True)
    
    interest_disc_means = arr_interest_disc[:, -last_n:].mean(axis=0)
    interest_undisc_means = arr_interest_undisc[:, -last_n:].mean(axis=0)
    retained_means = arr_retained[:, -last_n:].mean(axis=0)
    
    # Cumulative sums for stacking
    interest_disc_cumsum = interest_disc_means.cumsum()
    interest_undisc_cumsum = interest_undisc_means.cumsum()
    retained_cumsum = retained_means.cumsum()
    
    # Create stacked area plot
    ax.fill_between(last_months, 0, interest_disc_cumsum,
                    alpha=0.8, color='#4472C4', label='Interest on Discounted Amount', zorder=1)
    ax.fill_between(last_months, interest_disc_cumsum, interest_disc_cumsum + interest_undisc_cumsum,
                    alpha=0.8, color='#70AD47', label='Interest on Undiscounted Amount', zorder=2)
    ax.fill_between(last_months, interest_disc_cumsum + interest_undisc_cumsum, 
                    interest_disc_cumsum + interest_undisc_cumsum + retained_cumsum,
                    alpha=0.8, color='#FFC000', label='Retained Discount Value', zorder=3)
    
    # Add total line for reference
    total_cumsum = interest_disc_cumsum + interest_undisc_cumsum + retained_cumsum
    ax.plot(last_months, total_cumsum, color='black', linewidth=2, linestyle='--', 
            alpha=0.7, label='Total Revenue (All Components)', zorder=4)
    
    # Format final value labels
    def format_revenue_short(val):
        if val < 1e6:
            return f'${val/1e3:.0f}K'
        else:
            return f'${val/1e6:.2f}M'
    
    # Add component value labels at the end
    final_disc = interest_disc_cumsum[-1]
    final_undisc = interest_undisc_cumsum[-1]
    final_retained = retained_cumsum[-1]
    final_total = total_cumsum[-1]
    
    ax.text(last_months.iloc[-1], final_disc/2, f'{format_revenue_short(final_disc)}',
            fontsize=12, fontweight='bold', color='white', ha='left', va='center')
    ax.text(last_months.iloc[-1], final_disc + final_undisc/2, f'{format_revenue_short(final_undisc)}',
            fontsize=12, fontweight='bold', color='white', ha='left', va='center')
    ax.text(last_months.iloc[-1], final_disc + final_undisc + final_retained/2, f'{format_revenue_short(final_retained)}',
            fontsize=12, fontweight='bold', color='black', ha='left', va='center')
    ax.text(last_months.iloc[-1], final_total, f'  {format_revenue_short(final_total)}',
            fontsize=14, fontweight='bold', color='black', ha='left', va='bottom')
    
    ax.set_title('Cumulative Revenue by Component: Jan 2026 - Jan 2027\n(Using Discount Rate Forecast)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Revenue ($)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
    
    out_png3 = ALT_VIS / '15.6_revenue_component_breakdown.png'
    out_png3.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png3, dpi=300)
    print(f'✓ Saved component breakdown plot: {out_png3.name}')
    plt.close()
    
    # ================================================================
    # REVENUE BREAKDOWN TABLE
    # ================================================================
    print('\n' + '='*80)
    print('📊 FORECAST REVENUE COMPONENT BREAKDOWN TABLE')
    print('='*80)
    
    # Create aesthetic table with actual component breakdowns
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate last 12 months component totals
    last_n = min(12, n_months)
    
    # Get component totals from bootstrap arrays
    total_interest_disc = arr_interest_disc[:, -last_n:].sum(axis=1).mean()
    total_interest_undisc = arr_interest_undisc[:, -last_n:].sum(axis=1).mean()
    total_retained = arr_retained[:, -last_n:].sum(axis=1).mean()
    
    total_with_discount = total_interest_disc
    total_no_discount = total_interest_undisc + total_retained
    
    # Calculate confidence intervals
    total_with_ci_lower = np.percentile(arr_interest_disc[:, -last_n:].sum(axis=1), 2.5)
    total_with_ci_upper = np.percentile(arr_interest_disc[:, -last_n:].sum(axis=1), 97.5)
    
    total_no_ci_lower = np.percentile((arr_interest_undisc[:, -last_n:] + arr_retained[:, -last_n:]).sum(axis=1), 2.5)
    total_no_ci_upper = np.percentile((arr_interest_undisc[:, -last_n:] + arr_retained[:, -last_n:]).sum(axis=1), 97.5)
    
    # Prepare detailed table data
    table_data = [
        ['Component', 'Revenue ($)', '95% CI Range', 'Proportion', 'Description'],
        ['', '', '', '', ''],
        ['WITH DISCOUNT SCENARIO:', '', '', '', ''],
        ['Interest on Discounted Amounts', f'${total_interest_disc:,.0f}', f'${total_with_ci_lower:,.0f} - ${total_with_ci_upper:,.0f}', '100%', 'Interest earned on discounted invoice amounts only'],
        ['', '', '', '', ''],
        ['NO DISCOUNT SCENARIO BREAKDOWN:', '', '', '', ''],
        ['Interest on Undiscounted Amounts', f'${total_interest_undisc:,.0f}', '', f'{(total_interest_undisc/total_no_discount)*100:.1f}%', 'Interest earned on full undiscounted amounts'],
        ['Retained Discount Value', f'${total_retained:,.0f}', '', f'{(total_retained/total_no_discount)*100:.1f}%', 'Discount amounts retained when unpaid'],
        ['Total NO DISCOUNT Revenue', f'${total_no_discount:,.0f}', f'${total_no_ci_lower:,.0f} - ${total_no_ci_upper:,.0f}', '100%', 'Combined interest + retained discounts'],
        ['', '', '', '', ''],
        ['FORECAST METHODOLOGY:', '', '', '', ''],
        ['Discount Rate Source', '11.4 Forecast', '', '', 'Time-varying discount rates (not constant)'],
        ['Bootstrap Iterations', f'{RUNS}', '', '', 'Monte Carlo uncertainty quantification'],
        ['Forecast Period', 'Jan 2026 - Jan 2027', '', '', 'Last 12 months of 15-month forecast'],
        ['', '', '', '', ''],
        ['COMPARISON METRICS:', '', '', '', ''],
        ['Revenue Multiplier', f'{total_no_discount/total_with_discount:.2f}x', '', f'+{((total_no_discount/total_with_discount-1)*100):.1f}%', 'NO DISCOUNT advantage over WITH DISCOUNT'],
        ['Average Discount Rate', f'{forecast_df["forecast_discount_rate"].mean():.1f}%', f'{forecast_df["forecast_discount_rate"].min():.1f}% - {forecast_df["forecast_discount_rate"].max():.1f}%', '', 'Forecasted discount rate range'],
        ['Implied Undiscounted Multiplier', f'{multiplier_means.mean():.3f}', f'{multiplier_means.min():.3f} - {multiplier_means.max():.3f}', '', 'Undiscounted/Discounted ratio'],
    ]
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            cell_obj = table[(i, j)]
            
            if i == 0:  # Header row
                cell_obj.set_facecolor('#4472C4')
                cell_obj.set_text_props(weight='bold', color='white')
            elif 'WITH DISCOUNT SCENARIO' in str(cell):  # Section headers
                cell_obj.set_facecolor('#E8F4FD')
                cell_obj.set_text_props(weight='bold', color='#4472C4')
            elif 'Interest on Discounted' in str(cell):  # WITH DISCOUNT row
                cell_obj.set_facecolor('#F0F8FF')
                if j == 0:  # Component name
                    cell_obj.set_text_props(weight='bold', color='#4472C4')
            elif 'NO DISCOUNT SCENARIO' in str(cell):  # Section header
                cell_obj.set_facecolor('#E8F6E8')
                cell_obj.set_text_props(weight='bold', color='#70AD47')
            elif 'Interest on Undiscounted' in str(cell) or 'Retained Discount' in str(cell):  # Component rows
                cell_obj.set_facecolor('#F0FFF0')
                if j == 0:  # Component name
                    cell_obj.set_text_props(weight='bold', color='#228B22')
            elif 'Total NO DISCOUNT' in str(cell):  # Total row
                cell_obj.set_facecolor('#D4E8D4')
                if j == 0:  # Component name
                    cell_obj.set_text_props(weight='bold', color='#70AD47')
            elif 'FORECAST METHODOLOGY' in str(cell) or 'COMPARISON METRICS' in str(cell):  # Section headers
                cell_obj.set_facecolor('#F8F8F8')
                cell_obj.set_text_props(weight='bold')
            elif 'Revenue Multiplier' in str(cell) or 'Average Discount' in str(cell) or 'Implied' in str(cell):  # Metric rows
                cell_obj.set_facecolor('#FFF8DC')
                if j == 0:  # Metric name
                    cell_obj.set_text_props(weight='bold')
            elif j == 0 and str(cell).strip() != '':  # Other metric names
                if 'Discount Rate Source' in str(cell) or 'Bootstrap' in str(cell) or 'Forecast Period' in str(cell):
                    cell_obj.set_facecolor('#F5F5F5')
    
    plt.title('Jan 2026 - Jan 2027 Revenue Forecast Component Analysis\\n(Using Discount Rate Forecast from 11.4)', 
              fontsize=16, fontweight='bold', pad=20)
    
    table_output = ALT_VIS / '15.6_revenue_breakdown_table.png'
    table_output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(table_output, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved revenue breakdown table: {table_output.name}')
    plt.close()

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print('\n' + '='*80)
    print('✅ BOOTSTRAP COMPLETE (15.6)')
    print('='*80)
    
    print(f'\nKey Differences from 15.5:')
    print(f'  • Uses discount rate forecast from 11.4 (not constant multiplier from 11.5)')
    print(f'  • Undiscounted = Discounted × (1 + discount_rate/100)')
    print(f'  • Captures time-varying discount behavior')
    print(f'  • Should give more realistic undiscounted estimates')
    
    print(f'\nAverage implied multiplier: {multiplier_means.mean():.3f}')
    print(f'Multiplier range: {multiplier_means.min():.3f} - {multiplier_means.max():.3f}')
    
    print(f'\n📁 Output Files:')
    print(f'  • 15.6_bootstrap_component_summary.csv - Monthly revenue summary')
    print(f'  • 15.6_cumulative_revenue_last_12_months.png - Revenue scenarios')
    print(f'  • 15.6_cumulative_revenue_with_ci_last_12_months.png - Revenue with confidence intervals (large fonts)')
    print(f'  • 15.6_discount_rate_comparison.png - Multiplier and rate analysis')
    print(f'  • 15.6_revenue_component_breakdown.png - Stacked revenue component breakdown')
    print(f'  • 15.6_revenue_breakdown_table.png - Detailed component analysis table')
    
    print('\n' + '='*80)
    print('NEXT STEPS:')
    print('  1. Compare 15.6 results with 15.5 to see impact of discount rate approach')
    print('  2. Validate that NO_DISCOUNT scenario is now more realistic')
    print('  3. Use preferred approach for final revenue projections')
    print('='*80)

if __name__ == '__main__':
    main()