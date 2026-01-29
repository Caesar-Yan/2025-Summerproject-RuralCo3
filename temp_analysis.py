import pandas as pd

# Load forecast
forecast = pd.read_csv(r'\\file\Usersc$\cch155\Home\Desktop\2025\data605\2025-Summerproject-RuralCo3\visualisations\11.6_forecast_comparison_table.csv')
forecast['invoice_period'] = pd.to_datetime(forecast['invoice_period'])

# Last 12 months of forecast
last_12 = forecast.sort_values('invoice_period').tail(12)
total_disc_last12 = last_12['forecast_discounted_price'].sum()
print('Last 12 months forecast (2026-02 to 2027-01):')
print(f'  Total discounted amount: ${total_disc_last12:,.2f}')
print(f'  Months: {last_12["invoice_period"].min().date()} to {last_12["invoice_period"].max().date()}')

# Calculate what WITH_DISCOUNT revenue should be based on interest calculation
daily_rate = 0.2395 / 365
expected_days = 60
avg_late_rate = 0.08  # Approximate average late rate

interest_revenue_last12 = total_disc_last12 * avg_late_rate * daily_rate * expected_days
print(f'\n  Expected interest revenue (WITH_DISCOUNT scenario):')
print(f'    (${total_disc_last12:,.2f} * {avg_late_rate:.1%} late * 23.95% APR/365 * {expected_days} days)')
print(f'    = ${interest_revenue_last12:,.2f}')

# Historical data from 11.6 period summary
print(f'\nHistorical (2024-07-01 to 2025-07-01) from 11.6:')
print(f'  Total discounted amount: $83,817,141.66')
print(f'  This is the NET revenue (after discount applied)')

# Calculate what historical revenue should be
hist_disc_net = 83_817_141.66
interest_revenue_hist = hist_disc_net * avg_late_rate * daily_rate * expected_days
print(f'\n  Expected interest revenue (WITH_DISCOUNT scenario):')
print(f'    (${hist_disc_net:,.2f} * {avg_late_rate:.1%} late * 23.95% APR/365 * {expected_days} days)')
print(f'    = ${interest_revenue_hist:,.2f}')
