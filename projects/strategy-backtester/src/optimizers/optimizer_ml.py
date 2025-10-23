import pandas as pd
import numpy as np
from backtester import Backtester
from strategies import TrendFilteredMeanReversionMLStrategy
import itertools

def main():
    # Load data
    column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    data = pd.read_csv('BTCUSDT_1h.csv', names=column_names)

    # Fix timestamps
    data['open_time'] = data['open_time'].apply(lambda x: x / 1000 if x > 10**15 else x)

    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('open_time', inplace=True)

    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col])

    # Define parameter ranges for grid search
    trend_ma_range = [200, 300]
    short_ma_range = [20, 30]
    rsi_period_range = [14, 21]
    rsi_lower_range = [25, 30]
    rsi_exit_range = [70, 75]
    profit_target_range = [0.05, 0.1]
    prediction_threshold_range = [0.01, 0.02]

    best_sharpe = -np.inf
    best_params = None

    param_combinations = list(itertools.product(trend_ma_range, short_ma_range, rsi_period_range, rsi_lower_range, rsi_exit_range, profit_target_range, prediction_threshold_range))

    for i, params in enumerate(param_combinations):
        trend_ma, short_ma, rsi_period, rsi_lower, rsi_exit, profit_target, prediction_threshold = params
        print(f"Testing params {i+1}/{len(param_combinations)}: {params}")
        
        strategy = TrendFilteredMeanReversionMLStrategy(data, trend_ma=trend_ma, short_ma=short_ma, rsi_period=rsi_period, rsi_lower=rsi_lower, rsi_exit=rsi_exit, profit_target=profit_target, prediction_threshold=prediction_threshold)
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()

        if metrics is not None and metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_params = params
            best_metrics = metrics

    print("Best Parameters:")
    print(f"trend_ma: {best_params[0]}, short_ma: {best_params[1]}, rsi_period: {best_params[2]}, rsi_lower: {best_params[3]}, rsi_exit: {best_params[4]}, profit_target: {best_params[5]}, prediction_threshold: {best_params[6]}")
    print("\nPerformance Metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

if __name__ == '__main__':
    main()