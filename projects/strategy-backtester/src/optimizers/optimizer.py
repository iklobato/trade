import pandas as pd
import numpy as np
from backtester import Backtester
from strategies import TrendFollowingStrategy
import itertools

def main():
    # Load data
    column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    data = pd.read_csv('BTCUSDT_d.csv', names=column_names)

    # Fix timestamps
    data['open_time'] = data['open_time'].apply(lambda x: x / 1000 if x > 10**15 else x)

    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('open_time', inplace=True)

    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col])

    # Define parameter ranges for grid search
    fast_ma_range = [10, 20, 50]
    slow_ma_range = [100, 150, 200]
    atr_multiplier_range = [1.5, 2.0, 2.5]

    best_sharpe = -np.inf
    best_params = None

    param_combinations = list(itertools.product(fast_ma_range, slow_ma_range, atr_multiplier_range))

    for params in param_combinations:
        fast_ma, slow_ma, atr_multiplier = params
        
        strategy = TrendFollowingStrategy(data, fast_ma=fast_ma, slow_ma=slow_ma, atr_multiplier=atr_multiplier)
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()

        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_params = params
            best_metrics = metrics

    print("Best Parameters:")
    print(f"fast_ma: {best_params[0]}, slow_ma: {best_params[1]}, atr_multiplier: {best_params[2]}")
    print("\nPerformance Metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

if __name__ == '__main__':
    main()