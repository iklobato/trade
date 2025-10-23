import pandas as pd
from backtester import Backtester
from strategies import SimpleTrendFollowingStrategyV2

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

    # --- Simple Trend Following Strategy V2 ---
    print("--- Simple Trend Following Strategy V2 (Hourly Data) ---")
    strategy = SimpleTrendFollowingStrategyV2(data, slow_ma=200, entry_threshold=0.015, atr_multiplier=3.0, profit_target_multiplier=2.0)
    backtester = Backtester(data, strategy)
    backtester.run()
    metrics = backtester.calculate_metrics()
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    backtester.plot_results()


if __name__ == '__main__':
    main()