import pandas as pd
from backtester import Backtester
from enhanced_strategies import EnhancedTrendFollowingStrategy

def debug_enhanced_strategy():
    """Debug the enhanced strategy to see why no trades are generated"""
    # Load data
    column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                   'taker_buy_quote_asset_volume', 'ignore']
    data = pd.read_csv('BTCUSDT_1h.csv', names=column_names)
    
    # Fix timestamps
    data['open_time'] = data['open_time'].apply(lambda x: x / 1000 if x > 10**15 else x)
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('open_time', inplace=True)
    
    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col])
    
    print("Data shape:", data.shape)
    print("Close price range:", data['close'].min(), "to", data['close'].max())
    
    # Test with very simple parameters
    strategy = EnhancedTrendFollowingStrategy(
        data, slow_ma=10, entry_threshold=0.01, atr_multiplier=2.0,
        profit_target_multiplier=2.0, use_kelly_sizing=False
    )
    
    # Run backtest manually to debug
    data_copy = data.copy()
    data_copy['slow_ma'] = data_copy['close'].rolling(window=10).mean()
    
    # Calculate ATR
    high_low = data_copy['high'] - data_copy['low']
    high_close_prev = (data_copy['high'] - data_copy['close'].shift()).abs()
    low_close_prev = (data_copy['low'] - data_copy['close'].shift()).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    data_copy['atr'] = tr.rolling(window=14).mean()
    
    print("\nFirst 20 rows with indicators:")
    print(data_copy[['close', 'slow_ma', 'atr']].head(20))
    
    print("\nLast 20 rows with indicators:")
    print(data_copy[['close', 'slow_ma', 'atr']].tail(20))
    
    # Check entry conditions
    entry_condition = data_copy['close'] > data_copy['slow_ma'] * (1 + 0.01)
    print(f"\nEntry condition met {entry_condition.sum()} times")
    
    if entry_condition.sum() > 0:
        print("First few entry signals:")
        entry_signals = data_copy[entry_condition]
        print(entry_signals[['close', 'slow_ma']].head())
    
    # Run actual backtest
    backtester = Backtester(data, strategy)
    backtester.run()
    metrics = backtester.calculate_metrics()
    
    print("\nStrategy Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

if __name__ == '__main__':
    debug_enhanced_strategy()

