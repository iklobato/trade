import pandas as pd
from backtester import Backtester
from simple_enhanced_strategies import SimpleEnhancedTrendFollowingStrategy, SimpleMLStrategy

def test_simple_enhanced_strategies():
    """Test the simplified enhanced strategies"""
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
    
    print("=== Testing Simplified Enhanced Strategies ===")
    
    # Test Simple Enhanced Trend Following Strategy
    print("\n1. Testing Simple Enhanced Trend Following Strategy...")
    try:
        strategy = SimpleEnhancedTrendFollowingStrategy(
            data, slow_ma=30, entry_threshold=0.015, atr_multiplier=2.5,
            profit_target_multiplier=2.0, use_kelly_sizing=True
        )
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()
        
        print("Simple Enhanced Trend Following Strategy Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
            
    except Exception as e:
        print(f"Error testing Simple Enhanced Trend Following Strategy: {e}")
    
    # Test Simple ML Strategy
    print("\n2. Testing Simple ML Strategy...")
    try:
        strategy = SimpleMLStrategy(
            data, trend_ma=30, short_ma=10, rsi_period=14, rsi_lower=30,
            rsi_exit=70, profit_target=0.10, prediction_threshold=0.015
        )
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()
        
        print("Simple ML Strategy Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
            
    except Exception as e:
        print(f"Error testing Simple ML Strategy: {e}")

if __name__ == '__main__':
    test_simple_enhanced_strategies()

