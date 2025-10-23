import pandas as pd
from backtester import Backtester
from enhanced_strategies import EnhancedTrendFollowingStrategy, EnhancedMLStrategy

def test_enhanced_strategies():
    """Test the enhanced strategies"""
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
    
    print("=== Testing Enhanced Strategies ===")
    
    # Test Enhanced Trend Following Strategy
    print("\n1. Testing Enhanced Trend Following Strategy...")
    try:
        strategy = EnhancedTrendFollowingStrategy(
            data, slow_ma=30, entry_threshold=0.015, atr_multiplier=2.5,
            profit_target_multiplier=2.0, use_kelly_sizing=True
        )
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()
        
        print("Enhanced Trend Following Strategy Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
            
    except Exception as e:
        print(f"Error testing Enhanced Trend Following Strategy: {e}")
    
    # Test Enhanced ML Strategy
    print("\n2. Testing Enhanced ML Strategy...")
    try:
        strategy = EnhancedMLStrategy(
            data, trend_ma=30, short_ma=10, rsi_period=14, rsi_lower=30,
            rsi_exit=70, profit_target=0.15, prediction_threshold=0.02
        )
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()
        
        print("Enhanced ML Strategy Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
            
    except Exception as e:
        print(f"Error testing Enhanced ML Strategy: {e}")

if __name__ == '__main__':
    test_enhanced_strategies()
