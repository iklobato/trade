import pandas as pd
import numpy as np
from backtester import Backtester
from enhanced_strategies import EnhancedTrendFollowingStrategy, EnhancedMLStrategy
import os
import warnings
warnings.filterwarnings('ignore')

def load_btc_data():
    """Load BTC/USDT data"""
    column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                   'taker_buy_quote_asset_volume', 'ignore']
    
    if os.path.exists('BTCUSDT_1h.csv'):
        data = pd.read_csv('BTCUSDT_1h.csv', names=column_names)
    else:
        print("BTCUSDT_1h.csv not found")
        return None
    
    # Fix timestamps
    data['open_time'] = data['open_time'].apply(lambda x: x / 1000 if x > 10**15 else x)
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('open_time', inplace=True)
    
    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col])
    
    return data

def load_eth_data():
    """Load ETH/USDT data if available"""
    column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                   'taker_buy_quote_asset_volume', 'ignore']
    
    if os.path.exists('Bitstamp_ETHUSD_d.csv'):
        data = pd.read_csv('Bitstamp_ETHUSD_d.csv')
        # Convert to hourly format (assuming daily data)
        data['open_time'] = pd.to_datetime(data['Timestamp'])
        data.set_index('open_time', inplace=True)
        
        # Rename columns to match expected format
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume_(BTC)': 'volume'
        })
        
        # Resample daily data to hourly (forward fill)
        data = data.resample('1H').ffill()
        
        return data
    else:
        print("ETH data not found")
        return None

def test_strategy_on_asset(strategy_class, data, asset_name, **strategy_params):
    """Test a strategy on a specific asset"""
    print(f"\nTesting {strategy_class.__name__} on {asset_name}...")
    
    try:
        strategy = strategy_class(data, **strategy_params)
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()
        
        if metrics:
            print(f"{asset_name} Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
            return metrics
        else:
            print(f"Failed to calculate metrics for {asset_name}")
            return None
            
    except Exception as e:
        print(f"Error testing {strategy_class.__name__} on {asset_name}: {e}")
        return None

def compare_strategies_across_assets():
    """Compare strategies across different assets"""
    results = {}
    
    # Load data
    btc_data = load_btc_data()
    eth_data = load_eth_data()
    
    if btc_data is None:
        print("Cannot proceed without BTC data")
        return
    
    # Define strategies to test
    strategies = {
        'Enhanced Trend Following': {
            'class': EnhancedTrendFollowingStrategy,
            'params': {
                'slow_ma': 200, 'entry_threshold': 0.015, 'atr_multiplier': 3.0,
                'profit_target_multiplier': 2.0, 'use_kelly_sizing': True
            }
        },
        'Enhanced ML Strategy': {
            'class': EnhancedMLStrategy,
            'params': {
                'trend_ma': 200, 'short_ma': 20, 'rsi_period': 14,
                'rsi_lower': 30, 'rsi_exit': 70, 'profit_target': 0.15,
                'prediction_threshold': 0.02
            }
        }
    }
    
    # Test on BTC
    print("=== Testing on BTC/USDT ===")
    btc_results = {}
    for name, config in strategies.items():
        metrics = test_strategy_on_asset(
            config['class'], btc_data, 'BTC/USDT', **config['params']
        )
        if metrics:
            btc_results[name] = metrics
    
    results['BTC/USDT'] = btc_results
    
    # Test on ETH if available
    if eth_data is not None:
        print("\n=== Testing on ETH/USDT ===")
        eth_results = {}
        for name, config in strategies.items():
            metrics = test_strategy_on_asset(
                config['class'], eth_data, 'ETH/USDT', **config['params']
            )
            if metrics:
                eth_results[name] = metrics
        
        results['ETH/USDT'] = eth_results
    
    return results

def analyze_cross_asset_performance(results):
    """Analyze performance across different assets"""
    print("\n=== Cross-Asset Performance Analysis ===")
    
    for asset, asset_results in results.items():
        print(f"\n{asset}:")
        for strategy_name, metrics in asset_results.items():
            sharpe = metrics.get('sharpe_ratio', 0)
            return_pct = metrics.get('total_return', 0) * 100
            drawdown = metrics.get('max_drawdown', 0) * 100
            trades = metrics.get('num_trades', 0)
            
            print(f"  {strategy_name}:")
            print(f"    Sharpe Ratio: {sharpe:.3f}")
            print(f"    Total Return: {return_pct:.2f}%")
            print(f"    Max Drawdown: {drawdown:.2f}%")
            print(f"    Number of Trades: {trades}")
    
    # Find best performing strategy overall
    all_sharpes = []
    strategy_names = []
    
    for asset_results in results.values():
        for strategy_name, metrics in asset_results.items():
            sharpe = metrics.get('sharpe_ratio', 0)
            all_sharpes.append(sharpe)
            strategy_names.append(strategy_name)
    
    if all_sharpes:
        best_idx = np.argmax(all_sharpes)
        print(f"\nBest Overall Performance:")
        print(f"Strategy: {strategy_names[best_idx]}")
        print(f"Sharpe Ratio: {all_sharpes[best_idx]:.3f}")

def test_different_timeframes():
    """Test strategies on different timeframes"""
    print("\n=== Testing Different Timeframes ===")
    
    btc_data = load_btc_data()
    if btc_data is None:
        print("Cannot test timeframes without BTC data")
        return
    
    # Resample to different timeframes
    timeframes = {
        '4H': '4H',
        'Daily': '1D'
    }
    
    timeframe_results = {}
    
    for tf_name, tf_period in timeframes.items():
        print(f"\nTesting {tf_name} timeframe...")
        
        # Resample data
        tf_data = btc_data.resample(tf_period).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(tf_data) < 100:
            print(f"Not enough data for {tf_name} timeframe")
            continue
        
        # Test Enhanced Trend Following Strategy
        strategy = EnhancedTrendFollowingStrategy(
            tf_data, slow_ma=50, entry_threshold=0.02, atr_multiplier=2.5,
            profit_target_multiplier=3.0, use_kelly_sizing=True
        )
        
        backtester = Backtester(tf_data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()
        
        if metrics:
            print(f"{tf_name} Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
            timeframe_results[tf_name] = metrics
    
    return timeframe_results

def main():
    """Main function to run comprehensive testing"""
    print("=== Comprehensive Strategy Testing ===")
    
    # Test across assets
    cross_asset_results = compare_strategies_across_assets()
    analyze_cross_asset_performance(cross_asset_results)
    
    # Test different timeframes
    timeframe_results = test_different_timeframes()
    
    print("\n=== Testing Complete ===")
    
    return cross_asset_results, timeframe_results

if __name__ == '__main__':
    main()

