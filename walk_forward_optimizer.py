import pandas as pd
import numpy as np
from backtester import Backtester
from enhanced_strategies import EnhancedTrendFollowingStrategy, EnhancedMLStrategy
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and preprocess BTC/USDT data"""
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
    
    return data

def walk_forward_optimization(strategy_class, data, param_grid, train_months=6, test_months=1):
    """
    Perform walk-forward optimization to prevent overfitting
    
    Args:
        strategy_class: Strategy class to optimize
        data: Price data
        param_grid: Dictionary of parameters to test
        train_months: Number of months for training
        test_months: Number of months for testing
    """
    results = []
    param_combinations = list(ParameterGrid(param_grid))
    
    # Create date ranges for walk-forward
    start_date = data.index[0]
    end_date = data.index[-1]
    
    current_date = start_date
    while current_date < end_date:
        # Define training period
        train_start = current_date
        train_end = train_start + pd.DateOffset(months=train_months)
        
        # Define testing period
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        # Check if we have enough data
        if test_end > end_date:
            break
            
        print(f"Walk-forward period: {train_start.date()} to {test_end.date()}")
        
        # Get training and testing data
        train_data = data[train_start:train_end]
        test_data = data[test_start:test_end]
        
        if len(train_data) < 1000 or len(test_data) < 100:  # Minimum data requirements
            current_date += pd.DateOffset(months=1)
            continue
        
        best_params = None
        best_score = -np.inf
        
        # Test all parameter combinations on training data
        for params in param_combinations:
            try:
                if strategy_class == EnhancedTrendFollowingStrategy:
                    strategy = strategy_class(train_data, **params)
                else:
                    strategy = strategy_class(train_data, **params)
                
                backtester = Backtester(train_data, strategy)
                backtester.run()
                metrics = backtester.calculate_metrics()
                
                if metrics and metrics['sharpe_ratio'] > best_score:
                    best_score = metrics['sharpe_ratio']
                    best_params = params
                    
            except Exception as e:
                print(f"Error testing params {params}: {e}")
                continue
        
        if best_params is None:
            current_date += pd.DateOffset(months=1)
            continue
        
        # Test best parameters on out-of-sample data
        try:
            if strategy_class == EnhancedTrendFollowingStrategy:
                best_strategy = strategy_class(test_data, **best_params)
            else:
                best_strategy = strategy_class(test_data, **best_params)
            
            backtester = Backtester(test_data, best_strategy)
            backtester.run()
            test_metrics = backtester.calculate_metrics()
            
            if test_metrics:
                result = {
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'best_params': best_params,
                    'train_sharpe': best_score,
                    'test_sharpe': test_metrics['sharpe_ratio'],
                    'test_return': test_metrics['total_return'],
                    'test_drawdown': test_metrics['max_drawdown'],
                    'test_trades': test_metrics['num_trades']
                }
                results.append(result)
                
                print(f"  Best params: {best_params}")
                print(f"  Train Sharpe: {best_score:.3f}, Test Sharpe: {test_metrics['sharpe_ratio']:.3f}")
                
        except Exception as e:
            print(f"Error testing best params: {e}")
        
        current_date += pd.DateOffset(months=1)
    
    return results

def analyze_walk_forward_results(results):
    """Analyze walk-forward optimization results"""
    if not results:
        print("No results to analyze")
        return
    
    df = pd.DataFrame(results)
    
    print("\n=== Walk-Forward Analysis ===")
    print(f"Total periods tested: {len(df)}")
    print(f"Average test Sharpe ratio: {df['test_sharpe'].mean():.3f}")
    print(f"Average test return: {df['test_return'].mean():.3f}")
    print(f"Average test drawdown: {df['test_drawdown'].mean():.3f}")
    print(f"Average number of trades: {df['test_trades'].mean():.1f}")
    
    # Find most consistent parameters
    param_counts = {}
    for params in df['best_params']:
        param_key = str(sorted(params.items()))
        param_counts[param_key] = param_counts.get(param_key, 0) + 1
    
    most_common_params = max(param_counts.items(), key=lambda x: x[1])
    print(f"\nMost frequently selected parameters: {most_common_params[0]}")
    print(f"Selected {most_common_params[1]} times out of {len(df)} periods")
    
    # Performance consistency
    positive_periods = len(df[df['test_sharpe'] > 0])
    print(f"Positive Sharpe periods: {positive_periods}/{len(df)} ({positive_periods/len(df)*100:.1f}%)")
    
    return df

def main():
    """Main walk-forward optimization function"""
    data = load_data()
    
    # Define parameter grids for optimization
    trend_param_grid = {
        'slow_ma': [150, 200, 250],
        'entry_threshold': [0.01, 0.015, 0.02],
        'atr_multiplier': [2.0, 2.5, 3.0],
        'profit_target_multiplier': [2.0, 3.0, 4.0],
        'use_kelly_sizing': [True, False]
    }
    
    ml_param_grid = {
        'trend_ma': [150, 200, 250],
        'short_ma': [15, 20, 25],
        'rsi_period': [12, 14, 16],
        'rsi_lower': [25, 30, 35],
        'rsi_exit': [65, 70, 75],
        'profit_target': [0.10, 0.15, 0.20],
        'prediction_threshold': [0.015, 0.02, 0.025]
    }
    
    print("=== Walk-Forward Optimization ===")
    
    # Optimize Enhanced Trend Following Strategy
    print("\n1. Walk-forward optimization for Enhanced Trend Following Strategy...")
    trend_results = walk_forward_optimization(
        EnhancedTrendFollowingStrategy, data, trend_param_grid, 
        train_months=6, test_months=1
    )
    trend_df = analyze_walk_forward_results(trend_results)
    
    # Optimize Enhanced ML Strategy
    print("\n2. Walk-forward optimization for Enhanced ML Strategy...")
    ml_results = walk_forward_optimization(
        EnhancedMLStrategy, data, ml_param_grid,
        train_months=6, test_months=1
    )
    ml_df = analyze_walk_forward_results(ml_results)
    
    # Compare strategies
    if trend_df is not None and ml_df is not None:
        print("\n=== Strategy Comparison ===")
        print(f"Enhanced Trend Following - Avg Sharpe: {trend_df['test_sharpe'].mean():.3f}")
        print(f"Enhanced ML Strategy - Avg Sharpe: {ml_df['test_sharpe'].mean():.3f}")
        
        if trend_df['test_sharpe'].mean() > ml_df['test_sharpe'].mean():
            print("Enhanced Trend Following Strategy performs better on average")
        else:
            print("Enhanced ML Strategy performs better on average")
    
    return trend_results, ml_results

if __name__ == '__main__':
    main()

