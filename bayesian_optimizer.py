import pandas as pd
import numpy as np
from backtester import Backtester
from enhanced_strategies import EnhancedTrendFollowingStrategy, EnhancedMLStrategy
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
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

def optimize_enhanced_trend_following():
    """Optimize Enhanced Trend Following Strategy using Bayesian optimization"""
    data = load_data()
    
    # Define parameter space
    dimensions = [
        Integer(100, 400, name='slow_ma'),
        Real(0.005, 0.05, name='entry_threshold'),
        Real(1.5, 4.0, name='atr_multiplier'),
        Real(2.0, 5.0, name='profit_target_multiplier'),
        Real(0.01, 0.05, name='volatility_threshold'),
        Integer(10, 30, name='momentum_period')
    ]
    
    @use_named_args(dimensions=dimensions)
    def objective(slow_ma, entry_threshold, atr_multiplier, profit_target_multiplier, 
                  volatility_threshold, momentum_period):
        try:
            strategy = EnhancedTrendFollowingStrategy(
                data, slow_ma=int(slow_ma), entry_threshold=entry_threshold,
                atr_multiplier=atr_multiplier, profit_target_multiplier=profit_target_multiplier,
                volatility_threshold=volatility_threshold, momentum_period=int(momentum_period),
                use_kelly_sizing=True
            )
            backtester = Backtester(data, strategy)
            backtester.run()
            metrics = backtester.calculate_metrics()
            
            if metrics is None:
                return -1000
            
            # Multi-objective optimization: maximize Sharpe ratio while minimizing drawdown
            sharpe_penalty = -metrics['sharpe_ratio'] if metrics['sharpe_ratio'] > 0 else 1000
            drawdown_penalty = abs(metrics['max_drawdown']) * 10
            
            return sharpe_penalty + drawdown_penalty
            
        except Exception as e:
            print(f"Error in optimization: {e}")
            return 1000
    
    print("Starting Bayesian optimization for Enhanced Trend Following Strategy...")
    result = gp_minimize(func=objective, dimensions=dimensions, n_calls=50, random_state=42)
    
    print("Optimization completed!")
    print(f"Best parameters: {result.x}")
    print(f"Best score: {result.fun}")
    
    # Test best parameters
    best_params = dict(zip([d.name for d in dimensions], result.x))
    strategy = EnhancedTrendFollowingStrategy(
        data, slow_ma=int(best_params['slow_ma']), 
        entry_threshold=best_params['entry_threshold'],
        atr_multiplier=best_params['atr_multiplier'], 
        profit_target_multiplier=best_params['profit_target_multiplier'],
        volatility_threshold=best_params['volatility_threshold'], 
        momentum_period=int(best_params['momentum_period']),
        use_kelly_sizing=True
    )
    backtester = Backtester(data, strategy)
    backtester.run()
    metrics = backtester.calculate_metrics()
    
    print("\nBest Strategy Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    return result.x, metrics

def optimize_enhanced_ml_strategy():
    """Optimize Enhanced ML Strategy using Bayesian optimization"""
    data = load_data()
    
    # Define parameter space
    dimensions = [
        Integer(150, 300, name='trend_ma'),
        Integer(10, 30, name='short_ma'),
        Integer(10, 20, name='rsi_period'),
        Real(20, 40, name='rsi_lower'),
        Real(60, 80, name='rsi_exit'),
        Real(0.05, 0.25, name='profit_target'),
        Real(0.01, 0.05, name='prediction_threshold')
    ]
    
    @use_named_args(dimensions=dimensions)
    def objective(trend_ma, short_ma, rsi_period, rsi_lower, rsi_exit, profit_target, prediction_threshold):
        try:
            strategy = EnhancedMLStrategy(
                data, trend_ma=int(trend_ma), short_ma=int(short_ma),
                rsi_period=int(rsi_period), rsi_lower=rsi_lower, rsi_exit=rsi_exit,
                profit_target=profit_target, prediction_threshold=prediction_threshold
            )
            backtester = Backtester(data, strategy)
            backtester.run()
            metrics = backtester.calculate_metrics()
            
            if metrics is None:
                return -1000
            
            # Multi-objective optimization
            sharpe_penalty = -metrics['sharpe_ratio'] if metrics['sharpe_ratio'] > 0 else 1000
            drawdown_penalty = abs(metrics['max_drawdown']) * 10
            
            return sharpe_penalty + drawdown_penalty
            
        except Exception as e:
            print(f"Error in optimization: {e}")
            return 1000
    
    print("Starting Bayesian optimization for Enhanced ML Strategy...")
    result = gp_minimize(func=objective, dimensions=dimensions, n_calls=50, random_state=42)
    
    print("Optimization completed!")
    print(f"Best parameters: {result.x}")
    print(f"Best score: {result.fun}")
    
    # Test best parameters
    best_params = dict(zip([d.name for d in dimensions], result.x))
    strategy = EnhancedMLStrategy(
        data, trend_ma=int(best_params['trend_ma']), 
        short_ma=int(best_params['short_ma']),
        rsi_period=int(best_params['rsi_period']), 
        rsi_lower=best_params['rsi_lower'], 
        rsi_exit=best_params['rsi_exit'],
        profit_target=best_params['profit_target'], 
        prediction_threshold=best_params['prediction_threshold']
    )
    backtester = Backtester(data, strategy)
    backtester.run()
    metrics = backtester.calculate_metrics()
    
    print("\nBest ML Strategy Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    return result.x, metrics

def compare_strategies():
    """Compare all strategies"""
    data = load_data()
    
    strategies = {
        'Enhanced Trend Following': EnhancedTrendFollowingStrategy(
            data, slow_ma=200, entry_threshold=0.015, atr_multiplier=3.0, 
            profit_target_multiplier=2.0, use_kelly_sizing=True
        ),
        'Enhanced ML Strategy': EnhancedMLStrategy(
            data, trend_ma=200, short_ma=20, rsi_period=14, rsi_lower=30, 
            rsi_exit=70, profit_target=0.15, prediction_threshold=0.02
        )
    }
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\nTesting {name}...")
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()
        results[name] = metrics
        
        print(f"{name} Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    return results

if __name__ == '__main__':
    print("=== Cryptocurrency Trading Strategy Optimization ===")
    
    # Compare strategies first
    print("\n1. Comparing Enhanced Strategies...")
    compare_results = compare_strategies()
    
    # Optimize Enhanced Trend Following Strategy
    print("\n2. Optimizing Enhanced Trend Following Strategy...")
    trend_params, trend_metrics = optimize_enhanced_trend_following()
    
    # Optimize Enhanced ML Strategy
    print("\n3. Optimizing Enhanced ML Strategy...")
    ml_params, ml_metrics = optimize_enhanced_ml_strategy()
    
    print("\n=== Optimization Complete ===")

