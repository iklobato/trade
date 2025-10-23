import pandas as pd
import numpy as np
from backtester import Backtester
from strategies import (
    TrendFollowingStrategy, 
    TrendFollowingEMAStrategy, 
    SimpleTrendFollowingStrategy, 
    SimpleTrendFollowingStrategyV2,
    TrendFilteredMeanReversionMLStrategy
)
from simple_enhanced_strategies import (
    SimpleEnhancedTrendFollowingStrategy, 
    SimpleMLStrategy
)
import os
import warnings
warnings.filterwarnings('ignore')

def load_all_data():
    """Load and combine all available BTC/USDT data"""
    column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                   'taker_buy_quote_asset_volume', 'ignore']
    
    data_files = [
        'BTCUSDT_1h.csv',           # 2023-01 data
        'BTCUSDT-1h-2024-02.csv',   # 2024-02 data
        'BTCUSDT-1h-2024-03.csv',   # 2024-03 data
        'BTCUSDT-1h-2025-06.csv',   # 2025-06 data
        'BTCUSDT-1h-2025-08.csv',   # 2025-08 data
        'BTCUSDT-1h-2025-09.csv'    # 2025-09 data
    ]
    
    all_data = []
    
    for file in data_files:
        if os.path.exists(file):
            print(f"Loading {file}...")
            try:
                data = pd.read_csv(file, names=column_names)
                
                # Fix timestamps (some files have different timestamp formats)
                if data['open_time'].iloc[0] > 10**15:
                    data['open_time'] = data['open_time'] / 1000
                
                data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
                data.set_index('open_time', inplace=True)
                
                # Convert columns to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    data[col] = pd.to_numeric(data[col])
                
                all_data.append(data)
                print(f"  Loaded {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
                
            except Exception as e:
                print(f"  Error loading {file}: {e}")
    
    if not all_data:
        print("No data files found!")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=False)
    combined_data = combined_data.sort_index()
    combined_data = combined_data.drop_duplicates()
    
    print(f"\nCombined dataset:")
    print(f"  Total records: {len(combined_data)}")
    print(f"  Date range: {combined_data.index[0].date()} to {combined_data.index[-1].date()}")
    print(f"  Price range: ${combined_data['close'].min():.2f} to ${combined_data['close'].max():.2f}")
    
    return combined_data

def test_strategy(strategy_class, data, strategy_name, **params):
    """Test a single strategy and return results"""
    try:
        strategy = strategy_class(data, **params)
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()
        
        if metrics and metrics['num_trades'] > 0:
            return {
                'strategy_name': strategy_name,
                'params': params,
                'metrics': metrics,
                'success': True
            }
        else:
            return {
                'strategy_name': strategy_name,
                'params': params,
                'metrics': None,
                'success': False,
                'error': 'No trades generated'
            }
    except Exception as e:
        return {
            'strategy_name': strategy_name,
            'params': params,
            'metrics': None,
            'success': False,
            'error': str(e)
        }

def comprehensive_backtest():
    """Run comprehensive backtest on all strategies"""
    print("="*80)
    print("COMPREHENSIVE CRYPTOCURRENCY TRADING STRATEGY BACKTEST")
    print("="*80)
    
    # Load all available data
    data = load_all_data()
    if data is None:
        return
    
    # Define strategies to test with optimized parameters
    strategies_to_test = [
        # Original strategies
        {
            'class': TrendFollowingStrategy,
            'name': 'Trend Following (SMA)',
            'params': {'fast_ma': 50, 'slow_ma': 200, 'atr_multiplier': 2.0}
        },
        {
            'class': TrendFollowingEMAStrategy,
            'name': 'Trend Following (EMA)',
            'params': {'fast_ema': 50, 'slow_ema': 200, 'atr_multiplier': 2.0}
        },
        {
            'class': SimpleTrendFollowingStrategy,
            'name': 'Simple Trend Following',
            'params': {'slow_ma': 200, 'entry_threshold': 0.02, 'atr_multiplier': 2.0}
        },
        {
            'class': SimpleTrendFollowingStrategyV2,
            'name': 'Simple Trend Following V2',
            'params': {'slow_ma': 200, 'entry_threshold': 0.015, 'atr_multiplier': 3.0, 'profit_target_multiplier': 2.0}
        },
        {
            'class': TrendFilteredMeanReversionMLStrategy,
            'name': 'ML Mean Reversion',
            'params': {'trend_ma': 200, 'short_ma': 20, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_exit': 70, 'profit_target': 0.15, 'prediction_threshold': 0.02}
        },
        
        # Enhanced strategies with different parameters
        {
            'class': SimpleEnhancedTrendFollowingStrategy,
            'name': 'Enhanced Trend Following (Kelly)',
            'params': {'slow_ma': 30, 'entry_threshold': 0.01, 'atr_multiplier': 2.0, 'profit_target_multiplier': 2.0, 'use_kelly_sizing': True}
        },
        {
            'class': SimpleEnhancedTrendFollowingStrategy,
            'name': 'Enhanced Trend Following (No Kelly)',
            'params': {'slow_ma': 30, 'entry_threshold': 0.01, 'atr_multiplier': 2.0, 'profit_target_multiplier': 2.0, 'use_kelly_sizing': False}
        },
        {
            'class': SimpleEnhancedTrendFollowingStrategy,
            'name': 'Enhanced Trend Following (Conservative)',
            'params': {'slow_ma': 50, 'entry_threshold': 0.015, 'atr_multiplier': 2.5, 'profit_target_multiplier': 2.5, 'use_kelly_sizing': False}
        },
        {
            'class': SimpleMLStrategy,
            'name': 'Simple ML Strategy',
            'params': {'trend_ma': 30, 'short_ma': 10, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_exit': 70, 'profit_target': 0.10, 'prediction_threshold': 0.015}
        }
    ]
    
    print(f"\nTesting {len(strategies_to_test)} strategies...")
    print("-" * 80)
    
    results = []
    
    for i, strategy_config in enumerate(strategies_to_test):
        print(f"\n{i+1}. Testing {strategy_config['name']}...")
        result = test_strategy(
            strategy_config['class'], 
            data, 
            strategy_config['name'], 
            **strategy_config['params']
        )
        results.append(result)
        
        if result['success']:
            metrics = result['metrics']
            print(f"   ✓ Success - {metrics['num_trades']} trades")
            print(f"   Total Return: {metrics['total_return']*100:.2f}%")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
        else:
            print(f"   ✗ Failed: {result['error']}")
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("\nNo strategies generated trades successfully!")
        return
    
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    for result in successful_results:
        metrics = result['metrics']
        comparison_data.append({
            'Strategy': result['strategy_name'],
            'Total Return (%)': f"{metrics['total_return']*100:.2f}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
            'Max Drawdown (%)': f"{metrics['max_drawdown']*100:.2f}",
            'Win Rate (%)': f"{metrics['win_rate']*100:.1f}",
            'Num Trades': metrics['num_trades'],
            'Profit Factor': f"{metrics['profit_factor']:.2f}",
            'Calmar Ratio': f"{metrics['calmar_ratio']:.3f}"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Find best strategies
    print("\n" + "="*60)
    print("BEST PERFORMING STRATEGIES")
    print("="*60)
    
    # Best by different metrics
    best_sharpe = max(successful_results, key=lambda x: x['metrics']['sharpe_ratio'])
    best_return = max(successful_results, key=lambda x: x['metrics']['total_return'])
    best_calmar = max(successful_results, key=lambda x: x['metrics']['calmar_ratio'])
    lowest_drawdown = min(successful_results, key=lambda x: x['metrics']['max_drawdown'])
    most_trades = max(successful_results, key=lambda x: x['metrics']['num_trades'])
    
    print(f"🏆 Best Sharpe Ratio: {best_sharpe['strategy_name']}")
    print(f"   Sharpe: {best_sharpe['metrics']['sharpe_ratio']:.3f}, Return: {best_sharpe['metrics']['total_return']*100:.2f}%")
    
    print(f"\n💰 Best Total Return: {best_return['strategy_name']}")
    print(f"   Return: {best_return['metrics']['total_return']*100:.2f}%, Sharpe: {best_return['metrics']['sharpe_ratio']:.3f}")
    
    print(f"\n📊 Best Calmar Ratio: {best_calmar['strategy_name']}")
    print(f"   Calmar: {best_calmar['metrics']['calmar_ratio']:.3f}, Return: {best_calmar['metrics']['total_return']*100:.2f}%")
    
    print(f"\n🛡️ Lowest Drawdown: {lowest_drawdown['strategy_name']}")
    print(f"   Drawdown: {lowest_drawdown['metrics']['max_drawdown']*100:.2f}%, Return: {lowest_drawdown['metrics']['total_return']*100:.2f}%")
    
    print(f"\n📈 Most Active: {most_trades['strategy_name']}")
    print(f"   Trades: {most_trades['metrics']['num_trades']}, Return: {most_trades['metrics']['total_return']*100:.2f}%")
    
    # Overall recommendation
    print("\n" + "="*60)
    print("OVERALL RECOMMENDATION")
    print("="*60)
    
    # Score strategies based on multiple criteria
    scored_strategies = []
    for result in successful_results:
        metrics = result['metrics']
        score = (
            metrics['sharpe_ratio'] * 0.3 +  # Risk-adjusted returns
            metrics['total_return'] * 10 * 0.25 +  # Absolute returns
            (1 + metrics['max_drawdown']) * 0.2 +  # Drawdown (higher is better)
            metrics['win_rate'] * 0.15 +  # Win rate
            min(metrics['num_trades'] / 50, 1) * 0.1  # Trade frequency (capped)
        )
        scored_strategies.append((result, score))
    
    best_overall = max(scored_strategies, key=lambda x: x[1])
    
    print(f"🥇 BEST OVERALL STRATEGY: {best_overall[0]['strategy_name']}")
    print(f"   Composite Score: {best_overall[1]:.3f}")
    print(f"   Parameters: {best_overall[0]['params']}")
    
    metrics = best_overall[0]['metrics']
    print(f"\n   Performance Summary:")
    print(f"   • Total Return: {metrics['total_return']*100:.2f}%")
    print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"   • Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"   • Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"   • Number of Trades: {metrics['num_trades']}")
    print(f"   • Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   • Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    
    return successful_results, best_overall[0]

def detailed_analysis(best_strategy):
    """Provide detailed analysis of the best strategy"""
    print("\n" + "="*60)
    print("DETAILED ANALYSIS OF BEST STRATEGY")
    print("="*60)
    
    strategy_name = best_strategy['strategy_name']
    params = best_strategy['params']
    metrics = best_strategy['metrics']
    
    print(f"Strategy: {strategy_name}")
    print(f"Parameters: {params}")
    
    # Risk analysis
    print(f"\nRisk Analysis:")
    if metrics['max_drawdown'] > -0.1:
        print("✅ Low drawdown risk")
    elif metrics['max_drawdown'] > -0.2:
        print("⚠️ Moderate drawdown risk")
    else:
        print("❌ High drawdown risk")
    
    # Return analysis
    print(f"\nReturn Analysis:")
    if metrics['total_return'] > 0.2:
        print("✅ Excellent returns")
    elif metrics['total_return'] > 0.1:
        print("✅ Good returns")
    elif metrics['total_return'] > 0.05:
        print("⚠️ Moderate returns")
    else:
        print("❌ Low returns")
    
    # Risk-adjusted performance
    print(f"\nRisk-Adjusted Performance:")
    if metrics['sharpe_ratio'] > 2.0:
        print("✅ Excellent risk-adjusted returns")
    elif metrics['sharpe_ratio'] > 1.0:
        print("✅ Good risk-adjusted returns")
    elif metrics['sharpe_ratio'] > 0.5:
        print("⚠️ Moderate risk-adjusted returns")
    else:
        print("❌ Poor risk-adjusted returns")
    
    # Trading frequency
    print(f"\nTrading Activity:")
    if metrics['num_trades'] > 50:
        print("✅ High trading frequency")
    elif metrics['num_trades'] > 20:
        print("✅ Moderate trading frequency")
    elif metrics['num_trades'] > 10:
        print("⚠️ Low trading frequency")
    else:
        print("❌ Very low trading frequency")
    
    # Win rate
    print(f"\nWin Rate Analysis:")
    if metrics['win_rate'] > 0.6:
        print("✅ High win rate")
    elif metrics['win_rate'] > 0.5:
        print("✅ Good win rate")
    elif metrics['win_rate'] > 0.4:
        print("⚠️ Moderate win rate")
    else:
        print("❌ Low win rate")

if __name__ == '__main__':
    # Run comprehensive backtest
    results, best_strategy = comprehensive_backtest()
    
    if best_strategy:
        # Provide detailed analysis
        detailed_analysis(best_strategy)
        
        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)
        print(f"Best strategy identified: {best_strategy['strategy_name']}")
        print("Use this strategy for live trading with proper risk management.")

