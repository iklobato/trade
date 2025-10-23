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

def practical_strategy_backtest():
    """Focus on strategies with practical trade frequency (10+ trades)"""
    print("="*80)
    print("PRACTICAL STRATEGY BACKTEST - FOCUSING ON ACTIVE STRATEGIES")
    print("="*80)
    
    # Load all available data
    data = load_all_data()
    if data is None:
        return
    
    # Define strategies with parameters optimized for more trades
    strategies_to_test = [
        # Original strategies with adjusted parameters
        {
            'class': TrendFollowingStrategy,
            'name': 'Trend Following (SMA) - Short',
            'params': {'fast_ma': 20, 'slow_ma': 50, 'atr_multiplier': 2.0}
        },
        {
            'class': TrendFollowingEMAStrategy,
            'name': 'Trend Following (EMA) - Short',
            'params': {'fast_ema': 20, 'slow_ema': 50, 'atr_multiplier': 2.0}
        },
        {
            'class': SimpleTrendFollowingStrategy,
            'name': 'Simple Trend Following - Short',
            'params': {'slow_ma': 50, 'entry_threshold': 0.01, 'atr_multiplier': 2.0}
        },
        {
            'class': SimpleTrendFollowingStrategyV2,
            'name': 'Simple Trend Following V2 - Short',
            'params': {'slow_ma': 50, 'entry_threshold': 0.01, 'atr_multiplier': 2.0, 'profit_target_multiplier': 2.0}
        },
        {
            'class': SimpleTrendFollowingStrategyV2,
            'name': 'Simple Trend Following V2 - Medium',
            'params': {'slow_ma': 100, 'entry_threshold': 0.015, 'atr_multiplier': 2.5, 'profit_target_multiplier': 2.5}
        },
        {
            'class': SimpleTrendFollowingStrategyV2,
            'name': 'Simple Trend Following V2 - Long',
            'params': {'slow_ma': 200, 'entry_threshold': 0.02, 'atr_multiplier': 3.0, 'profit_target_multiplier': 3.0}
        },
        
        # Enhanced strategies
        {
            'class': SimpleEnhancedTrendFollowingStrategy,
            'name': 'Enhanced Trend Following (Kelly) - Short',
            'params': {'slow_ma': 20, 'entry_threshold': 0.01, 'atr_multiplier': 2.0, 'profit_target_multiplier': 2.0, 'use_kelly_sizing': True}
        },
        {
            'class': SimpleEnhancedTrendFollowingStrategy,
            'name': 'Enhanced Trend Following (No Kelly) - Short',
            'params': {'slow_ma': 20, 'entry_threshold': 0.01, 'atr_multiplier': 2.0, 'profit_target_multiplier': 2.0, 'use_kelly_sizing': False}
        },
        {
            'class': SimpleEnhancedTrendFollowingStrategy,
            'name': 'Enhanced Trend Following (No Kelly) - Medium',
            'params': {'slow_ma': 30, 'entry_threshold': 0.015, 'atr_multiplier': 2.5, 'profit_target_multiplier': 2.5, 'use_kelly_sizing': False}
        }
    ]
    
    print(f"\nTesting {len(strategies_to_test)} strategies with practical parameters...")
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
    
    # Filter for practical strategies (10+ trades)
    practical_results = [r for r in results if r['success'] and r['metrics']['num_trades'] >= 10]
    
    if not practical_results:
        print("\nNo strategies generated 10+ trades!")
        return
    
    print("\n" + "="*80)
    print("PRACTICAL STRATEGY PERFORMANCE COMPARISON (10+ TRADES)")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    for result in practical_results:
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
    
    # Find best practical strategies
    print("\n" + "="*60)
    print("BEST PRACTICAL STRATEGIES")
    print("="*60)
    
    # Best by different metrics
    best_sharpe = max(practical_results, key=lambda x: x['metrics']['sharpe_ratio'])
    best_return = max(practical_results, key=lambda x: x['metrics']['total_return'])
    best_calmar = max(practical_results, key=lambda x: x['metrics']['calmar_ratio'])
    lowest_drawdown = min(practical_results, key=lambda x: x['metrics']['max_drawdown'])
    most_trades = max(practical_results, key=lambda x: x['metrics']['num_trades'])
    
    print(f"🏆 Best Sharpe Ratio: {best_sharpe['strategy_name']}")
    print(f"   Sharpe: {best_sharpe['metrics']['sharpe_ratio']:.3f}, Return: {best_sharpe['metrics']['total_return']*100:.2f}%, Trades: {best_sharpe['metrics']['num_trades']}")
    
    print(f"\n💰 Best Total Return: {best_return['strategy_name']}")
    print(f"   Return: {best_return['metrics']['total_return']*100:.2f}%, Sharpe: {best_return['metrics']['sharpe_ratio']:.3f}, Trades: {best_return['metrics']['num_trades']}")
    
    print(f"\n📊 Best Calmar Ratio: {best_calmar['strategy_name']}")
    print(f"   Calmar: {best_calmar['metrics']['calmar_ratio']:.3f}, Return: {best_calmar['metrics']['total_return']*100:.2f}%, Trades: {best_calmar['metrics']['num_trades']}")
    
    print(f"\n🛡️ Lowest Drawdown: {lowest_drawdown['strategy_name']}")
    print(f"   Drawdown: {lowest_drawdown['metrics']['max_drawdown']*100:.2f}%, Return: {lowest_drawdown['metrics']['total_return']*100:.2f}%, Trades: {lowest_drawdown['metrics']['num_trades']}")
    
    print(f"\n📈 Most Active: {most_trades['strategy_name']}")
    print(f"   Trades: {most_trades['metrics']['num_trades']}, Return: {most_trades['metrics']['total_return']*100:.2f}%, Sharpe: {most_trades['metrics']['sharpe_ratio']:.3f}")
    
    # Overall recommendation for practical trading
    print("\n" + "="*60)
    print("PRACTICAL TRADING RECOMMENDATION")
    print("="*60)
    
    # Score strategies based on practical trading criteria
    scored_strategies = []
    for result in practical_results:
        metrics = result['metrics']
        # Weight more heavily on risk-adjusted returns and drawdown control
        score = (
            metrics['sharpe_ratio'] * 0.4 +  # Risk-adjusted returns (most important)
            metrics['total_return'] * 5 * 0.25 +  # Absolute returns
            (1 + metrics['max_drawdown']) * 0.25 +  # Drawdown control (higher is better)
            metrics['win_rate'] * 0.1  # Win rate
        )
        scored_strategies.append((result, score))
    
    best_practical = max(scored_strategies, key=lambda x: x[1])
    
    print(f"🥇 BEST PRACTICAL STRATEGY: {best_practical[0]['strategy_name']}")
    print(f"   Practical Score: {best_practical[1]:.3f}")
    print(f"   Parameters: {best_practical[0]['params']}")
    
    metrics = best_practical[0]['metrics']
    print(f"\n   Performance Summary:")
    print(f"   • Total Return: {metrics['total_return']*100:.2f}%")
    print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"   • Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"   • Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"   • Number of Trades: {metrics['num_trades']}")
    print(f"   • Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   • Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    
    # Trading recommendations
    print(f"\n   Trading Recommendations:")
    if metrics['sharpe_ratio'] > 2.0:
        print("   ✅ Excellent for live trading")
    elif metrics['sharpe_ratio'] > 1.0:
        print("   ✅ Good for live trading")
    else:
        print("   ⚠️ Consider additional optimization")
    
    if metrics['max_drawdown'] > -0.15:
        print("   ✅ Manageable drawdown risk")
    elif metrics['max_drawdown'] > -0.25:
        print("   ⚠️ Moderate drawdown risk - use position sizing")
    else:
        print("   ❌ High drawdown risk - implement strict risk management")
    
    if metrics['num_trades'] > 50:
        print("   ✅ Good trading frequency for active management")
    elif metrics['num_trades'] > 20:
        print("   ✅ Adequate trading frequency")
    else:
        print("   ⚠️ Low trading frequency - consider shorter timeframes")
    
    return practical_results, best_practical[0]

if __name__ == '__main__':
    # Run practical strategy backtest
    results, best_strategy = practical_strategy_backtest()
    
    if best_strategy:
        print("\n" + "="*80)
        print("PRACTICAL BACKTEST COMPLETE")
        print("="*80)
        print(f"Best practical strategy: {best_strategy['strategy_name']}")
        print("This strategy is recommended for live trading with proper risk management.")
        print("\nNext steps:")
        print("1. Implement the recommended strategy with proper position sizing")
        print("2. Set up risk management rules (max 2% risk per trade)")
        print("3. Monitor performance and adjust parameters as needed")
        print("4. Consider paper trading before live implementation")

