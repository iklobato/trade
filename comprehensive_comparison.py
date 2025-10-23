import pandas as pd
from backtester import Backtester
from strategies import SimpleTrendFollowingStrategyV2
from simple_enhanced_strategies import SimpleEnhancedTrendFollowingStrategy, SimpleMLStrategy

def comprehensive_strategy_comparison():
    """Compare all available strategies"""
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
    
    print("=== Comprehensive Strategy Comparison ===")
    print(f"Data period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total data points: {len(data)}")
    print(f"Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    
    # Define strategies to test
    strategies = {
        'Original SimpleTrendFollowingStrategyV2': SimpleTrendFollowingStrategyV2(
            data, slow_ma=200, entry_threshold=0.015, atr_multiplier=3.0,
            profit_target_multiplier=2.0
        ),
        'Simple Enhanced Trend Following (Kelly)': SimpleEnhancedTrendFollowingStrategy(
            data, slow_ma=30, entry_threshold=0.015, atr_multiplier=2.5,
            profit_target_multiplier=2.0, use_kelly_sizing=True
        ),
        'Simple Enhanced Trend Following (No Kelly)': SimpleEnhancedTrendFollowingStrategy(
            data, slow_ma=30, entry_threshold=0.015, atr_multiplier=2.5,
            profit_target_multiplier=2.0, use_kelly_sizing=False
        ),
        'Simple ML Strategy': SimpleMLStrategy(
            data, trend_ma=30, short_ma=10, rsi_period=14, rsi_lower=30,
            rsi_exit=70, profit_target=0.10, prediction_threshold=0.015
        )
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n--- Testing {name} ---")
        try:
            backtester = Backtester(data, strategy)
            backtester.run()
            metrics = backtester.calculate_metrics()
            
            if metrics and metrics['num_trades'] > 0:
                results[name] = metrics
                print(f"✓ Successfully tested - {metrics['num_trades']} trades")
                print(f"  Total Return: {metrics['total_return']*100:.2f}%")
                print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
                print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
            else:
                print("✗ No trades generated")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Create comparison summary
    if results:
        print("\n" + "="*80)
        print("STRATEGY COMPARISON SUMMARY")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for name, metrics in results.items():
            comparison_data.append({
                'Strategy': name,
                'Total Return (%)': f"{metrics['total_return']*100:.2f}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                'Max Drawdown (%)': f"{metrics['max_drawdown']*100:.2f}",
                'Win Rate (%)': f"{metrics['win_rate']*100:.1f}",
                'Num Trades': metrics['num_trades'],
                'Profit Factor': f"{metrics['profit_factor']:.2f}",
                'Calmar Ratio': f"{metrics['calmar_ratio']:.3f}"
            })
        
        # Print formatted table
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Find best performing strategies
        print("\n" + "="*50)
        print("BEST PERFORMING STRATEGIES")
        print("="*50)
        
        # Best Sharpe Ratio
        best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
        print(f"Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
        
        # Best Total Return
        best_return = max(results.items(), key=lambda x: x[1]['total_return'])
        print(f"Best Total Return: {best_return[0]} ({best_return[1]['total_return']*100:.2f}%)")
        
        # Best Risk-Adjusted Return (Calmar Ratio)
        best_calmar = max(results.items(), key=lambda x: x[1]['calmar_ratio'])
        print(f"Best Calmar Ratio: {best_calmar[0]} ({best_calmar[1]['calmar_ratio']:.3f})")
        
        # Lowest Drawdown
        lowest_dd = min(results.items(), key=lambda x: x[1]['max_drawdown'])
        print(f"Lowest Drawdown: {lowest_dd[0]} ({lowest_dd[1]['max_drawdown']*100:.2f}%)")
        
        # Most Trades
        most_trades = max(results.items(), key=lambda x: x[1]['num_trades'])
        print(f"Most Active: {most_trades[0]} ({most_trades[1]['num_trades']} trades)")
        
        print("\n" + "="*50)
        print("RECOMMENDATIONS")
        print("="*50)
        
        # Analyze results and provide recommendations
        if best_sharpe[1]['sharpe_ratio'] > 2.0:
            print(f"✓ {best_sharpe[0]} shows excellent risk-adjusted returns")
        
        if best_return[1]['total_return'] > 0.1:
            print(f"✓ {best_return[0]} shows strong absolute returns")
        
        if lowest_dd[1]['max_drawdown'] > -0.2:
            print(f"⚠️ All strategies show significant drawdowns - consider risk management improvements")
        
        if most_trades[1]['num_trades'] < 20:
            print(f"⚠️ Low trade frequency may indicate overly restrictive entry conditions")
        
        print("\nNext steps:")
        print("1. Optimize the best-performing strategy parameters")
        print("2. Implement additional risk management features")
        print("3. Test on different timeframes and assets")
        print("4. Consider ensemble methods combining multiple strategies")
    
    else:
        print("\nNo strategies generated trades. Consider:")
        print("1. Relaxing entry conditions")
        print("2. Using shorter moving average periods")
        print("3. Adjusting threshold parameters")
    
    return results

def parameter_sensitivity_test():
    """Test parameter sensitivity for the best strategy"""
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
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
    
    # Test different parameters for Simple Enhanced Trend Following Strategy
    param_tests = [
        {'slow_ma': 20, 'entry_threshold': 0.01, 'atr_multiplier': 2.0},
        {'slow_ma': 30, 'entry_threshold': 0.015, 'atr_multiplier': 2.5},
        {'slow_ma': 40, 'entry_threshold': 0.02, 'atr_multiplier': 3.0},
        {'slow_ma': 20, 'entry_threshold': 0.015, 'atr_multiplier': 2.5},
        {'slow_ma': 30, 'entry_threshold': 0.01, 'atr_multiplier': 2.0},
    ]
    
    sensitivity_results = []
    
    for i, params in enumerate(param_tests):
        print(f"\nTest {i+1}: {params}")
        try:
            strategy = SimpleEnhancedTrendFollowingStrategy(data, **params, use_kelly_sizing=True)
            backtester = Backtester(data, strategy)
            backtester.run()
            metrics = backtester.calculate_metrics()
            
            if metrics and metrics['num_trades'] > 0:
                sensitivity_results.append({
                    'params': params,
                    'sharpe': metrics['sharpe_ratio'],
                    'return': metrics['total_return'],
                    'drawdown': metrics['max_drawdown'],
                    'trades': metrics['num_trades']
                })
                print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}, Return: {metrics['total_return']*100:.2f}%, Trades: {metrics['num_trades']}")
            else:
                print("  No trades generated")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    if sensitivity_results:
        print(f"\nParameter sensitivity analysis completed with {len(sensitivity_results)} successful tests")
        
        # Find best parameters
        best_params = max(sensitivity_results, key=lambda x: x['sharpe'])
        print(f"\nBest parameters: {best_params['params']}")
        print(f"Best Sharpe ratio: {best_params['sharpe']:.3f}")
    
    return sensitivity_results

if __name__ == '__main__':
    # Run comprehensive comparison
    results = comprehensive_strategy_comparison()
    
    # Run parameter sensitivity analysis
    sensitivity_results = parameter_sensitivity_test()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

