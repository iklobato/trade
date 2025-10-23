import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backtester import Backtester
from enhanced_strategies import EnhancedTrendFollowingStrategy, EnhancedMLStrategy
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load BTC/USDT data"""
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

def create_performance_dashboard():
    """Create comprehensive performance dashboard"""
    data = load_data()
    
    # Define strategies to test
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
    
    # Run backtests and collect results
    results = {}
    equity_curves = {}
    
    for name, strategy in strategies.items():
        print(f"Running backtest for {name}...")
        backtester = Backtester(data, strategy)
        backtester.run()
        metrics = backtester.calculate_metrics()
        results[name] = metrics
        equity_curves[name] = backtester.results['equity_curve']
    
    # Create dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cryptocurrency Trading Strategy Performance Dashboard', fontsize=16)
    
    # 1. Equity Curves
    ax1 = axes[0, 0]
    for name, equity_curve in equity_curves.items():
        ax1.plot(equity_curve.index, equity_curve.values, label=name, linewidth=2)
    ax1.set_title('Equity Curves')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance Metrics Comparison
    ax2 = axes[0, 1]
    metrics_to_plot = ['sharpe_ratio', 'total_return', 'max_drawdown', 'profit_factor']
    strategy_names = list(results.keys())
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, strategy_name in enumerate(strategy_names):
        values = [results[strategy_name].get(metric, 0) for metric in metrics_to_plot]
        ax2.bar(x + i*width, values, width, label=strategy_name, alpha=0.8)
    
    ax2.set_title('Key Performance Metrics')
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Values')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(metrics_to_plot, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown Analysis
    ax3 = axes[0, 2]
    for name, equity_curve in equity_curves.items():
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=name)
    ax3.set_title('Drawdown Analysis')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Trade Distribution
    ax4 = axes[1, 0]
    for name, strategy_results in results.items():
        trades = strategy_results.get('trades', [])
        if trades:
            ax4.hist(trades, bins=20, alpha=0.6, label=name, density=True)
    ax4.set_title('Trade Return Distribution')
    ax4.set_xlabel('Trade Return')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Rolling Sharpe Ratio
    ax5 = axes[1, 1]
    window = 1000  # Rolling window for hourly data
    for name, equity_curve in equity_curves.items():
        returns = equity_curve.pct_change().fillna(0)
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(365*24)
        ax5.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=2)
    ax5.set_title(f'Rolling Sharpe Ratio ({window} periods)')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for name, metrics in results.items():
        summary_data.append([
            name,
            f"{metrics.get('total_return', 0)*100:.2f}%",
            f"{metrics.get('sharpe_ratio', 0):.3f}",
            f"{metrics.get('max_drawdown', 0)*100:.2f}%",
            f"{metrics.get('num_trades', 0)}",
            f"{metrics.get('win_rate', 0)*100:.1f}%"
        ])
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Strategy', 'Return', 'Sharpe', 'Max DD', 'Trades', 'Win Rate'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.show()
    
    return results, equity_curves

def create_risk_analysis():
    """Create detailed risk analysis"""
    data = load_data()
    
    # Test Enhanced Trend Following Strategy
    strategy = EnhancedTrendFollowingStrategy(
        data, slow_ma=200, entry_threshold=0.015, atr_multiplier=3.0,
        profit_target_multiplier=2.0, use_kelly_sizing=True
    )
    
    backtester = Backtester(data, strategy)
    backtester.run()
    results = backtester.results
    metrics = backtester.calculate_metrics()
    
    # Risk analysis
    equity_curve = results['equity_curve']
    trades = results['trades']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Risk Analysis - Enhanced Trend Following Strategy', fontsize=16)
    
    # 1. Value at Risk (VaR)
    ax1 = axes[0, 0]
    returns = equity_curve.pct_change().dropna()
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    ax1.hist(returns, bins=50, alpha=0.7, density=True)
    ax1.axvline(var_95, color='red', linestyle='--', label=f'VaR 95%: {var_95:.4f}')
    ax1.axvline(var_99, color='darkred', linestyle='--', label=f'VaR 99%: {var_99:.4f}')
    ax1.set_title('Value at Risk Analysis')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Maximum Drawdown Duration
    ax2 = axes[0, 1]
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    
    # Find drawdown periods
    in_drawdown = drawdown < 0
    drawdown_periods = []
    current_period_start = None
    
    for i, is_dd in enumerate(in_drawdown):
        if is_dd and current_period_start is None:
            current_period_start = i
        elif not is_dd and current_period_start is not None:
            drawdown_periods.append(i - current_period_start)
            current_period_start = None
    
    if current_period_start is not None:
        drawdown_periods.append(len(in_drawdown) - current_period_start)
    
    if drawdown_periods:
        ax2.hist(drawdown_periods, bins=20, alpha=0.7)
        ax2.set_title('Drawdown Duration Distribution')
        ax2.set_xlabel('Duration (hours)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
    
    # 3. Trade Analysis
    ax3 = axes[1, 0]
    if trades:
        win_trades = [t for t in trades if t > 0]
        loss_trades = [t for t in trades if t <= 0]
        
        ax3.scatter(range(len(win_trades)), win_trades, color='green', alpha=0.6, label='Wins')
        ax3.scatter(range(len(loss_trades)), loss_trades, color='red', alpha=0.6, label='Losses')
        ax3.set_title('Individual Trade Returns')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Return')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Risk Metrics Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate additional risk metrics
    volatility = returns.std() * np.sqrt(365*24)
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    risk_metrics = [
        ['Metric', 'Value'],
        ['Volatility (Annualized)', f'{volatility:.3f}'],
        ['Skewness', f'{skewness:.3f}'],
        ['Kurtosis', f'{kurtosis:.3f}'],
        ['VaR 95%', f'{var_95:.4f}'],
        ['VaR 99%', f'{var_99:.4f}'],
        ['Max Drawdown', f'{metrics.get("max_drawdown", 0):.4f}'],
        ['Sharpe Ratio', f'{metrics.get("sharpe_ratio", 0):.3f}'],
        ['Sortino Ratio', f'{metrics.get("sortino_ratio", 0):.3f}']
    ]
    
    table = ax4.table(cellText=risk_metrics,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Risk Metrics Summary')
    
    plt.tight_layout()
    plt.show()
    
    return metrics

def create_parameter_sensitivity_analysis():
    """Analyze parameter sensitivity"""
    data = load_data()
    
    # Test different parameter values
    param_ranges = {
        'slow_ma': [150, 200, 250, 300],
        'entry_threshold': [0.01, 0.015, 0.02, 0.025],
        'atr_multiplier': [2.0, 2.5, 3.0, 3.5],
        'profit_target_multiplier': [2.0, 2.5, 3.0, 3.5]
    }
    
    results = {}
    
    for param_name, param_values in param_ranges.items():
        param_results = []
        for value in param_values:
            try:
                if param_name == 'slow_ma':
                    strategy = EnhancedTrendFollowingStrategy(data, slow_ma=value)
                elif param_name == 'entry_threshold':
                    strategy = EnhancedTrendFollowingStrategy(data, entry_threshold=value)
                elif param_name == 'atr_multiplier':
                    strategy = EnhancedTrendFollowingStrategy(data, atr_multiplier=value)
                elif param_name == 'profit_target_multiplier':
                    strategy = EnhancedTrendFollowingStrategy(data, profit_target_multiplier=value)
                
                backtester = Backtester(data, strategy)
                backtester.run()
                metrics = backtester.calculate_metrics()
                
                if metrics:
                    param_results.append({
                        'value': value,
                        'sharpe': metrics.get('sharpe_ratio', 0),
                        'return': metrics.get('total_return', 0),
                        'drawdown': metrics.get('max_drawdown', 0)
                    })
            except Exception as e:
                print(f"Error testing {param_name}={value}: {e}")
        
        results[param_name] = param_results
    
    # Create sensitivity plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
    
    for i, (param_name, param_results) in enumerate(results.items()):
        ax = axes[i//2, i%2]
        
        if param_results:
            values = [r['value'] for r in param_results]
            sharpes = [r['sharpe'] for r in param_results]
            
            ax.plot(values, sharpes, 'o-', linewidth=2, markersize=8)
            ax.set_title(f'{param_name} vs Sharpe Ratio')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Sharpe Ratio')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    """Main function to create comprehensive dashboard"""
    print("Creating Performance Dashboard...")
    
    # Create main performance dashboard
    results, equity_curves = create_performance_dashboard()
    
    # Create risk analysis
    print("\nCreating Risk Analysis...")
    risk_metrics = create_risk_analysis()
    
    # Create parameter sensitivity analysis
    print("\nCreating Parameter Sensitivity Analysis...")
    sensitivity_results = create_parameter_sensitivity_analysis()
    
    print("\nDashboard creation complete!")
    
    return results, equity_curves, risk_metrics, sensitivity_results

if __name__ == '__main__':
    main()

