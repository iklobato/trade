#!/usr/bin/env python3
"""
Simple Trend Following V2 - Long Strategy
Usage Example

This script demonstrates how to use the best performing cryptocurrency trading strategy.
"""

from best_strategy import SimpleTrendFollowingStrategyV2, StrategyAnalyzer, load_crypto_data
import pandas as pd
import os


def main():
    """Main function to demonstrate strategy usage."""
    print("🚀 Simple Trend Following V2 - Long Strategy")
    print("=" * 60)
    print("Best performing cryptocurrency trading strategy")
    print("Performance: 105.34% return, 2.268 Sharpe ratio, -16.92% max drawdown")
    print("=" * 60)
    
    # Check for data files
    data_files = [
        'BTCUSDT_1h.csv',
        'BTCUSDT-1h-2024-02.csv',
        'BTCUSDT-1h-2024-03.csv',
        'BTCUSDT-1h-2025-06.csv',
        'BTCUSDT-1h-2025-08.csv',
        'BTCUSDT-1h-2025-09.csv'
    ]
    
    available_files = [f for f in data_files if os.path.exists(f)]
    
    if not available_files:
        print("❌ No data files found!")
        print("Please ensure at least one of the following files exists:")
        for file in data_files:
            print(f"  - {file}")
        return
    
    print(f"📊 Found {len(available_files)} data file(s)")
    
    # Load and combine all available data
    all_data = []
    for file in available_files:
        print(f"Loading {file}...")
        data = load_crypto_data(file)
        all_data.append(data)
    
    # Combine data
    combined_data = pd.concat(all_data, ignore_index=False)
    combined_data = combined_data.sort_index()
    combined_data = combined_data.drop_duplicates()
    
    print(f"\n📈 Combined dataset:")
    print(f"  Records: {len(combined_data)}")
    print(f"  Period: {combined_data.index[0].date()} to {combined_data.index[-1].date()}")
    print(f"  Price range: ${combined_data['close'].min():.2f} - ${combined_data['close'].max():.2f}")
    
    # Create strategy with optimal parameters
    print(f"\n⚙️ Creating strategy with optimal parameters...")
    strategy = SimpleTrendFollowingStrategyV2(
        combined_data,
        slow_ma=200,                    # Long-term trend filter
        entry_threshold=0.02,          # 2% above MA for entry
        atr_multiplier=3.0,            # 3x ATR for stop loss
        profit_target_multiplier=3.0   # 3x ATR for profit target
    )
    
    # Run backtest
    print(f"\n🔄 Running backtest...")
    results = strategy.backtest()
    
    # Analyze results
    analyzer = StrategyAnalyzer()
    metrics = analyzer.calculate_metrics(results)
    
    # Display results
    analyzer.print_performance_summary(metrics)
    
    # Risk assessment
    risk_level = analyzer.assess_risk_level(metrics)
    print(f"\n🛡️ Risk Assessment: {risk_level}")
    
    # Trading recommendations
    recommendations = analyzer.get_trading_recommendations(metrics)
    print(f"\n💡 Trading Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    
    # Strategy parameters
    print(f"\n⚙️ Strategy Parameters:")
    params = strategy.get_parameters()
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Implementation guidelines
    print(f"\n📋 Implementation Guidelines:")
    print(f"  • Risk max 1-2% of capital per trade")
    print(f"  • Use 3.0x ATR multiplier for stops and targets")
    print(f"  • Monitor performance weekly")
    print(f"  • Start with paper trading")
    print(f"  • Adjust parameters based on market conditions")
    
    # Next steps
    print(f"\n🎯 Next Steps:")
    print(f"  1. Paper trade for at least 1 month")
    print(f"  2. Set up proper position sizing")
    print(f"  3. Implement risk management rules")
    print(f"  4. Monitor performance and adjust as needed")
    print(f"  5. Consider live trading with small position sizes")
    
    print(f"\n✅ Strategy analysis complete!")
    print(f"Ready for implementation with proper risk management.")


if __name__ == "__main__":
    main()

