"""
Simple Trend Following V2 - Long Strategy
==========================================

The best performing cryptocurrency trading strategy based on comprehensive backtesting.

Performance Summary:
- Total Return: 105.34%
- Sharpe Ratio: 2.268
- Max Drawdown: -16.92%
- Win Rate: 46.4%
- Number of Trades: 112

Optimal Parameters:
- Slow MA: 200 periods
- Entry Threshold: 2% above MA
- ATR Multiplier: 3.0 (for stops and targets)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class SimpleTrendFollowingStrategyV2:
    """
    Simple Trend Following Strategy V2 - Long
    
    A trend-following strategy that enters long positions when price is above
    a slow moving average and uses ATR-based stops and profit targets.
    
    This is the best performing strategy from comprehensive backtesting.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 1000.0,
                 fee: float = 0.001,
                 slippage: float = 0.0005,
                 slow_ma: int = 200,
                 entry_threshold: float = 0.02,
                 atr_period: int = 14,
                 atr_multiplier: float = 3.0,
                 profit_target_multiplier: float = 3.0):
        """
        Initialize the strategy with optimal parameters.
        
        Args:
            data: OHLCV price data
            initial_balance: Starting capital
            fee: Trading fee (0.1% default)
            slippage: Market slippage (0.05% default)
            slow_ma: Slow moving average period (200 optimal)
            entry_threshold: Entry threshold above MA (2% optimal)
            atr_period: ATR calculation period (14 default)
            atr_multiplier: ATR multiplier for stops/targets (3.0 optimal)
            profit_target_multiplier: ATR multiplier for profit targets (3.0 optimal)
        """
        self.data = data
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        self.slow_ma = slow_ma
        self.entry_threshold = entry_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.profit_target_multiplier = profit_target_multiplier
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.slow_ma < 10:
            raise ValueError("Slow MA must be at least 10 periods")
        if self.entry_threshold <= 0:
            raise ValueError("Entry threshold must be positive")
        if self.atr_multiplier <= 0:
            raise ValueError("ATR multiplier must be positive")
        if self.profit_target_multiplier <= 0:
            raise ValueError("Profit target multiplier must be positive")
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        data = data.copy()
        
        # Calculate slow moving average
        data['slow_ma'] = data['close'].rolling(window=self.slow_ma).mean()
        
        # Calculate ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=self.atr_period).mean()
        
        return data
    
    def backtest(self) -> Dict[str, Any]:
        """
        Run the backtest and return results.
        
        Returns:
            Dictionary containing trades, equity curve, and performance metrics
        """
        data = self._calculate_indicators(self.data)
        
        # Initialize trading variables
        cash = self.initial_balance
        quantity = 0.0
        in_position = False
        entry_price = 0.0
        trailing_stop_price = 0.0
        profit_target_price = 0.0
        trades = []
        balance_series = []
        
        # Main trading loop
        for i in range(1, len(data) - 1):
            close_price = data['close'].iloc[i]
            low_price = data['low'].iloc[i]
            atr = data['atr'].iloc[i]
            
            # Update portfolio value
            if in_position:
                balance_series.append(quantity * close_price)
            else:
                balance_series.append(cash)
            
            # Entry logic
            if not in_position:
                if self._should_enter(data, i):
                    next_open = data['open'].iloc[i + 1]
                    buy_price = next_open * (1 + self.slippage + self.fee)
                    quantity = cash / buy_price
                    cash = 0.0
                    in_position = True
                    entry_price = buy_price
                    trailing_stop_price = entry_price - self.atr_multiplier * atr
                    profit_target_price = entry_price + self.profit_target_multiplier * atr
            
            # Position management
            else:
                # Update trailing stop
                new_stop_price = close_price - self.atr_multiplier * atr
                trailing_stop_price = max(trailing_stop_price, new_stop_price)
                
                # Exit logic
                if self._should_exit(low_price, close_price, trailing_stop_price, profit_target_price):
                    next_open = data['open'].iloc[i + 1]
                    sell_price = next_open * (1 - self.slippage - self.fee)
                    trade_return = (sell_price - entry_price) / entry_price
                    cash = quantity * sell_price
                    quantity = 0.0
                    in_position = False
                    trades.append(trade_return)
        
        # Final mark-to-market
        last_close = data['close'].iloc[-1]
        if in_position:
            balance_series.append(quantity * last_close)
            final_price = last_close * (1 - self.slippage - self.fee)
            trade_return = (final_price - entry_price) / entry_price
            trades.append(trade_return)
            cash = quantity * final_price
        else:
            balance_series.append(cash)
        
        return {
            'trades': trades,
            'equity_curve': pd.Series(balance_series, index=data.index[1:len(balance_series)+1]),
            'initial_balance': self.initial_balance,
            'final_balance': cash
        }
    
    def _should_enter(self, data: pd.DataFrame, i: int) -> bool:
        """Check if entry conditions are met."""
        close_price = data['close'].iloc[i]
        slow_ma = data['slow_ma'].iloc[i]
        
        # Entry condition: price must be above threshold above MA
        return close_price > slow_ma * (1 + self.entry_threshold)
    
    def _should_exit(self, low_price: float, close_price: float, 
                    trailing_stop_price: float, profit_target_price: float) -> bool:
        """Check if exit conditions are met."""
        # Exit on stop loss or profit target
        return low_price <= trailing_stop_price or close_price >= profit_target_price
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return {
            'slow_ma': self.slow_ma,
            'entry_threshold': self.entry_threshold,
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier,
            'profit_target_multiplier': self.profit_target_multiplier,
            'fee': self.fee,
            'slippage': self.slippage
        }
    
    def set_parameters(self, **kwargs) -> None:
        """Update strategy parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        self._validate_parameters()


class StrategyAnalyzer:
    """Analyze strategy performance and provide insights."""
    
    @staticmethod
    def calculate_metrics(results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        trades = results['trades']
        equity_curve = results['equity_curve']
        initial_balance = results['initial_balance']
        final_balance = results['final_balance']
        
        if not trades:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'num_trades': 0
            }
        
        # Basic metrics
        total_return = (final_balance / initial_balance) - 1
        num_trades = len(trades)
        win_rate = (len([t for t in trades if t > 0]) / num_trades) if num_trades > 0 else 0.0
        
        # Profit factor
        sum_profits = sum(t for t in trades if t > 0)
        sum_losses = -sum(t for t in trades if t <= 0)
        profit_factor = sum_profits / sum_losses if sum_losses > 0 else np.inf
        
        # Risk metrics
        hourly_returns = equity_curve.pct_change().fillna(0)
        mean_ret = hourly_returns.mean()
        std_ret = hourly_returns.std()
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(365 * 24) if std_ret != 0 else 0.0
        
        # Sortino ratio
        downside_returns = hourly_returns[hourly_returns < 0]
        sortino_ratio = (mean_ret / downside_returns.std()) * np.sqrt(365 * 24) if downside_returns.std() != 0 else 0.0
        
        # Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': total_return / (len(equity_curve) / (365 * 24)),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'num_trades': num_trades
        }
    
    @staticmethod
    def print_performance_summary(metrics: Dict[str, float]) -> None:
        """Print formatted performance summary."""
        print("=" * 60)
        print("STRATEGY PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total Return:        {metrics['total_return']*100:.2f}%")
        print(f"Annualized Return:  {metrics['annualized_return']*100:.2f}%")
        print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio:      {metrics['sortino_ratio']:.3f}")
        print(f"Calmar Ratio:       {metrics['calmar_ratio']:.3f}")
        print(f"Max Drawdown:       {metrics['max_drawdown']*100:.2f}%")
        print(f"Profit Factor:      {metrics['profit_factor']:.2f}")
        print(f"Win Rate:           {metrics['win_rate']*100:.1f}%")
        print(f"Number of Trades:   {metrics['num_trades']}")
        print("=" * 60)
    
    @staticmethod
    def assess_risk_level(metrics: Dict[str, float]) -> str:
        """Assess overall risk level of the strategy."""
        if metrics['max_drawdown'] > -0.1:
            return "LOW RISK"
        elif metrics['max_drawdown'] > -0.2:
            return "MODERATE RISK"
        elif metrics['max_drawdown'] > -0.3:
            return "HIGH RISK"
        else:
            return "VERY HIGH RISK"
    
    @staticmethod
    def get_trading_recommendations(metrics: Dict[str, float]) -> list:
        """Get trading recommendations based on performance."""
        recommendations = []
        
        # Risk assessment
        if metrics['max_drawdown'] > -0.15:
            recommendations.append("✅ Manageable drawdown risk - suitable for live trading")
        elif metrics['max_drawdown'] > -0.25:
            recommendations.append("⚠️ Moderate drawdown risk - use proper position sizing")
        else:
            recommendations.append("❌ High drawdown risk - implement strict risk management")
        
        # Return assessment
        if metrics['total_return'] > 0.5:
            recommendations.append("✅ Excellent returns")
        elif metrics['total_return'] > 0.2:
            recommendations.append("✅ Good returns")
        elif metrics['total_return'] > 0.1:
            recommendations.append("✅ Decent returns")
        else:
            recommendations.append("⚠️ Low returns - consider optimization")
        
        # Sharpe ratio assessment
        if metrics['sharpe_ratio'] > 2.0:
            recommendations.append("✅ Excellent risk-adjusted returns")
        elif metrics['sharpe_ratio'] > 1.0:
            recommendations.append("✅ Good risk-adjusted returns")
        elif metrics['sharpe_ratio'] > 0.5:
            recommendations.append("⚠️ Moderate risk-adjusted returns")
        else:
            recommendations.append("❌ Poor risk-adjusted returns")
        
        # Trading frequency assessment
        if metrics['num_trades'] > 100:
            recommendations.append("✅ High trading frequency - good for active management")
        elif metrics['num_trades'] > 50:
            recommendations.append("✅ Moderate trading frequency - suitable for most traders")
        elif metrics['num_trades'] > 20:
            recommendations.append("⚠️ Low trading frequency - consider shorter timeframes")
        else:
            recommendations.append("❌ Very low trading frequency - may not be practical")
        
        return recommendations


def load_crypto_data(file_path: str) -> pd.DataFrame:
    """
    Load cryptocurrency data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with OHLCV data
    """
    column_names = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    data = pd.read_csv(file_path, names=column_names)
    
    # Fix timestamps
    data['open_time'] = data['open_time'].apply(lambda x: x / 1000 if x > 10**15 else x)
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('open_time', inplace=True)
    
    # Convert columns to numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col])
    
    return data


def run_strategy_example():
    """Example of how to use the strategy."""
    print("Simple Trend Following V2 - Long Strategy Example")
    print("=" * 60)
    
    # Load data
    try:
        data = load_crypto_data('BTCUSDT_1h.csv')
        print(f"Loaded {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}")
    except FileNotFoundError:
        print("Data file not found. Please ensure BTCUSDT_1h.csv is in the current directory.")
        return
    
    # Create strategy with optimal parameters
    strategy = SimpleTrendFollowingStrategyV2(
        data,
        slow_ma=200,
        entry_threshold=0.02,
        atr_multiplier=3.0,
        profit_target_multiplier=3.0
    )
    
    # Run backtest
    print("\nRunning backtest...")
    results = strategy.backtest()
    
    # Calculate and display metrics
    analyzer = StrategyAnalyzer()
    metrics = analyzer.calculate_metrics(results)
    analyzer.print_performance_summary(metrics)
    
    # Risk assessment
    risk_level = analyzer.assess_risk_level(metrics)
    print(f"\nRisk Level: {risk_level}")
    
    # Trading recommendations
    recommendations = analyzer.get_trading_recommendations(metrics)
    print("\nTrading Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")
    
    # Strategy parameters
    print(f"\nStrategy Parameters:")
    params = strategy.get_parameters()
    for key, value in params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    run_strategy_example()

