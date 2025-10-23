"""
Performance metrics calculation for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for trading strategies.
    """
    
    def __init__(self):
        pass
    
    def calculate_all_metrics(self, equity_curve: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            equity_curve: DataFrame with equity curve data
            trades: DataFrame with trade data
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_return_metrics(equity_curve))
        metrics.update(self._calculate_risk_metrics(equity_curve))
        metrics.update(self._calculate_trade_metrics(trades))
        metrics.update(self._calculate_advanced_metrics(equity_curve, trades))
        
        return metrics
    
    def _calculate_return_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate return-based metrics."""
        if len(equity_curve) < 2:
            return {'total_return': 0.0, 'annualized_return': 0.0}
        
        equity_values = equity_curve['equity'].values
        initial_capital = equity_values[0]
        final_capital = equity_values[-1]
        
        # Total return
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Annualized return
        time_span = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / (365.25 * 24 * 3600)
        annualized_return = (1 + total_return) ** (1 / time_span) - 1 if time_span > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'final_capital': final_capital,
            'initial_capital': initial_capital
        }
    
    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        if len(equity_curve) < 2:
            return {
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0
            }
        
        equity_values = equity_curve['equity'].values
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized for minute data
        
        # Sharpe ratio
        risk_free_rate = 0.0  # Assume zero risk-free rate
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
        sharpe_ratio *= np.sqrt(252 * 24 * 60)  # Annualized
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = (equity_values - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        annualized_return = self._calculate_return_metrics(equity_curve)['annualized_return']
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Value at Risk (VaR) and Conditional VaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 else 0.0
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-based metrics."""
        if len(trades) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_holding_period': 0.0
            }
        
        trade_pnls = trades['net_pnl'].values
        winning_trades = trade_pnls[trade_pnls > 0]
        losing_trades = trade_pnls[trade_pnls < 0]
        
        # Basic trade metrics
        num_trades = len(trades)
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0
        
        # Profit factor
        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0.0
        
        # Average trade metrics
        avg_trade_return = np.mean(trade_pnls) if num_trades > 0 else 0.0
        avg_winning_trade = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        avg_losing_trade = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
        
        # Extreme trades
        largest_win = np.max(trade_pnls) if num_trades > 0 else 0.0
        largest_loss = np.min(trade_pnls) if num_trades > 0 else 0.0
        
        # Average holding period
        avg_holding_period = np.mean(trades['holding_period']) if 'holding_period' in trades.columns else 0.0
        
        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_holding_period': avg_holding_period,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _calculate_advanced_metrics(self, equity_curve: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced performance metrics."""
        if len(equity_curve) < 2 or len(trades) == 0:
            return {
                'sortino_ratio': 0.0,
                'omega_ratio': 0.0,
                'kelly_criterion': 0.0,
                'recovery_factor': 0.0,
                'expectancy': 0.0,
                'sqn': 0.0
            }
        
        equity_values = equity_curve['equity'].values
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0.0
        sortino_ratio *= np.sqrt(252 * 24 * 60)  # Annualized
        
        # Omega ratio
        threshold = 0.0
        positive_returns = returns[returns > threshold]
        negative_returns = returns[returns <= threshold]
        omega_ratio = np.sum(positive_returns - threshold) / abs(np.sum(negative_returns - threshold)) if len(negative_returns) > 0 else np.inf
        
        # Kelly criterion
        win_rate = self._calculate_trade_metrics(trades)['win_rate']
        avg_win = self._calculate_trade_metrics(trades)['avg_winning_trade']
        avg_loss = abs(self._calculate_trade_metrics(trades)['avg_losing_trade'])
        kelly_criterion = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0.0
        
        # Recovery factor
        total_return = self._calculate_return_metrics(equity_curve)['total_return']
        max_drawdown = self._calculate_risk_metrics(equity_curve)['max_drawdown']
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Expectancy
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        
        # System Quality Number (SQN)
        trade_pnls = trades['net_pnl'].values
        sqn = np.mean(trade_pnls) / np.std(trade_pnls) * np.sqrt(len(trade_pnls)) if np.std(trade_pnls) > 0 else 0.0
        
        return {
            'sortino_ratio': sortino_ratio,
            'omega_ratio': omega_ratio,
            'kelly_criterion': kelly_criterion,
            'recovery_factor': recovery_factor,
            'expectancy': expectancy,
            'sqn': sqn
        }
    
    def calculate_rolling_metrics(self, equity_curve: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            equity_curve: DataFrame with equity curve data
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        if len(equity_curve) < window:
            return pd.DataFrame()
        
        rolling_metrics = []
        
        for i in range(window, len(equity_curve)):
            window_data = equity_curve.iloc[i-window:i]
            
            metrics = self._calculate_return_metrics(window_data)
            metrics.update(self._calculate_risk_metrics(window_data))
            
            metrics['timestamp'] = equity_curve.index[i]
            rolling_metrics.append(metrics)
        
        return pd.DataFrame(rolling_metrics)
    
    def generate_performance_report(self, equity_curve: pd.DataFrame, trades: pd.DataFrame) -> str:
        """
        Generate a formatted performance report.
        
        Args:
            equity_curve: DataFrame with equity curve data
            trades: DataFrame with trade data
            
        Returns:
            Formatted report string
        """
        metrics = self.calculate_all_metrics(equity_curve, trades)
        
        report = f"""
PERFORMANCE REPORT
==================

RETURN METRICS
--------------
Total Return: {metrics['total_return']:.2%}
Annualized Return: {metrics['annualized_return']:.2%}
Final Capital: ${metrics['final_capital']:,.2f}

RISK METRICS
------------
Volatility: {metrics['volatility']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2%}
Calmar Ratio: {metrics['calmar_ratio']:.2f}
VaR (95%): {metrics['var_95']:.2%}
CVaR (95%): {metrics['cvar_95']:.2%}

TRADE METRICS
-------------
Number of Trades: {metrics['num_trades']}
Win Rate: {metrics['win_rate']:.2%}
Profit Factor: {metrics['profit_factor']:.2f}
Average Trade Return: {metrics['avg_trade_return']:.2%}
Average Winning Trade: ${metrics['avg_winning_trade']:.2f}
Average Losing Trade: ${metrics['avg_losing_trade']:.2f}
Largest Win: ${metrics['largest_win']:.2f}
Largest Loss: ${metrics['largest_loss']:.2f}
Average Holding Period: {metrics['avg_holding_period']:.1f} minutes

ADVANCED METRICS
----------------
Sortino Ratio: {metrics['sortino_ratio']:.2f}
Omega Ratio: {metrics['omega_ratio']:.2f}
Kelly Criterion: {metrics['kelly_criterion']:.2f}
Recovery Factor: {metrics['recovery_factor']:.2f}
Expectancy: ${metrics['expectancy']:.2f}
SQN: {metrics['sqn']:.2f}
"""
        
        return report
