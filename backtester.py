import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, data, strategy, initial_balance=1000.0, fee=0.001, slippage=0.0005):
        self.data = data
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        self.results = None

    def run(self):
        self.strategy.data = self.data
        self.strategy.initial_balance = self.initial_balance
        self.strategy.fee = self.fee
        self.strategy.slippage = self.slippage
        self.results = self.strategy.backtest()

    def plot_results(self):
        if self.results is None:
            print("Please run the backtest first.")
            return

        equity_curve = self.results['equity_curve']
        equity_curve.plot(title='Equity Curve')
        plt.show()

    def calculate_metrics(self):
        if self.results is None:
            print("Please run the backtest first.")
            return

        trades = self.results['trades']
        equity_curve = self.results['equity_curve']
        initial_balance = self.results['initial_balance']
        final_balance = self.results['final_balance']

        total_return = (final_balance / initial_balance) - 1
        num_trades = len(trades)
        win_rate = (len([t for t in trades if t > 0]) / num_trades) if num_trades > 0 else 0.0
        sum_profits = sum(t for t in trades if t > 0)
        sum_losses = -sum(t for t in trades if t <= 0)
        profit_factor = sum_profits / sum_losses if sum_losses > 0 else np.inf

        hourly_returns = equity_curve.pct_change().fillna(0)
        mean_ret = hourly_returns.mean()
        std_ret = hourly_returns.std()
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(365 * 24) if std_ret != 0 else 0.0

        downside_returns = hourly_returns[hourly_returns < 0]
        sortino_ratio = (mean_ret / downside_returns.std()) * np.sqrt(365 * 24) if downside_returns.std() != 0 else 0.0

        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        max_drawdown = drawdown.min()

        calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        return {
            'total_return': total_return,
            'annualized_return': total_return / (len(self.data) / (365 * 24)),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'num_trades': num_trades,
        }
