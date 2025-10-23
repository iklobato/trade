"""
trend_mean_reversion_strategy.py
===============================

This script implements a trend-filtered mean-reversion strategy with an
ATR-based trailing stop-loss. The goal is to achieve a high win rate
while improving risk-adjusted returns. The strategy is inspired by
successful crypto bots that trade in the direction of the prevailing
trend and use dynamic stops to manage risk.

Key features:

* **Trend filter:** A 150-day simple moving average (SMA) defines the
  long-term trend. Long trades are allowed only when the current
  closing price is above this SMA.
* **Oversold entry:** Within an uptrend, the strategy waits for a
  pullback where either the RSI(14) drops below 35 or the price
  closes below the lower Bollinger band (20-day SMA ± 2×STD). A
  long position is then opened at the next day’s open price.
* **Exit rules:** Positions are closed when any of the following
  conditions is met: (1) the price has risen at least 15 % above the
  entry price; (2) RSI rises above 65; (3) the close reverts to or
  above the 20-day moving average; or (4) the price hits the ATR-based
  trailing stop-loss. All exits occur at the next day’s open. Fees
  (0.1 % per side) and slippage (0.05 %) are applied.

To run the strategy simply execute:

    python trend_mean_reversion_strategy.py

You can adjust parameters near the bottom of the file to explore
different trend periods, RSI thresholds, profit targets, or stop-loss
levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def load_data(file_path: str = 'Bitstamp_ETHUSD_d.csv') -> pd.DataFrame:
    """Load the historical price data and sort it chronologically."""
    try:
        df = pd.read_csv(file_path, skiprows=1)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        print("Please download the dataset from https://www.cryptodatadownload.com/data/bitstamp/")
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators needed for the strategy.

    Adds columns for long-term SMA (100, 150, 200 day), 20-day SMA and
    its standard deviation (for Bollinger bands), RSI(14), and ATR(14).
    RSI is computed using a simple moving average of gains and losses.

    Args:
        df: Price DataFrame with 'open', 'high', 'low', 'close'.

    Returns:
        DataFrame with additional indicator columns.
    """
    data = df.copy()
    # Long-term trend filters
    data['SMA_long100'] = data['close'].rolling(window=100).mean()
    data['SMA_long150'] = data['close'].rolling(window=150).mean()
    data['SMA_long200'] = data['close'].rolling(window=200).mean()
    # Bollinger bands (20-day SMA ± 2×STD)
    data['SMA_short'] = data['close'].rolling(window=20).mean()
    data['STD_short'] = data['close'].rolling(window=20).std()
    data['BB_lower'] = data['SMA_short'] - 2 * data['STD_short']
    data['BB_upper'] = data['SMA_short'] + 2 * data['STD_short']
    # RSI(14)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - 100 / (1 + rs)
    # ATR(14)
    high_low = data['high'] - data['low']
    high_close_prev = (data['high'] - data['close'].shift()).abs()
    low_close_prev = (data['low'] - data['close'].shift()).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=14).mean()
    return data


def backtest_strategy(
    df: pd.DataFrame,
    trend_col: str = 'SMA_long150',
    rsi_lower: float = 35,
    rsi_exit: float = 65,
    profit_target: float = 0.15,
    atr_multiplier: float = 2.0,
    initial_balance: float = 1000.0,
    slippage: float = 0.0005,
    fee: float = 0.001,
) -> Dict[str, float]:
    """Back-test the trend-filtered mean-reversion with ATR trailing stop.

    Args:
        df: DataFrame with price and indicator columns.
        trend_col: Column name for the long-term SMA trend filter.
        rsi_lower: RSI threshold for oversold entry.
        rsi_exit: RSI threshold for overbought exit.
        profit_target: Fractional gain for take-profit.
        atr_multiplier: Multiplier for ATR to set the trailing stop.
        initial_balance: Starting capital.
        slippage: Slippage applied to entry and exit prices.
        fee: Exchange fee applied to entry and exit prices.

    Returns:
        Dictionary with performance metrics.
    """
    cash = initial_balance
    quantity = 0.0
    in_position = False
    entry_price = 0.0
    trailing_stop_price = 0.0
    trades: List[float] = []
    balance_series: List[float] = []

    for i in range(len(df) - 1):
        close_price = df['close'].iloc[i]
        low_price = df['low'].iloc[i]
        atr = df['ATR'].iloc[i]

        if in_position:
            balance_series.append(quantity * close_price)
        else:
            balance_series.append(cash)

        # Entry logic
        if not in_position:
            sma_trend = df[trend_col].iloc[i]
            bb_lower = df['BB_lower'].iloc[i]
            rsi = df['RSI'].iloc[i]
            if (
                not np.isnan(sma_trend)
                and not np.isnan(bb_lower)
                and not np.isnan(rsi)
                and close_price > sma_trend
                and (close_price < bb_lower or rsi < rsi_lower)
            ):
                next_open = df['open'].iloc[i + 1]
                buy_price = next_open * (1 + slippage + fee)
                quantity = cash / buy_price
                cash = 0.0
                in_position = True
                entry_price = buy_price
                trailing_stop_price = entry_price - atr_multiplier * atr
        else:
            # Update trailing stop
            new_stop_price = close_price - atr_multiplier * atr
            trailing_stop_price = max(trailing_stop_price, new_stop_price)

            # Evaluate exit conditions
            exit_flag = False
            gain = (close_price - entry_price) / entry_price
            rsi = df['RSI'].iloc[i]
            sma_short = df['SMA_short'].iloc[i]

            if gain >= profit_target:
                exit_flag = True
            if rsi >= rsi_exit:
                exit_flag = True
            if not np.isnan(sma_short) and close_price >= sma_short:
                exit_flag = True
            if low_price <= trailing_stop_price:
                exit_flag = True

            if exit_flag:
                next_open = df['open'].iloc[i + 1]
                sell_price = next_open * (1 - slippage - fee)
                trade_return = (sell_price - entry_price) / entry_price
                cash = quantity * sell_price
                quantity = 0.0
                in_position = False
                trades.append(trade_return)

    # Final day mark-to-market
    last_close = df['close'].iloc[-1]
    if in_position:
        balance_series.append(quantity * last_close)
        final_price = last_close * (1 - slippage - fee)
        trade_return = (final_price - entry_price) / entry_price
        trades.append(trade_return)
        cash = quantity * final_.price
    else:
        balance_series.append(cash)

    # Compute performance metrics
    equity = pd.Series(balance_series)
    daily_returns = equity.pct_change().fillna(0)
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(365) if std_ret != 0 else 0.0
    total_return = (cash / initial_balance) - 1
    num_trades = len(trades)
    win_rate = (len([t for t in trades if t > 0]) / num_trades) if num_trades > 0 else 0.0
    sum_profits = sum(t for t in trades if t > 0)
    sum_losses = -sum(t for t in trades if t <= 0)
    profit_factor = sum_profits / sum_losses if sum_losses > 0 else np.inf
    running_max = equity.cummax()
    max_drawdown = (equity / running_max - 1).min()

    return {
        'final_balance': cash,
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
    }


def main() -> None:
    df = load_data('Bitstamp_ETHUSD_d.csv')
    if df.empty:
        return
    df = compute_indicators(df)

    # Baseline Strategy (from user)
    print("Baseline Mean-Reversion Strategy Summary")
    print("=" * 50)
    baseline_result = backtest_strategy(
        df,
        trend_col='SMA_long150',
        rsi_lower=35,
        rsi_exit=65,
        profit_target=0.15,
        atr_multiplier=0,  # Effectively a fixed stop_loss of 0
        initial_balance=1000.0,
        slippage=0.0005,
        fee=0.001,
    )
    print(f"Final balance:            ${baseline_result['final_balance']:.2f}")
    print(f"Total return (ROI):       {baseline_result['total_return'] * 100:.2f}%")
    print(f"Number of trades:         {baseline_result['num_trades'] }")
    print(f"Win rate:                 {baseline_result['win_rate'] * 100:.2f}%")
    print(f"Profit factor:            {baseline_result['profit_factor']:.2f}")
    print(f"Sharpe ratio:             {baseline_result['sharpe_ratio']:.2f}")
    print(f"Max drawdown:             {baseline_result['max_drawdown'] * 100:.2f}%")
    print("\n" + "=" * 50 + "\n")


    # Optimized Strategy
    trend_col = 'SMA_long150'
    rsi_lower = 35
    rsi_exit = 65
    profit_target = 0.15
    atr_multiplier = 2.0

    result = backtest_strategy(
        df,
        trend_col=trend_col,
        rsi_lower=rsi_lower,
        rsi_exit=rsi_exit,
        profit_target=profit_target,
        atr_multiplier=atr_multiplier,
        initial_balance=1000.0,
        slippage=0.0005,
        fee=0.001,
    )
    print("Optimized Strategy with ATR Trailing Stop")
    print("=" * 50)
    print(f"Trend SMA period:         {trend_col}")
    print(f"RSI entry threshold:      {rsi_lower}")
    print(f"RSI exit threshold:       {rsi_exit}")
    print(f"Profit target:            {profit_target * 100:.1f}%")
    print(f"ATR Multiplier:           {atr_multiplier}")
    print(f"Initial balance:          $1000.00")
    print(f"Final balance:            ${result['final_balance']:.2f}")
    print(f"Total return (ROI):       {result['total_return'] * 100:.2f}%")
    print(f"Number of trades:         {result['num_trades']}")
    print(f"Win rate:                 {result['win_rate'] * 100:.2f}%")
    print(f"Profit factor:            {result['profit_factor']:.2f}")
    print(f"Sharpe ratio:             {result['sharpe_ratio']:.2f}")
    print(f"Max drawdown:             {result['max_drawdown'] * 100:.2f}%")


if __name__ == '__main__':
    main()