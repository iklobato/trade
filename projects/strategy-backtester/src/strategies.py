import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

class Strategy:
    def __init__(self, data, initial_balance=1000.0, fee=0.001, slippage=0.0005):
        self.data = data
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage

    def backtest(self):
        raise NotImplementedError

class TrendFollowingStrategy(Strategy):
    def __init__(self, data, initial_balance=1000.0, fee=0.001, slippage=0.0005, fast_ma=50, slow_ma=200, atr_period=14, atr_multiplier=2.0):
        super().__init__(data, initial_balance, fee, slippage)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def backtest(self):
        data = self.data.copy()
        data['fast_ma'] = data['close'].rolling(window=self.fast_ma).mean()
        data['slow_ma'] = data['close'].rolling(window=self.slow_ma).mean()
        
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=self.atr_period).mean()

        cash = self.initial_balance
        quantity = 0.0
        in_position = False
        entry_price = 0.0
        trailing_stop_price = 0.0
        trades = []
        balance_series = []

        for i in range(len(data) - 1):
            close_price = data['close'].iloc[i]
            low_price = data['low'].iloc[i]
            atr = data['atr'].iloc[i]

            if in_position:
                balance_series.append(quantity * close_price)
            else:
                balance_series.append(cash)

            # Entry logic
            if not in_position:
                if data['fast_ma'].iloc[i] > data['slow_ma'].iloc[i] and data['fast_ma'].iloc[i-1] <= data['slow_ma'].iloc[i-1]:
                    next_open = data['open'].iloc[i + 1]
                    buy_price = next_open * (1 + self.slippage + self.fee)
                    quantity = cash / buy_price
                    cash = 0.0
                    in_position = True
                    entry_price = buy_price
                    trailing_stop_price = entry_price - self.atr_multiplier * atr
            else:
                # Update trailing stop
                new_stop_price = close_price - self.atr_multiplier * atr
                trailing_stop_price = max(trailing_stop_price, new_stop_price)

                # Exit logic
                if low_price <= trailing_stop_price:
                    next_open = data['open'].iloc[i + 1]
                    sell_price = next_open * (1 - self.slippage - self.fee)
                    trade_return = (sell_price - entry_price) / entry_price
                    cash = quantity * sell_price
                    quantity = 0.0
                    in_position = False
                    trades.append(trade_return)

        # Final day mark-to-market
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
            'equity_curve': pd.Series(balance_series, index=data.index[:len(balance_series)]),
            'initial_balance': self.initial_balance,
            'final_balance': cash
        }

class TrendFollowingEMAStrategy(Strategy):
    def __init__(self, data, initial_balance=1000.0, fee=0.001, slippage=0.0005, fast_ema=50, slow_ema=200, atr_period=14, atr_multiplier=2.0):
        super().__init__(data, initial_balance, fee, slippage)
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def backtest(self):
        data = self.data.copy()
        data['fast_ema'] = data['close'].ewm(span=self.fast_ema, adjust=False).mean()
        data['slow_ema'] = data['close'].ewm(span=self.slow_ema, adjust=False).mean()
        
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=self.atr_period).mean()

        cash = self.initial_balance
        quantity = 0.0
        in_position = False
        entry_price = 0.0
        trailing_stop_price = 0.0
        trades = []
        balance_series = []

        for i in range(1, len(data) - 1):
            close_price = data['close'].iloc[i]
            low_price = data['low'].iloc[i]
            atr = data['atr'].iloc[i]

            if in_position:
                balance_series.append(quantity * close_price)
            else:
                balance_series.append(cash)

            # Entry logic
            if not in_position:
                # Crossover entry
                if data['fast_ema'].iloc[i] > data['slow_ema'].iloc[i] and data['fast_ema'].iloc[i-1] <= data['slow_ema'].iloc[i-1]:
                    next_open = data['open'].iloc[i + 1]
                    buy_price = next_open * (1 + self.slippage + self.fee)
                    quantity = cash / buy_price
                    cash = 0.0
                    in_position = True
                    entry_price = buy_price
                    trailing_stop_price = entry_price - self.atr_multiplier * atr
                # Pullback entry
                elif data['fast_ema'].iloc[i] > data['slow_ema'].iloc[i] and data['low'].iloc[i] < data['fast_ema'].iloc[i] and data['close'].iloc[i] > data['fast_ema'].iloc[i]:
                    next_open = data['open'].iloc[i + 1]
                    buy_price = next_open * (1 + self.slippage + self.fee)
                    quantity = cash / buy_price
                    cash = 0.0
                    in_position = True
                    entry_price = buy_price
                    trailing_stop_price = entry_price - self.atr_multiplier * atr
            else:
                # Update trailing stop
                new_stop_price = close_price - self.atr_multiplier * atr
                trailing_stop_price = max(trailing_stop_price, new_stop_price)

                # Exit logic
                if low_price <= trailing_stop_price:
                    next_open = data['open'].iloc[i + 1]
                    sell_price = next_open * (1 - self.slippage - self.fee)
                    trade_return = (sell_price - entry_price) / entry_price
                    cash = quantity * sell_price
                    quantity = 0.0
                    in_position = False
                    trades.append(trade_return)

        # Final day mark-to-market
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

class SimpleTrendFollowingStrategy(Strategy):
    def __init__(self, data, initial_balance=1000.0, fee=0.001, slippage=0.0005, slow_ma=200, entry_threshold=0.02, atr_period=14, atr_multiplier=2.0):
        super().__init__(data, initial_balance, fee, slippage)
        self.slow_ma = slow_ma
        self.entry_threshold = entry_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def backtest(self):
        data = self.data.copy()
        data['slow_ma'] = data['close'].rolling(window=self.slow_ma).mean()
        
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=self.atr_period).mean()

        cash = self.initial_balance
        quantity = 0.0
        in_position = False
        entry_price = 0.0
        trailing_stop_price = 0.0
        trades = []
        balance_series = []

        for i in range(1, len(data) - 1):
            close_price = data['close'].iloc[i]
            low_price = data['low'].iloc[i]
            atr = data['atr'].iloc[i]

            if in_position:
                balance_series.append(quantity * close_price)
            else:
                balance_series.append(cash)

            # Entry logic
            if not in_position:
                if close_price > data['slow_ma'].iloc[i] * (1 + self.entry_threshold):
                    next_open = data['open'].iloc[i + 1]
                    buy_price = next_open * (1 + self.slippage + self.fee)
                    quantity = cash / buy_price
                    cash = 0.0
                    in_position = True
                    entry_price = buy_price
                    trailing_stop_price = entry_price - self.atr_multiplier * atr
            else:
                # Update trailing stop
                new_stop_price = close_price - self.atr_multiplier * atr
                trailing_stop_price = max(trailing_stop_price, new_stop_price)

                # Exit logic
                if low_price <= trailing_stop_price:
                    next_open = data['open'].iloc[i + 1]
                    sell_price = next_open * (1 - self.slippage - self.fee)
                    trade_return = (sell_price - entry_price) / entry_price
                    cash = quantity * sell_price
                    quantity = 0.0
                    in_position = False
                    trades.append(trade_return)

        # Final day mark-to-market
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

class SimpleTrendFollowingStrategyV2(Strategy):
    def __init__(self, data, initial_balance=1000.0, fee=0.001, slippage=0.0005, slow_ma=200, entry_threshold=0.02, atr_period=14, atr_multiplier=2.0, profit_target_multiplier=3.0, use_kelly_sizing=False, kelly_lookback=50):
        super().__init__(data, initial_balance, fee, slippage)
        self.slow_ma = slow_ma
        self.entry_threshold = entry_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.profit_target_multiplier = profit_target_multiplier
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_lookback = kelly_lookback

    def calculate_kelly_fraction(self, recent_trades):
        """Calculate Kelly criterion fraction based on recent trade performance"""
        if len(recent_trades) < 10:
            return 0.25  # Default conservative position size
        
        wins = [t for t in recent_trades if t > 0]
        losses = [t for t in recent_trades if t <= 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.25
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return 0.25
        
        kelly_fraction = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        return max(0.1, min(0.5, kelly_fraction))  # Cap between 10% and 50%

    def backtest(self):
        data = self.data.copy()
        data['slow_ma'] = data['close'].rolling(window=self.slow_ma).mean()
        
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=self.atr_period).mean()

        cash = self.initial_balance
        quantity = 0.0
        in_position = False
        entry_price = 0.0
        trailing_stop_price = 0.0
        profit_target_price = 0.0
        trades = []
        balance_series = []
        recent_trades = []

        for i in range(1, len(data) - 1):
            close_price = data['close'].iloc[i]
            low_price = data['low'].iloc[i]
            atr = data['atr'].iloc[i]

            if in_position:
                balance_series.append(quantity * close_price)
            else:
                balance_series.append(cash)

            # Entry logic
            if not in_position:
                if close_price > data['slow_ma'].iloc[i] * (1 + self.entry_threshold):
                    next_open = data['open'].iloc[i + 1]
                    buy_price = next_open * (1 + self.slippage + self.fee)
                    
                    if self.use_kelly_sizing:
                        kelly_fraction = self.calculate_kelly_fraction(recent_trades)
                        position_value = cash * kelly_fraction
                        quantity = position_value / buy_price
                        cash -= position_value
                    else:
                        quantity = cash / buy_price
                        cash = 0.0
                    
                    in_position = True
                    entry_price = buy_price
                    trailing_stop_price = entry_price - self.atr_multiplier * atr
                    profit_target_price = entry_price + self.profit_target_multiplier * atr
            else:
                # Update trailing stop
                new_stop_price = close_price - self.atr_multiplier * atr
                trailing_stop_price = max(trailing_stop_price, new_stop_price)

                # Exit logic
                if low_price <= trailing_stop_price or close_price >= profit_target_price:
                    next_open = data['open'].iloc[i + 1]
                    sell_price = next_open * (1 - self.slippage - self.fee)
                    trade_return = (sell_price - entry_price) / entry_price
                    cash += quantity * sell_price
                    quantity = 0.0
                    in_position = False
                    trades.append(trade_return)
                    recent_trades.append(trade_return)
                    if len(recent_trades) > self.kelly_lookback:
                        recent_trades.pop(0)

        # Final day mark-to-market
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

class TrendFilteredMeanReversionMLStrategy(Strategy):
    def __init__(self, data, initial_balance=1000.0, fee=0.001, slippage=0.0005, trend_ma=200, short_ma=20, rsi_period=14, rsi_lower=30, rsi_exit=70, profit_target=0.15, prediction_threshold=0.02):
        super().__init__(data, initial_balance, fee, slippage)
        self.trend_ma = trend_ma
        self.short_ma = short_ma
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_exit = rsi_exit
        self.profit_target = profit_target
        self.prediction_threshold = prediction_threshold

    def backtest(self):
        data = self.data.copy()
        # Calculate indicators
        data['trend_ma'] = data['close'].rolling(window=self.trend_ma).mean()
        data['short_ma'] = data['close'].rolling(window=self.short_ma).mean()
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - 100 / (1 + rs)
        data['std_short'] = data['close'].rolling(window=self.short_ma).std()
        data['bb_lower'] = data['short_ma'] - 2 * data['std_short']
        data['bb_width'] = (data['short_ma'] + 2 * data['std_short']) - data['bb_lower']
        data['macd'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        high_low = data['high'] - data['low']
        high_close_prev = (data['high'] - data['close'].shift()).abs()
        low_close_prev = (data['low'] - data['close'].shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()

        # Create features for ML model
        features = ['rsi', 'short_ma', 'trend_ma', 'bb_lower', 'bb_width', 'macd', 'macd_signal', 'atr']
        data['signal'] = np.where((data['close'] < data['bb_lower']) | (data['rsi'] < self.rsi_lower), 1, 0)
        data['target'] = (data['close'].shift(-5) - data['close']) / data['close']
        data = data.dropna()

        # Split data for training and testing
        train_size = int(len(data) * 0.7)
        train_data = data[:train_size]
        test_data = data[train_size:]

        # Train ML model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_data[features], train_data['target'])

        # Make predictions
        test_data = test_data.copy()
        test_data['prediction'] = model.predict(test_data[features])

        # Backtesting loop
        cash = self.initial_balance
        quantity = 0.0
        in_position = False
        entry_price = 0.0
        trades = []
        balance_series = []

        for i in range(len(test_data) - 1):
            close_price = test_data['close'].iloc[i]

            if in_position:
                balance_series.append(quantity * close_price)
            else:
                balance_series.append(cash)

            # Entry logic
            if not in_position:
                if test_data['signal'].iloc[i] == 1 and test_data['prediction'].iloc[i] > self.prediction_threshold and test_data['close'].iloc[i] > test_data['trend_ma'].iloc[i]:
                    next_open = test_data['open'].iloc[i + 1]
                    buy_price = next_open * (1 + self.slippage + self.fee)
                    quantity = cash / buy_price
                    cash = 0.0
                    in_position = True
                    entry_price = buy_price
            else:
                # Exit logic
                exit_flag = False
                gain = (close_price - entry_price) / entry_price
                if gain >= self.profit_target:
                    exit_flag = True
                if test_data['rsi'].iloc[i] >= self.rsi_exit:
                    exit_flag = True
                if close_price >= test_data['short_ma'].iloc[i]:
                    exit_flag = True
                
                if exit_flag:
                    next_open = test_data['open'].iloc[i + 1]
                    sell_price = next_open * (1 - self.slippage - self.fee)
                    trade_return = (sell_price - entry_price) / entry_price
                    cash = quantity * sell_price
                    quantity = 0.0
                    in_position = False
                    trades.append(trade_return)

        # Final day mark-to-market
        last_close = test_data['close'].iloc[-1]
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
            'equity_curve': pd.Series(balance_series, index=test_data.index[:len(balance_series)]),
            'initial_balance': self.initial_balance,
            'final_balance': cash
        }
