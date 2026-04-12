import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timedelta
import sys
import logging
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('trading.log'),
                        logging.StreamHandler()
                    ])

if not os.getenv("ALPACA_PAPER_API_KEY_ID"):
    load_dotenv()

API_KEY = os.getenv("ALPACA_PAPER_API_KEY_ID")
API_SECRET = os.getenv("ALPACA_PAPER_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    logging.error("Alpaca API keys or base URL not found in .env file.")
    sys.exit(1)

try:
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
except Exception as e:
    logging.error(f"Error initializing Alpaca API: {e}")
    sys.exit(1)

class AlpacaTradingEnvironment(gym.Env):
    """
    A trading environment for SPY using Alpaca API, with dynamic data updates and rate limiting.
    Supports live and historical trading modes.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, symbol='SPY', window_size=50, initial_balance=10000, commission=0.002, interval='1Min', live_mode=False):
        super().__init__()
        self.symbol = symbol
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.current_step = window_size
        self.live_mode = live_mode
        self.action_space = spaces.Discrete(3)  # 0: Buy, 1: Hold, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size + 5,),
                                           dtype=np.float32)  # Price window + balance + shares + RSI + MACD + VXX
        self.interval = interval
        self.commission = commission
        self.slippage = 0.001
        self.portfolio_history = [(0, initial_balance)]
        self.entry_price = 0.0
        self.last_data_timestamp = None
        self.update_counter = 0
        # Price caching
        self.last_vxx_price = 0.0
        self.last_vxx_time = 0
        self.last_spy_price = 1.0
        self.last_spy_time = 0
        self.cache_duration = 5  # Cache for 5 seconds
        if not self.live_mode:
            try:
                self.historical_data = self._get_historical_data(symbol, 100000)
                self.max_steps = len(self.historical_data) - 1
                logging.info(f"Initialized historical environment with {len(self.historical_data)} data points")
            except Exception as e:
                logging.error(f"Failed to initialize historical data: {e}")
                sys.exit(1)
        else:
            self.live_data = np.array([self.get_latest_price() for _ in range(window_size)])  # Initialize with latest price
            self.max_steps = None  # No limit in live mode
            logging.info("Initialized live environment")

    def _get_historical_data(self, symbol, period, start_date=None):
        now = datetime.now()
        end_date = now.strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (now - timedelta(days=365)).strftime('%Y-%m-%d')
        logging.info(
            f"Fetching historical data for {symbol}, period: {period}, interval: {self.interval}, "
            f"start: {start_date}, end: {end_date}, feed: iex"
        )
        try:
            barset = api.get_bars(symbol, self.interval, start=start_date, end=end_date, limit=period, feed='iex').df
            logging.info(f"Received {len(barset)} data points")
            if len(barset) < self.window_size:
                raise ValueError(f"Not enough historical data for {symbol}. Got {len(barset)}, need at least {self.window_size}")
            prices = barset['close'].values.flatten()
            if np.any(np.isnan(prices)) or np.any(prices <= 0):
                raise ValueError("Historical data contains invalid values (NaN or non-positive)")
            self.last_data_timestamp = barset.index[-1] if not barset.empty else now
            return prices
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            raise RuntimeError(f"Failed to get historical data: {e}")

    def _update_historical_data(self):
        try:
            start_date = self.last_data_timestamp.strftime('%Y-%m-%d')
            new_prices = self._get_historical_data(self.symbol, 1000, start_date)
            if len(new_prices) > 0:
                self.historical_data = np.concatenate([self.historical_data, new_prices])
                self.max_steps = len(self.historical_data) - 1
                logging.info(f"Updated historical data: {len(self.historical_data)} points, new prices: {len(new_prices)}")
        except Exception as e:
            logging.error(f"Error updating historical data: {e}")

    def get_latest_price(self):
        if time.time() - self.last_spy_time < self.cache_duration:
            return self.last_spy_price
        for attempt in range(3):
            try:
                quote = api.get_latest_trade(self.symbol)
                self.last_spy_price = quote.price
                self.last_spy_time = time.time()
                return quote.price
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/3 failed to fetch price for {self.symbol}: {e}")
                if attempt < 2:
                    time.sleep(3)
                else:
                    logging.error(f"Error fetching latest price: {e}")
                    return self.last_spy_price if self.last_spy_time > 0 else self.historical_data[-1] if not self.live_mode else 1.0

    def get_vxx_price(self):
        if time.time() - self.last_vxx_time < self.cache_duration:
            return self.last_vxx_price
        for attempt in range(3):
            try:
                quote = api.get_latest_trade('VXX')
                self.last_vxx_price = quote.price
                self.last_vxx_time = time.time()
                return quote.price
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/3 failed to fetch price for VXX: {e}")
                if attempt < 2:
                    time.sleep(3)
                else:
                    logging.error(f"Error fetching VXX price: {e}")
                    return self.last_vxx_price if self.last_vxx_time > 0 else 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        self.portfolio_history = [(0, self.initial_balance)]
        self.entry_price = 0.0
        self.update_counter = 0
        self.historical_data = None  # Clear historical data
        self.last_data_timestamp = None  # Reset timestamp
        if not self.live_mode:
            try:
                self.historical_data = self._get_historical_data(self.symbol, 100000)
                self.max_steps = len(self.historical_data) - 1
                logging.info(f"Reset historical environment with {len(self.historical_data)} data points")
            except Exception as e:
                logging.error(f"Failed to reset historical data: {e}")
                raise RuntimeError(f"Failed to reset historical data: {e}")
        else:
            self.live_data = np.array([self.get_latest_price() for _ in range(self.window_size)])
        observation = self._get_observation()
        current_price = self.live_data[-1] if self.live_mode else self.historical_data[self.current_step]
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': self.balance,
            'current_price': current_price,
            'shares_bought': 0,
            'shares_sold': 0,
            'data_start_time': start_date,
            'data_end_time': end_date
        }
        logging.info(f"Environment reset: {info}")
        return observation, info

    def _calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = deltas.clip(min=0)
        losses = -deltas.clip(max=0)
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 1e-10
        rs = avg_gain / avg_loss if avg_loss != 0 else 1e10
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        if len(prices) < slow:
            return 0.0
        exp_fast = np.convolve(prices, np.ones(fast)/fast, mode='valid')[-1] if len(prices) >= fast else prices[-1]
        exp_slow = np.convolve(prices, np.ones(slow)/slow, mode='valid')[-1] if len(prices) >= slow else prices[-1]
        macd_line = exp_fast - exp_slow
        return macd_line

    def _get_observation(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        if self.live_mode:
            price_window = self.live_data[-self.window_size:] if len(self.live_data) >= self.window_size else \
                           np.pad(self.live_data, (self.window_size - len(self.live_data), 0), 'edge')
        else:
            price_window = self.historical_data[start:end] if self.current_step < len(self.historical_data) else \
                           np.concatenate([self.historical_data[-self.window_size+1:], [self.get_latest_price()]])
        rsi = self._calculate_rsi(price_window)
        macd = self._calculate_macd(price_window)
        scaled_balance = self.balance / self.initial_balance if self.initial_balance > 0 else 0
        current_price = self.live_data[-1] if self.live_mode else \
                        self.historical_data[self.current_step - 1] if self.current_step < len(self.historical_data) else self.get_latest_price()
        scaled_shares = self.shares_held / (self.initial_balance / current_price) if current_price > 0 else 0
        vxx_price = self.get_vxx_price()
        observation = np.concatenate([price_window, [scaled_balance, scaled_shares, rsi/100, macd, vxx_price/100]]).astype(np.float32)
        return observation

    def step(self, action):
        if self.live_mode:
            current_price = self.get_latest_price()
            self.live_data = np.append(self.live_data, current_price)
            if len(self.live_data) > self.window_size:
                self.live_data = self.live_data[-self.window_size:]
        else:
            if self.current_step >= len(self.historical_data):
                current_price = self.get_latest_price()
            else:
                current_price = self.historical_data[self.current_step]

        if current_price <= 0:
            logging.error(f"Invalid current price at step {self.current_step}: {current_price}")
            current_price = self.live_data[-2] if self.live_mode and len(self.live_data) > 1 else \
                            self.historical_data[self.current_step - 1] if self.current_step > 0 else 1

        # Remove dynamic data updates in historical mode
        self.update_counter += 1

        if self.shares_held > 0 and current_price < self.entry_price * 0.95:
            logging.info(f"Stop-loss triggered: Price {current_price:.2f} < {self.entry_price * 0.95:.2f}")
            action = 2

        prev_portfolio_value = self.balance + self.shares_held * (
            self.live_data[-2] if self.live_mode and len(self.live_data) > 1 else
            self.historical_data[self.current_step - 1] if self.current_step > 0 and not self.live_mode else current_price
        )
        shares_bought = 0
        shares_sold = 0

        logging.info(
            f"Step: {self.current_step}, Action: {action}, Price: {current_price:.2f}, "
            f"Balance: {self.balance:.2f}, Shares: {self.shares_held}, "
            f"Prev Portfolio: {prev_portfolio_value:.2f}"
        )

        if action == 0:  # Buy
            if current_price > 0 and self.balance >= current_price:
                buy_price = current_price * (1 + self.slippage)
                shares_to_buy = int(self.balance / buy_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * buy_price * (1 + self.commission)
                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    shares_bought = shares_to_buy
                    self.entry_price = current_price
                    logging.info(
                        f"Simulated Buy: {shares_to_buy} shares of {self.symbol} at {buy_price:.2f}, "
                        f"Cost: {cost:.2f}, New Balance: {self.balance:.2f}, Shares: {self.shares_held}"
                    )
                else:
                    logging.info("Buy action skipped: Cannot buy zero shares")
            else:
                logging.info("Buy action skipped: Insufficient balance or invalid price")
        elif action == 2:  # Sell
            if self.shares_held > 0 and current_price > 0:
                sell_price = current_price * (1 - self.slippage)
                shares_to_sell = self.shares_held
                gain = shares_to_sell * sell_price * (1 - self.commission)
                self.balance += gain
                self.shares_held = 0
                shares_sold = shares_to_sell
                logging.info(
                    f"Simulated Sell: {shares_to_sell} shares of {self.symbol} at {sell_price:.2f}, "
                    f"Gain: {gain:.2f}, New Balance: {self.balance:.2f}, Shares: {self.shares_held}"
                )
            else:
                logging.info("Sell action skipped: No shares to sell or invalid price")
        else:
            logging.info("Action: Hold")

        self.current_step += 1
        done = self.current_step >= self.max_steps if not self.live_mode else False

        current_portfolio_value = self.balance + self.shares_held * current_price
        raw_reward = current_portfolio_value - prev_portfolio_value
        portfolio_returns = [val[1] / self.initial_balance - 1 for val in self.portfolio_history[-100:]]
        if len(portfolio_returns) > 1:
            std_returns = np.std(portfolio_returns) if portfolio_returns else 1e-6
            sharpe_ratio = (raw_reward / self.initial_balance) / (std_returns + 1e-6)
            reward = sharpe_ratio
        else:
            reward = raw_reward / self.initial_balance
        
        if current_portfolio_value < prev_portfolio_value * 0.95:  # 5% loss
            reward -= 0.1

        self.portfolio_history.append((self.current_step, current_portfolio_value))

        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': current_portfolio_value,
            'current_price': current_price,
            'shares_bought': shares_bought,
            'shares_sold': shares_sold
        }

        logging.info(
            f"Step Complete: Portfolio Value: {current_portfolio_value:.2f}, "
            f"Reward: {reward:.2f}, Done: {done}, Shares Bought: {shares_bought}, Sold: {shares_sold}"
        )

        return observation, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass