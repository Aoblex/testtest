import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pandas as pd
import yfinance as yf
import os
import hashlib

class MultiStockEnvTrain(gym.Env):
    """A multi-stock trading environment for OpenAI gym"""
    metadata = {'render_modes': [None, 'human', 'rgb_array']}

    def __init__(
        self,
        start: str,
        end: str,
        tickers: list[str] | None = None,
        initial_balance: float = 1000000,
        transaction_fee_percent: float = 0.001,
        reward_scaling: float = 1.0,
        max_shares_norm: int = 100,
        cache_dir: str = "ticker_data",
        proxy: str | None = None,
        render_mode: str | None = None,
        window_size: int = 10,  # Default window size of 10 days
    ):
        """
        Initialize the stock trading environment
        
        Parameters:
        - start: start date in format YYYY-MM-DD
        - end: end date in format YYYY-MM-DD
        - tickers: list of ticker symbols
        - initial_balance: starting cash balance
        - transaction_fee_percent: cost of trade as a percentage
        - reward_scaling: scaling factor for rewards
        - max_shares_norm: normalization factor for maximum shares per trade
        - cache_dir: directory to store cached data
        - proxy: proxy server for downloading data (e.g., "http://10.10.1.10:1080")
        - window_size: number of days of historical data to include in each observation
        """
        super(MultiStockEnvTrain, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Default tickers if none provided
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
        
        self.start_date = start
        self.end_date = end
        self.tickers = tickers
        self.stock_dim = len(tickers)
        self.cache_dir = cache_dir
        self.proxy = proxy
        self.window_size = window_size
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Constants
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scaling = reward_scaling
        self.max_shares_norm = max_shares_norm
        
        # Download and process data
        self.data = self._load_or_download_data()
        self.dates = self._get_common_dates()
        
        # We need at least window_size days of data
        if len(self.dates) < self.window_size:
            raise ValueError(f"Not enough trading days. Need at least {self.window_size} days, but only have {len(self.dates)}.")
        
        self.day = self.window_size - 1  # Start after we have enough history for a full window
        
        # Action space: continuous values between -1 and 1 for each stock
        # -1 = sell max, 0 = hold, 1 = buy max
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32
        )
        
        # Observation space includes:
        # - Account balance (1)
        # - Owned shares (stock_dim)
        # - Stock prices for window_size days (stock_dim * window_size)
        # - MACD values for window_size days (stock_dim * window_size)
        # - RSI values for window_size days (stock_dim * window_size)
        # Total dimension = 1 + 3*stock_dim*window_size + stock_dim
        features_per_day = 3  # price, MACD, RSI
        state_dim = 1 + self.stock_dim + (features_per_day * self.stock_dim * self.window_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.terminal = False
        self.portfolio_value = self.initial_balance
        self.balance = self.initial_balance
        self.shares = np.zeros(self.stock_dim)
        self.state = self._get_observation()
        
        # Initialize memory for tracking performance
        self.asset_memory = [self.initial_balance]
        self.rewards_memory = []
        self.actions_memory = []
        self.trades = 0
        self.cost = 0
        
        # Dictionary to store results
        self.results = {}
        
        # Seed random number generator
        self._seed()

    def _get_cache_filename(self, ticker):
        """Generate a unique cache filename based on parameters"""
        # Create a hash of the parameters to ensure uniqueness
        param_str = f"{ticker}_{self.start_date}_{self.end_date}"
        filename = f"{param_str}.csv"
        return os.path.join(self.cache_dir, filename)

    def _load_or_download_data(self):
        """Load data from cache or download if not available"""
        data_dict = {}
        
        for ticker in self.tickers:
            cache_file = self._get_cache_filename(ticker)
            
            # Check if cache file exists
            if os.path.exists(cache_file):
                ticker_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                ticker_data.sort_index(inplace=True)
            else:
                # Download data for this ticker
                ticker_data = self._download_and_process_ticker(ticker)
                # Save to cache
                ticker_data.to_csv(cache_file)
            
            data_dict[ticker] = ticker_data
        
        if not data_dict:
            raise ValueError("Could not load data for any tickers. Please check your date range and ticker symbols.")
        
        return data_dict

    def _download_and_process_ticker(self, ticker):
        """Download and process data for a single ticker"""
        # Download data (always using daily interval "1d")
        # Use yf.Ticker.history() to download data, it's columns have only one level
        yf_ticker = yf.Ticker(ticker, proxy=self.proxy)
        ticker_data = yf_ticker.history(
            start=self.start_date, 
            end=self.end_date, 
            interval="1d",
            proxy=self.proxy,
            auto_adjust=False,
        )
        
        # Check if data was actually downloaded
        if ticker_data.empty:
            raise ValueError(f"No data found for ticker {ticker} in date range {self.start_date} to {self.end_date}")
        
        return self._process_ticker_data(ticker_data)

    def _process_ticker_data(self, ticker_data):
        """Process raw ticker data to include technical indicators"""
        # MACD
        exp1 = ticker_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = ticker_data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        
        # RSI
        delta = ticker_data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        # Handle division by zero
        avg_loss = avg_loss.replace(0, 0.001)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Create a dataframe with the processed data
        processed_df = pd.DataFrame({
            'adjcp': ticker_data['Adj Close'],
            'open': ticker_data['Open'],
            'high': ticker_data['High'],
            'low': ticker_data['Low'],
            'volume': ticker_data['Volume'],
            'macd': macd,
            'rsi': rsi
        })
        
        # Handle NaN values
        processed_df.ffill(inplace=True)
        processed_df.bfill(inplace=True)

        # Check if there are any NaN values
        if processed_df.isna().sum().sum() > 0:
            raise ValueError("NaN values found in processed data")
        
        return processed_df

    def _get_common_dates(self):
        """Get common dates across all ticker data"""
        # Extract all dates from all tickers
        all_dates = set()
        for ticker, df in self.data.items():
            all_dates.update(df.index)
        
        # Get the intersection of dates across all tickers
        common_dates = set.intersection(*[set(df.index) for df in self.data.values()])
        
        # If no common dates, raise an error
        if not common_dates:
            raise ValueError("No common trading dates found across all tickers!")
        
        # Sort the dates
        return sorted(list(common_dates))

    def _get_observation(self):
        """Get current state observation with historical data"""
        # Initialize state with balance
        state = [self.balance]
        
        # Add current owned shares
        state.extend(self.shares)
        
        # Add historical data for each feature over the window
        for day_offset in range(self.window_size):
            # Calculate the index in the dates array
            date_idx = self.day - (self.window_size - 1) + day_offset
            current_date = self.dates[date_idx]
            
            # Add prices for this day
            for ticker in self.tickers:
                if current_date in self.data[ticker].index:
                    state.append(self.data[ticker].loc[current_date, 'adjcp'])
                else:
                    state.append(0)
            
            # Add MACD values for this day
            for ticker in self.tickers:
                if current_date in self.data[ticker].index:
                    state.append(self.data[ticker].loc[current_date, 'macd'])
                else:
                    state.append(0)
            
            # Add RSI values for this day
            for ticker in self.tickers:
                if current_date in self.data[ticker].index:
                    state.append(self.data[ticker].loc[current_date, 'rsi'])
                else:
                    state.append(0)
        
        return np.array(state, dtype=np.float32)

    def _sell_stock(self, index, action):
        """Sell stocks based on the action"""
        ticker = self.tickers[index]
        current_date = self.dates[self.day]
        
        # Get current price
        if current_date in self.data[ticker].index:
            current_price = self.data[ticker].loc[current_date, 'adjcp']
        else:
            # Skip if no data for this date
            return
        
        # Calculate shares to sell (action is between -1 and 0)
        # Negative action means sell, so take absolute value
        shares_to_sell = min(abs(action) * self.max_shares_norm, self.shares[index])
        
        if shares_to_sell > 0:
            # Update balance with sale proceeds minus transaction fee
            self.balance += current_price * shares_to_sell * (1 - self.transaction_fee_percent)
            
            # Update owned shares
            self.shares[index] -= shares_to_sell
            
            # Track costs and trades
            self.cost += current_price * shares_to_sell * self.transaction_fee_percent
            self.trades += 1

    def _buy_stock(self, index, action):
        """Buy stocks based on the action"""
        ticker = self.tickers[index]
        current_date = self.dates[self.day]
        
        # Get current price
        if current_date in self.data[ticker].index:
            current_price = self.data[ticker].loc[current_date, 'adjcp']
        else:
            # Skip if no data for this date
            return
        
        # Calculate maximum shares we can buy with current balance
        max_possible_shares = self.balance // (current_price * (1 + self.transaction_fee_percent))
        
        # Calculate shares to buy (action is between 0 and 1)
        shares_to_buy = min(action * self.max_shares_norm, max_possible_shares)
        
        if shares_to_buy > 0:
            # Update balance
            self.balance -= current_price * shares_to_buy * (1 + self.transaction_fee_percent)
            
            # Update owned shares
            self.shares[index] += shares_to_buy
            
            # Track costs and trades
            self.cost += current_price * shares_to_buy * self.transaction_fee_percent
            self.trades += 1

    def _calculate_portfolio_value(self):
        """Calculate current portfolio value (cash + stock holdings)"""
        current_date = self.dates[self.day]
        portfolio_value = self.balance
        
        for i, ticker in enumerate(self.tickers):
            if current_date in self.data[ticker].index:
                portfolio_value += self.shares[i] * self.data[ticker].loc[current_date, 'adjcp']
        
        return portfolio_value

    def step(self, actions):
        """
        Take an action in the environment
        
        Parameters:
        - actions: numpy array of actions for each stock (values between -1 and 1)
        
        Returns:
        - observation: current state
        - reward: reward for the action
        - terminated: whether the episode is done
        - truncated: whether the episode was truncated
        - info: additional information
        """
        # Check if episode is over
        self.terminal = self.day >= len(self.dates) - 1
        
        if self.terminal:
            # Calculate final portfolio value
            end_portfolio_value = self._calculate_portfolio_value()
            
            # Save results to CSV
            df_total_value = pd.DataFrame(self.asset_memory, columns=['portfolio_value'])
            df_total_value['daily_return'] = df_total_value['portfolio_value'].pct_change(1)
            if df_total_value['daily_return'].std() != 0:
                sharpe = (252**0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            else:
                sharpe = 0
            
            # Store results in dictionary
            self.results = {
                "initial_balance": self.initial_balance,
                "final_portfolio_value": end_portfolio_value,
                "total_return": end_portfolio_value - self.initial_balance,
                "total_transaction_cost": self.cost,
                "total_trades": self.trades,
                "sharpe_ratio": sharpe,
                "portfolio_values": self.asset_memory,
                "rewards": self.rewards_memory,
                "actions": self.actions_memory,
            }

            # Return terminal state
            return self.state, self.reward, self.terminal, False, {}
        
        # Process the actions
        actions = np.clip(actions, -1, 1)  # Ensure actions are in range
        
        # Calculate portfolio value before action
        begin_portfolio_value = self._calculate_portfolio_value()
        
        # Process sell actions first (negative values in actions)
        sell_indices = np.where(actions < 0)[0]
        for index in sell_indices:
            self._sell_stock(index, actions[index])
        
        # Then process buy actions (positive values in actions)
        buy_indices = np.where(actions > 0)[0]
        for index in buy_indices:
            self._buy_stock(index, actions[index])
        
        # Move to the next day
        self.day += 1
        
        # Update state
        self.state = self._get_observation()
        
        # Calculate portfolio value after action
        end_portfolio_value = self._calculate_portfolio_value()
        
        # Calculate reward as change in portfolio value
        self.reward = end_portfolio_value - begin_portfolio_value
        
        # Apply reward scaling
        self.reward = self.reward * self.reward_scaling
        
        # Save portfolio value and reward
        self.asset_memory.append(end_portfolio_value)
        self.rewards_memory.append(self.reward)
        self.actions_memory.append(actions)
        
        return self.state, self.reward, self.terminal, False, {}
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        # Set day counter to have enough history for a full window
        self.day = self.window_size - 1
        
        # Reset balance and shares
        self.balance = self.initial_balance
        self.shares = np.zeros(self.stock_dim)
        
        # Reset tracking variables
        self.asset_memory = [self.initial_balance]
        self.rewards_memory = []
        self.actions_memory = []
        self.trades = 0
        self.cost = 0
        self.terminal = False
        
        # Get initial state
        self.state = self._get_observation()
        
        return self.state, {}
    
    def render(self, mode='human'):
        """Render the environment"""
        return self.state
    
    def _seed(self, seed=None):
        """Seed the random number generator"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]