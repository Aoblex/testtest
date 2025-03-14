import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class StockPortfolioEnv(gym.Env):
    metadata = {
        "render_modes": [None],
    }

    def __init__(
        self,
        stock_data: pd.core.frame.DataFrame | str,
        window_size: int,
        initial_money: float,
        transaction_cost_percentage: float,
        render_mode: str | None = None,
    ):
        if isinstance(stock_data, pd.core.frame.DataFrame):
            self.stock_data = stock_data
        elif isinstance(stock_data, str):
            if stock_data.endswith(".pkl"):
                self.stock_data = pd.read_pickle(stock_data)
            elif stock_data.endswith(".csv"):
                self.stock_data = pd.read_csv(stock_data)
            else:
                raise TypeError(f"Unsupported file extension: {stock_data}")
        else:
            raise TypeError(f"Unsupported stock data type: {type(stock_data)}")

        # Check the stock data
        assert isinstance(self.stock_data, pd.core.frame.DataFrame), \
               f"The stock data must be a pandas DataFrame, but got {type(self.stock_data)}"
        
        assert "date" in self.stock_data.columns, \
               "The stock data must contain the 'date' column"

        assert "ticker" in self.stock_data.columns, \
               "The stock data must contain the 'ticker' column"
        
        assert "close" in self.stock_data.columns, \
               "The stock data must contain the 'close' column"

        # WARNING: the date and tickers must be sorted
        self.stock_data.sort_values(by=["date", "ticker"], inplace=True)
        self.stock_data.reset_index(drop=True, inplace=True)
        self.stock_data.index, _ = self.stock_data["date"].factorize()
        self.stock_data.drop(columns=["date"], inplace=True)

        self.signal_feature_names = self.stock_data.columns.to_list()
        self.num_features = len(self.signal_feature_names)
        self.ticker_list = self.stock_data["ticker"].unique().tolist()
        self.num_tickers = len(self.ticker_list)
        self.num_days = len(self.stock_data)
        self.window_size = window_size
        self.initial_money = initial_money
        self.transaction_cost_percentage = transaction_cost_percentage
        self.last_day = len(self.stock_data) - 1


        # Current information
        self._position_status = None
        self._close_prices = None
        self._today = None
        self._total_asset = None
        self.date_buffer = [] # record days
        self.total_asset_buffer = [] # record asset changes

        # The observation space consists of:
        # 1. The signal features of the past `window_size` days
        # 2. The position status of today
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (1 + self.num_features + self.num_tickers, self.num_tickers),
            dtype = np.float32,
        )

        # Notice that we use logits for the action space,
        # the real stock position is the softmax of the action
        self.action_space = spaces.Box(
            low=0,
            high=1.0,
            shape=(self.num_tickers,),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_softmax_action(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        """Convert the action to a softmax action"""
        return np.exp(action) / np.sum(np.exp(action))
    
    def _get_close_prices(
        self,
        day: int,
    ) -> np.ndarray:
        """Get the close prices"""
        return self.stock_data.loc[day, "close"].to_numpy()

    def _get_observation(self):
        """Get the observation"""

        # Fuck the observation....
        observation = np.zeros((1 + self.num_features + self.num_tickers, self.num_tickers))
        data_lookback = self.stock_data.loc[self._today - self.window_size:self._today, :]
        data_lookback.reset_index(drop=False, inplace=True, names=["date"])
        price_lookback = data_lookback.pivot_table(index="date", columns="ticker", values="close")
        return_lookback = price_lookback.pct_change(periods=1).dropna()
        covariance_matrix = return_lookback.cov().to_numpy()

        observation[0, :] = self._position_status
        observation[1:self.num_features + 1, :] = np.concat([
            self.stock_data.loc[self._today][self.stock_data["ticker"] == ticker].to_numpy()
            for ticker in self.ticker_list
        ])
        observation[self.num_features + 1:, :] = covariance_matrix
        return observation
    
    def _get_information(self):
        """Get the information"""
        return {
            "total_asset": self._total_asset,
        }
    
    def _is_terminated(self) -> bool:
        """Check if the episode is terminated"""
        return self._today >= self.last_day

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        """Set the related information"""

        # today is the index of the current day
        self._today = self.window_size

        # the position status always sum up to 1
        self._position_status = np.ones(self.num_tickers) / self.num_tickers

        # close prices is of shape (window_size,)
        self._close_prices = self._get_close_prices(self._today)

        # total asset is the initial money minus the initial transaction cost
        self._total_asset = self.initial_money \
                          - self.transaction_cost_percentage * self.initial_money

        # record trading information
        self.date_buffer.append(self._today)
        self.total_asset_buffer.append(self._total_asset)

        observation = self._get_observation()
        info = self._get_information()

        return observation, info

    def _get_position_status(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        """Get the position status according to the action"""
        return self._get_softmax_action(action)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment

        Args:
            action (np.ndarray): The action to take

        Returns:
            tuple: The observation, reward, terminated, truncated, info
        """

        # get the new position status
        new_position_status = self._get_position_status(action)

        # get the transaction cost according to the new position status
        # the prices are the close prices of today
        position_diff = new_position_status - self._position_status
        total_trade_amount = self._close_prices * position_diff
        transaction_cost = self.transaction_cost_percentage * total_trade_amount

        # get close prices of next day
        new_close_prices = self._get_close_prices(self._today + 1)

        # calculate new total asset, the prices are the close prices of next day
        new_total_asset = new_close_prices * new_position_status - transaction_cost

        # calculate the reward
        reward = new_total_asset - self._total_asset

        # step to next day
        self._today += 1

        # update information
        self._position_status = new_position_status
        self._close_prices = new_close_prices
        self._total_asset = new_total_asset

        observation = self._get_observation()
        info = self._get_information()
        terminated = self._is_terminated()

        # record information
        self.date_buffer.append(self._today)
        self.total_asset_buffer.append(self._total_asset)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        raise NotImplementedError("render is not implemented")