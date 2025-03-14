from gymnasium.envs.registration import register

register(
    id="stock_trading_env/StockTradingEnv-v0",
    entry_point="stock_trading_env.envs:StockTradingEnv",
)
