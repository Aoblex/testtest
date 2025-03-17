from gymnasium.envs.registration import register

register(
    id="stock_portfolio_env/MultiStockEnv-v0",
    entry_point="stock_portfolio_env.envs:MultiStockEnv",
)