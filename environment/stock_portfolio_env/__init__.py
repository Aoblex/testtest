from gymnasium.envs.registration import register

register(
    id="stock_portfolio_env/StockPortfolioEnv-v0",
    entry_point="stock_portfolio_env.envs:StockPortfolioEnv",
)
