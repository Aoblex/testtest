from gymnasium.envs.registration import register

register(
    id="stock_portfolio_env/MultiStockEnvTrade-v0",
    entry_point="stock_portfolio_env.envs:MultiStockEnvTrade",
)

register(
    id="stock_portfolio_env/MultiStockEnvTrain-v0",
    entry_point="stock_portfolio_env.envs:MultiStockEnvTrain",
)