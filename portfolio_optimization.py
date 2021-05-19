from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

from general_utils import _fix_weights


def get_portfolio_weights(df, clean=False, req_return=0.02):
    """
    Returns the portfolio weights, following Markowitz's
    Modern Portfolio Theory to maximize sharpe ratio.
    """
    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe(risk_free_rate=req_return)
    if clean:
        weights = ef.clean_weights()
    # weights = _fix_weights(weights)
    return weights
