import numpy as np
from scipy.optimize import minimize

def portfolio_return(weights, returns):
    """
    Calculates portfolio return given a set of weights and asset returns
    """
    return np.dot(weights, returns)

def sharpe_ratio(weights, returns, risk_free_rate):
    """
    Calculates Sharpe ratio given a set of weights, asset returns and risk-free rate
    """
    return (portfolio_return(weights, returns) - risk_free_rate) / np.sqrt(np.dot(weights, weights))

def optimize_portfolio(returns, risk_free_rate):
    """
    Optimizes portfolio weights using Sharpe ratio as the objective function
    """
    num_assets = returns.shape[0]
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    initial_guess = np.array([1/num_assets] * num_assets)
    result = minimize(fun=negative_sharpe_ratio, x0=initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    """
    Returns the negative of the Sharpe ratio
    """
    return -sharpe_ratio(weights, returns, risk_free_rate)

if __name__ == '__main__':
    # Input asset expected returns and risk-free rate
    returns = np.array([float(x) for x in input("Enter expected returns of assets separated by space: ").split()])
    risk_free_rate = float("-2.945")

    # Optimize portfolio
    result = optimize_portfolio(returns, risk_free_rate)
    optimized_weights = result.x

    # Output results
    print("Optimized Weights:", optimized_weights)
    print("Portfolio Return:", portfolio_return(optimized_weights, returns))
    print("Portfolio Sharpe Ratio:", sharpe_ratio(optimized_weights, returns, risk_free_rate))
