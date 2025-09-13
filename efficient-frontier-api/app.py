from fastapi import FastAPI, Query
import yfinance as yf
import numpy as np
import pandas as pd

app = FastAPI(title="Efficient Frontier Optimizer")

@app.get("/optimize")
def optimize(
    tickers: str = Query(..., description="Comma-separated tickers e.g. AAPL,MSFT,NVDA"),
    start: str = Query("2022-01-01", description="Start date for historical data")
):
    tickers_list = tickers.split(",")
    
    # 1. Fetch data
    data = yf.download(tickers_list, start=start)
    if data.empty or "Adj Close" not in data.columns:
        return {"error": "No price data available for given tickers", "tickers": tickers_list}
    
    prices = data["Adj Close"].dropna()
    if prices.empty:
        return {"error": "No adjusted close data found", "tickers": tickers_list}

    # 2. Compute expected returns & covariance
    returns = prices.pct_change().dropna()
    mu = returns.mean() * 252
    Sigma = returns.cov() * 252

    # 3. Monte Carlo portfolios
    n_portfolios = 3000
    results = []
    for _ in range(n_portfolios):
        weights = np.random.random(len(tickers_list))
        weights /= np.sum(weights)
        port_return = np.dot(weights, mu)
        port_var = np.dot(weights.T, np.dot(Sigma, weights))
        results.append({
            "variance": float(port_var),
            "expected_return": float(port_return),
            "weights": dict(zip(tickers_list, map(float, weights)))
        })
    
    results = sorted(results, key=lambda x: x["variance"])
    return {"frontier": results}

