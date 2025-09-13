from fastapi import FastAPI, Query
import numpy as np
import pandas as pd
import requests
import os

app = FastAPI(title="Efficient Frontier Optimizer")

# Load your EODHD API key from Render environment variable
EODHD_API_KEY = os.getenv("EODHD_API_KEY")

@app.get("/optimize")
def optimize(
    tickers: str = Query(..., description="Comma-separated tickers e.g. AAPL.US,MSFT.US,NVDA.US"),
    start: str = Query("2022-01-01", description="Start date for historical data"),
    end: str = Query(None, description="End date for historical data")
):
    tickers_list = tickers.split(",")
    all_data = pd.DataFrame()

    for ticker in tickers_list:
        url = f"https://eodhd.com/api/eod/{ticker}"
        params = {
            "from": start,
            "to": end,
            "api_token": EODHD_API_KEY,
            "period": "d",
            "fmt": "json"
        }
        r = requests.get(url, params=params)
        if r.status_code != 200:
            return {"error": f"Failed to fetch {ticker}", "details": r.text}
        
        data = r.json()
        if not data:
            return {"error": f"No data for {ticker}"}
        
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        all_data[ticker] = df["adjusted_close"]

    if all_data.empty:
        return {"error": "No usable price data from EODHD"}

    # Compute returns, mean returns, covariance
    returns = all_data.pct_change().dropna()
    mu = returns.mean() * 252
    Sigma = returns.cov() * 252

    # Monte Carlo portfolios
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


