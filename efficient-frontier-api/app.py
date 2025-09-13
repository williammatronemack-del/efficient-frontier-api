from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
import os
from typing import List

app = FastAPI()

# Enable CORS for WP plugin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👉 in production, restrict to ["https://mackresearch.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EODHD API key (set as env var in Render dashboard)
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "YOUR_API_KEY")  # 🔑 replace if not using env vars

def fetch_eodhd_data(ticker: str, start: str = None):
    url = f"https://eodhd.com/api/eod/{ticker}"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json"
    }
    if start:
        params["from"] = start

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code} for {ticker}")

    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise Exception(f"No data for {ticker}")

    df = pd.DataFrame(data)
    if "adjusted_close" not in df.columns:
        raise Exception(f"No adjusted_close in data for {ticker}")

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df[["adjusted_close"]].rename(columns={"adjusted_close": ticker})

@app.get("/optimize")
def optimize(
    tickers: List[str] = Query(...),
    start: str = None,
    n_points: int = 50
):
    # Fetch data for each asset
    frames = []
    for t in tickers:
        try:
            df = fetch_eodhd_data(t, start=start)
            frames.append(df)
        except Exception as e:
            return {"error": f"Failed to fetch {t}: {str(e)}"}

    if not frames:
        return {"error": "No data fetched"}

    data = pd.concat(frames, axis=1).sort_index()

    # IPO date trimming (newest asset’s start date)
    ipo_dates = data.apply(lambda col: col.first_valid_index())
    latest_ipo = ipo_dates.max()
    if pd.isna(latest_ipo):
        return {"error": "No valid data for provided tickers"}

    data = data.loc[latest_ipo:].dropna()

    # Daily returns
    daily_returns = data.pct_change().dropna()
    if daily_returns.empty:
        return {"error": "Not enough return data after IPO trimming"}

    # Geometric mean returns (annualized, compounded)
    log_returns = np.log(1 + daily_returns)
    geo_returns = np.exp(log_returns.mean() * 252) - 1

    # Covariance matrix (annualized)
    cov_matrix = daily_returns.cov() * 252

    # Generate limited random frontier portfolios
    results = []
    n_assets = len(tickers)
    for _ in range(n_points):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        port_return = np.dot(weights, geo_returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

        results.append({
            "expected_return": round(float(port_return), 4),
            "variance": round(float(port_variance), 6),
            "weights": {tickers[i]: round(float(weights[i]), 3) for i in range(n_assets)}
        })

    # Global Minimum Variance Portfolio (GMVP)
    try:
        ones = np.ones(n_assets)
        inv_cov = np.linalg.inv(cov_matrix.values)
        w_gmvp = inv_cov.dot(ones) / ones.dot(inv_cov).dot(ones)
        gmvp_return = np.dot(w_gmvp, geo_returns)
        gmvp_variance = np.dot(w_gmvp.T, np.dot(cov_matrix, w_gmvp))
        gmvp = {
            "expected_return": round(float(gmvp_return), 4),
            "variance": round(float(gmvp_variance), 6),
            "weights": {tickers[i]: round(float(w_gmvp[i]), 3) for i in range(n_assets)}
        }
    except Exception as e:
        gmvp = {"error": f"Failed to compute GMVP: {str(e)}"}

    return {
        "frontier": results,
        "gmvp": gmvp,
        "method": "geometric",
        "note": f"All assets limited to IPO of newest asset (since {latest_ipo.date()})."
    }
