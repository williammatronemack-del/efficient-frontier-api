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
    allow_origins=["*"],  # 👉 restrict to ["https://mackresearch.io"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EODHD_API_KEY = os.getenv("EODHD_API_KEY", "YOUR_API_KEY")

def fetch_eodhd_data(ticker: str, start: str = None):
    url = f"https://eodhd.com/api/eod/{ticker}"
    params = {"api_token": EODHD_API_KEY, "fmt": "json"}
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
    n_points: int = 5000  # simulate 5000 portfolios by default
):
    frames = []
    bad_tickers = []
    for t in tickers:
        try:
            df = fetch_eodhd_data(t, start=start)
            frames.append(df)
        except Exception as e:
            bad_tickers.append(f"{t}: {str(e)}")

    if not frames:
        return {"error": "No data fetched", "skipped": bad_tickers}

    data = pd.concat(frames, axis=1).sort_index()

    # IPO date trimming
    ipo_dates = data.apply(lambda col: col.first_valid_index())
    latest_ipo = ipo_dates.max()
    if pd.isna(latest_ipo):
        return {"error": "No valid data for provided tickers", "skipped": bad_tickers}

    data = data.loc[latest_ipo:].dropna()

    # Daily returns
    daily_returns = data.pct_change().dropna()
    if daily_returns.empty:
        return {"error": "Not enough return data after IPO trimming", "skipped": bad_tickers}

    # Geometric mean returns (annualized)
    log_returns = np.log(1 + daily_returns)
    geo_returns = np.exp(log_returns.mean() * 252) - 1

    # Covariance matrix (annualized)
    cov_matrix = daily_returns.cov() * 252

    # Generate random portfolios
    n_assets = len(tickers)
    portfolios = []
    for _ in range(n_points):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        port_return = np.dot(weights, geo_returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

        portfolios.append({
            "expected_return": float(port_return),
            "variance": float(port_variance),
            "weights": {tickers[i]: float(weights[i]) for i in range(n_assets)}
        })

    # Pareto-efficient filtering
    df = pd.DataFrame(portfolios).sort_values("variance")
    efficient = []
    max_return = -np.inf
    for _, row in df.iterrows():
        if row["expected_return"] > max_return:
            efficient.append(row)
            max_return = row["expected_return"]

    results = [
        {
            "expected_return": round(r["expected_return"], 4),
            "variance": round(r["variance"], 6),
            "weights": {k: round(v, 3) for k, v in r["weights"].items()}
        }
        for r in efficient
    ]

    return {
        "frontier": results,
        "method": "geometric",
        "note": f"All assets limited to IPO of newest asset (since {latest_ipo.date()}).",
        "skipped": bad_tickers
    }

