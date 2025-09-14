# app.py
import os
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# ✅ Enable CORS for your WP site
CORS(app, origins=["https://mackresearch.io"])

# 🔎 Simple test endpoint
@app.route("/ping")
def ping():
    return jsonify({"ok": True, "message": "CORS is working 🚀"})

# 📈 Helper: fetch daily closes from EODHD
def fetch_prices(ticker, api_token, period="1y"):
    url = f"https://eodhd.com/api/eod/{ticker}?api_token={api_token}&fmt=json&period=d"
    res = requests.get(url)
    if res.status_code != 200:
        raise ValueError(f"Failed to fetch {ticker}: {res.text}")
    data = res.json()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df["close"]

# 🧮 Optimize endpoint
@app.route("/optimize")
def optimize():
    api_token = os.environ.get("EODHD_API_TOKEN")
    if not api_token:
        return jsonify({"error": "EODHD_API_TOKEN not set"}), 500

    tickers = request.args.getlist("tickers")
    if len(tickers) < 2:
        return jsonify({"error": "Please provide at least 2 tickers"}), 400

    try:
        # Fetch historical prices for all tickers
        prices = {}
        for t in tickers:
            prices[t] = fetch_prices(t, api_token)

        # Align dates
        df = pd.concat(prices.values(), axis=1, keys=prices.keys()).dropna()

        # Daily returns
        returns = df.pct_change().dropna()

        # Mean & covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        n_assets = len(tickers)
        n_portfolios = 5000
        results = []

        allocations = {}

        for i in range(n_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            # Portfolio return & volatility
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            results.append({"risk": float(port_vol), "return": float(port_return), "index": i})
            allocations[i] = dict(zip(tickers, map(float, weights)))

        return jsonify({
            "tickers": tickers,
            "portfolios": results,
            "allocations": allocations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


