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

# 🔎 Debug endpoint (safe)
@app.route("/debug")
def debug_env():
    token = os.environ.get("EODHD_API_TOKEN")
    if token:
        return jsonify({
            "EODHD_API_TOKEN_present": True,
            "EODHD_API_TOKEN_preview": token[:6] + "..."  # only show first 6 chars
        })
    else:
        return jsonify({
            "EODHD_API_TOKEN_present": False
        })

# 📈 Helper: fetch daily closes from EODHD
def fetch_prices(ticker, api_token, period="1y"):
    if "." not in ticker:
        ticker = ticker + ".US"
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
            try:
                prices[t] = fetch_prices(t, api_token)
            except Exception as e:
                print(f"⚠️ Skipping {t}: {e}")

        if not prices:
            return jsonify({"error": "No valid tickers"}), 400

        # Align dates
        df = pd.concat(prices.values(), axis=1, keys=prices.keys()).dropna()
        if df.empty:
            return jsonify({"error": "No overlapping data across tickers"}), 400

        # Daily returns
        returns = df.pct_change().dropna()

        # Mean & covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        n_assets = len(mean_returns)
        n_portfolios = 5000
        results = []
        allocations = {}

        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            results.append({
                "risk": float(port_vol),
                "return": float(port_return),
                "index": i
            })
            allocations[i] = dict(zip(mean_returns.index, map(float, weights)))

        return jsonify({
            "tickers": list(mean_returns.index),
            "portfolios": results,
            "allocations": allocations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



