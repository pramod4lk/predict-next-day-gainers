import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------
# Parameters
# ---------------------------------------------
HIST_DAYS = 60       # lookback window for features
TARGET_HORIZON = 1   # days ahead to predict
THRESHOLD_PCT = 40   # minimum predicted gain %
MAX_RESULTS = 10     # max number of coins to show
PAGE_SIZE = 250      # CoinGecko max per_page

# ---------------------------------------------
# Functions
# ---------------------------------------------
@st.cache_data
def get_all_coin_ids() -> list:
    cg = CoinGeckoAPI()
    cg.session.verify = False
    coin_ids = []
    page = 1
    while True:
        markets = cg.get_coins_markets(vs_currency='usd', per_page=PAGE_SIZE, page=page)
        if not markets:
            break
        coin_ids.extend([c['id'] for c in markets])
        page += 1
    return coin_ids

@st.cache_data
def fetch_market_data(ids: list, days: int = HIST_DAYS) -> pd.DataFrame:
    requests.packages.urllib3.disable_warnings(
        requests.packages.urllib3.exceptions.InsecureRequestWarning
    )
    cg = CoinGeckoAPI()
    cg.session.verify = False

    end = int(datetime.utcnow().timestamp())
    start = end - days * 24 * 3600
    records = []
    for coin in ids:
        try:
            data = cg.get_coin_market_chart_range_by_id(
                coin, vs_currency='usd', from_timestamp=start, to_timestamp=end
            )
            df = pd.DataFrame(data['prices'], columns=['time', 'price'])
            df['volume'] = [v[1] for v in data['total_volumes']]
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            df['coin'] = coin
            records.append(df)
        except Exception:
            continue
    return pd.concat(records)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for coin, grp in df.groupby('coin'):
        g = grp.resample('24H').agg({'price':'last','volume':'sum'})
        g['ret_1d'] = g['price'].pct_change(TARGET_HORIZON) * 100
        g['vol_7d'] = g['volume'].rolling(7).mean()
        g['volatility_7d'] = g['ret_1d'].rolling(7).std()
        g['momentum_14d'] = g['price'].pct_change(14)
        g['target'] = g['ret_1d'].shift(-TARGET_HORIZON)
        g['coin'] = coin
        out.append(g.dropna())
    return pd.concat(out)


def train_and_predict(features: pd.DataFrame) -> pd.DataFrame:
    data = features[['ret_1d','vol_7d','volatility_7d','momentum_14d','coin','target']].dropna()
    X = data[['ret_1d','vol_7d','volatility_7d','momentum_14d']]
    y = data['target']

    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    results = X_test.copy()
    results['coin'] = data['coin'].iloc[split_idx:].values
    results['predicted_pct'] = preds
    results['actual_pct'] = y.iloc[split_idx:].values
    results.index = data.index[split_idx:]
    return results

# ---------------------------------------------
# UI
# ---------------------------------------------
def main():
    st.set_page_config(page_title="Crypto Gainer Predictor", layout="wide")
    st.title("🔮 Predict Next-Day >40% Gainers")
    st.markdown(f"Forecast coins predicted to gain >{THRESHOLD_PCT}% in the next 24h.")

    if st.button("Run Prediction"):
        with st.spinner("Fetching coin list..."):
            all_ids = get_all_coin_ids()
        with st.spinner(f"Fetching market data for {len(all_ids)} coins..."):
            raw = fetch_market_data(all_ids)
        with st.spinner("Engineering features..."):
            feats = engineer_features(raw)
        with st.spinner("Training model & computing predictions..."):
            res = train_and_predict(feats)

        latest_date = res.index.max().normalize()
        today_preds = res[res.index.normalize() == latest_date]
        # Filter by threshold and limit results
        filtered = today_preds[today_preds['predicted_pct'] > THRESHOLD_PCT]
        top_filtered = filtered.nlargest(MAX_RESULTS, 'predicted_pct')

        tomorrow = (latest_date + timedelta(days=1)).date()
        if top_filtered.empty:
            st.info(f"No coins predicted to exceed {THRESHOLD_PCT}% gain on {tomorrow}.")
        else:
            st.subheader(f"Coins Predicted >{THRESHOLD_PCT}% for {tomorrow}")
            st.table(
                top_filtered[['coin','predicted_pct','actual_pct']]
                .rename(columns={
                    'coin':'Coin',
                    'predicted_pct':'Predicted % Gain',
                    'actual_pct':'Actual % Gain'
                })
            )

if __name__ == "__main__":
    main()
