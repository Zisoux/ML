import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

API_KEY = 'e8b0302adda9bfcec623fb1ef6dc870f'
BASE_URL = "http://ws.audioscrobbler.com/2.0/"

st.set_page_config(page_title="Music Trends Forecast", layout="wide")
st.markdown("""
    <style>
    .block-container {
        max-width: 1100px;
        margin: auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Music Listening Trend Forecast (via Last.fm Weekly Charts + LSTM)")

raw_input = st.text_input("🔍 예측할 음악 태그를 입력하세요 (예: kpop):", "kpop")

def get_weekly_chart(tag):
    url = f"{BASE_URL}?method=tag.getweeklyartistchart&tag={tag}&api_key={API_KEY}&format=json"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    try:
        data = res.json()
        artists = data['weeklyartistchart']['artist']
        date = datetime.now()
        listeners = sum(int(artist['playcount']) for artist in artists)
        return listeners
    except:
        return None

if raw_input:
    tag = raw_input.strip()
    st.markdown(f"### 🔄 '{tag}' 데이터 수집 중...")

    # 수집 (최근 20주치)
    trend_data = []
    dates = []

    for week in range(20, 0, -1):
        date = datetime.now() - timedelta(weeks=week)
        listeners = get_weekly_chart(tag)
        if listeners is not None:
            trend_data.append(listeners)
            dates.append(date)
        else:
            trend_data.append(0)
            dates.append(date)

    df = pd.DataFrame({'date': pd.to_datetime(dates), tag: trend_data})
    df.set_index('date', inplace=True)

    st.subheader("📅 수집된 데이터")
    st.dataframe(df)

    # 예측
    st.markdown(f"### 🔮 예측 중...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[tag].values.reshape(-1, 1))

    def create_dataset(data, time_step=4):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 4
    X, y = create_dataset(scaled_data, time_step)

    if len(X) == 0:
        st.warning("⚠ 데이터 부족으로 예측 불가")
    else:
        X = X.reshape(X.shape[0], X.shape[1], 1)
        model = Sequential()
        model.add(LSTM(32, input_shape=(time_step, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=8, verbose=0)

        # 예측 (4주)
        future_input = X[-1]
        future_pred_scaled = []
        for _ in range(4):
            pred = model.predict(future_input.reshape(1, time_step, 1), verbose=0)
            future_pred_scaled.append(pred[0][0])
            future_input = np.append(future_input[1:], pred[0])

        future_pred = scaler.inverse_transform(np.array(future_pred_scaled).reshape(-1, 1)).flatten()
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(4)]

        # 시각화
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df[tag], label="Actual", color="blue")
        ax.plot([last_date] + future_dates, [df[tag].iloc[-1]] + list(future_pred),
                label="Predicted", color="red", linewidth=2)
        ax.axvline(last_date, linestyle='--', color='gray', label="Prediction Start")
        ax.set_title(f"Music Trend Forecast - {tag}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Playcount (aggregated)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
