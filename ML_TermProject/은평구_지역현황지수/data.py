import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import folium
from folium.plugins import MarkerCluster
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 불러오기
df = pd.read_csv("은평구_지역현황지수.csv", encoding="cp949")

# 2. 분석 대상 항목
features = ['비만율', '고혈압신규의료이용률', '당뇨병신규의료이용률']
all_outliers = {}

for feature in features:
    target_df = df[df['항목명'] == feature].copy()
    dong_columns = target_df.columns[3:]
    melted = target_df.melt(id_vars=['기준일자', '항목명'], value_vars=dong_columns,
                            var_name='행정동', value_name='값')
    model = IsolationForest(contamination=0.05, random_state=42)
    melted['이상치'] = model.fit_predict(melted[['값']])
    melted['이상치'] = melted['이상치'].map({1: '정상', -1: '이상치'})
    all_outliers[feature] = melted

# 3. 이상치 시각화
fig, axs = plt.subplots(nrows=len(features), figsize=(12, 5 * len(features)))
for i, feature in enumerate(features):
    sns.countplot(data=all_outliers[feature], x='기준일자', hue='이상치', ax=axs[i])
    axs[i].set_title(f"{feature} - 연도별 이상치 탐지 결과")
    axs[i].set_xlabel("연도")
    axs[i].set_ylabel("건수")
    axs[i].legend(title='상태')
plt.tight_layout()
plt.show()

# 4. Geo 위치 기반 시각화 - 예시용 좌표
coords = {
    "진관동": [37.6344, 126.9184], "신사제2동": [37.6026, 126.9129],
    "불광제1동": [37.6101, 126.9313], "신사제1동": [37.5982, 126.9178],
    "구산동": [37.6134, 126.9093]
}

# 5. 비만율 증가율 계산 (최근 3년 vs 과거)
obesity_df = all_outliers['비만율']
obesity_df['기준일자'] = obesity_df['기준일자'].astype(int)
recent_years = sorted(obesity_df['기준일자'].unique())[-3:]
recent_avg = obesity_df[obesity_df['기준일자'].isin(recent_years)].groupby('행정동')['값'].mean()
past_avg = obesity_df[~obesity_df['기준일자'].isin(recent_years)].groupby('행정동')['값'].mean()
increase_rate = ((recent_avg - past_avg) / past_avg * 100).dropna()

# 6. 지도 시각화 (팝업 폰트 및 스타일 커스터마이징)
m = folium.Map(location=[37.615, 126.92], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

for dong, rate in increase_rate.items():
    if dong in coords:
        popup_html = f"""
        <div style="width: 280px; font-size: 15px;">
            <strong style="font-size: 16px;">{dong}</strong><br>
            비만 증가율: <b>{rate:.1f}%</b>
        </div>
        """
        folium.Marker(
            location=coords[dong],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="red" if rate > 5 else "green")
        ).add_to(marker_cluster)

m.save("은평구_비만율_지도.html")  # HTML 저장

# 7. 시계열 예측 (LSTM) – 진관동 비만율 예측
def predict_lstm_timeseries(data_series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data_series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled) - 3):
        X.append(scaled[i:i+3])
        y.append(scaled[i+3])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(3, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    pred_scaled = model.predict(scaled[-3:].reshape((1, 3, 1)))
    return scaler.inverse_transform(pred_scaled)[0][0]

jin_gwan_series = obesity_df[obesity_df['행정동'] == '진관동'].sort_values('기준일자')['값']
predicted_value = predict_lstm_timeseries(jin_gwan_series)

# 8. 자연어 요약 설명
explanations = []
for dong, rate in increase_rate.sort_values(ascending=False).head(5).items():
    explanations.append(f"{dong}은(는) 최근 3년간 비만율이 과거 대비 {rate:.1f}% 증가했습니다.")
explanations.append(f"진관동의 다음 해 비만율은 약 {predicted_value:.2f}%로 예측됩니다.")

# 출력
for sentence in explanations:
    print(sentence)

# 9. 결과 CSV 저장
df.to_csv("은평구_이상치분석결과.csv", index=False)
