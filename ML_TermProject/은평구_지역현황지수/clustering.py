import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 불러오기
df = pd.read_csv("은평구_지역현황지수.csv", encoding="cp949")

# 2. 분석할 지표 선택
features = ['비만율', '고혈압신규의료이용률', '당뇨병신규의료이용률']
mean_data = pd.DataFrame()

for feature in features:
    temp = df[df['항목명'] == feature]
    temp = temp.drop(columns=['항목명'])
    
    # ⛏️ melt → 숫자 변환 → NaN 제거
    melted = temp.melt(id_vars='기준일자', var_name='행정동', value_name='값')
    melted['값'] = pd.to_numeric(melted['값'], errors='coerce')
    melted.dropna(subset=['값'], inplace=True)
    
    grouped = melted.groupby('행정동')['값'].mean()
    mean_data[feature] = grouped

# 3. 표준화
scaler = StandardScaler()
scaled = scaler.fit_transform(mean_data)

# 4. KMeans 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled)
mean_data['클러스터'] = clusters

# 5. PCA 시각화
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
mean_data['PCA1'] = pca_result[:, 0]
mean_data['PCA2'] = pca_result[:, 1]

# 6. 시각화
plt.figure(figsize=(10, 7))
sns.scatterplot(data=mean_data, x='PCA1', y='PCA2', hue='클러스터', s=100, palette='Set2')
for i in range(len(mean_data)):
    plt.text(mean_data.iloc[i]['PCA1'], mean_data.iloc[i]['PCA2'], 
             mean_data.index[i], fontsize=9, weight='bold')
plt.title("은평구 행정동 건강지표 클러스터링 (PCA 시각화)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title='클러스터')
plt.tight_layout()
plt.show()

# 7. 결과 저장 (선택)
mean_data.reset_index().to_csv("은평구_건강지표_클러스터링.csv", index=False)

import folium
from folium.plugins import MarkerCluster

# 데이터 불러오기
df = pd.read_csv("은평구_건강지표_클러스터링.csv")

# 좌표 정보 (예시: 주요 행정동만 포함)
coords = {
    "진관동": [37.6344, 126.9184],
    "신사제2동": [37.6026, 126.9129],
    "불광제1동": [37.6101, 126.9313],
    "불광제2동": [37.6098, 126.9272],
    "응암제1동": [37.5999, 126.9187],
    "구산동": [37.6134, 126.9093],
    "녹번동": [37.6005, 126.9356],
    "역촌동": [37.6066, 126.9222],
    "신사제1동": [37.5982, 126.9178]
}

# 클러스터 색상 정의
cluster_colors = {
    0: "green",
    1: "blue",
    2: "red"
}

# 지도 초기화
m = folium.Map(location=[37.615, 126.92], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

# 마커 추가
for _, row in df.iterrows():
    dong = row['행정동']
    if dong in coords:
        lat, lon = coords[dong]
        cluster = int(row['클러스터'])
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(
                f"""
                <div style='font-size:14px; width:280px;'>
                <b style='font-size:16px'>{dong}</b><br>
                클러스터: <b>{cluster}</b><br>
                비만율 평균: {row['비만율']:.1f}%<br>
                고혈압 신규 이용률: {row['고혈압신규의료이용률']:.1f}%<br>
                당뇨병 신규 이용률: {row['당뇨병신규의료이용률']:.1f}%
                </div>
                """,
                max_width=300
            ),
            icon=folium.Icon(color=cluster_colors.get(cluster, 'gray'))
        ).add_to(marker_cluster)

# 지도 저장
m.save("은평구_건강지표_클러스터_지도.html")
