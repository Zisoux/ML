import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
st.title("은평구 건강지표 클러스터링 분석")

# 데이터 불러오기
cluster_df = pd.read_csv("은평구_건강지표_클러스터링.csv")

# 좌표 정의 (예시)
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

# 클러스터 색상 지정
cluster_colors = {0: 'green', 1: 'blue', 2: 'red'}

# 지도 만들기
m = folium.Map(location=[37.615, 126.92], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

for _, row in cluster_df.iterrows():
    dong = row['행정동']
    if dong in coords:
        folium.Marker(
            location=coords[dong],
            popup=folium.Popup(
                f"""
                <div style='font-size:14px; width:280px;'>
                <b style='font-size:16px'>{dong}</b><br>
                클러스터: <b>{row['클러스터']}</b><br>
                비만율 평균: {row['비만율']:.1f}%<br>
                고혈압 신규 이용률: {row['고혈압신규의료이용률']:.1f}%<br>
                당뇨병 신규 이용률: {row['당뇨병신규의료이용률']:.1f}%
                </div>
                """,
                max_width=300
            ),
            icon=folium.Icon(color=cluster_colors.get(row['클러스터'], 'gray'))
        ).add_to(marker_cluster)

# Streamlit에 지도 렌더링
st.markdown("## 📍 클러스터링 지도 시각화")
st_data = st_folium(m, width=1000, height=600)

# 테이블 출력
st.markdown("## 📊 행정동별 클러스터링 결과")
st.dataframe(cluster_df.style.highlight_max(axis=0))
