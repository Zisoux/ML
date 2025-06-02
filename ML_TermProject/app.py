import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ✅ 페이지 설정
st.set_page_config(page_title="유튜브 채널 추천 시스템", layout="centered")

# ✅ CSS 스타일
st.markdown("""
    <style>
    .main > div {
        display: flex;
        justify-content: center;
    }
    .block-container {
        max-width: 800px;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .bar-container {
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin-top: 4px;
        margin-bottom: 16px;
    }
    .bar-fill {
        height: 100%;
        background-color: #6699ff;
        text-align: right;
        padding-right: 8px;
        color: white;
        line-height: 20px;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ 맨 위 앵커
st.markdown('<a name="top"></a>', unsafe_allow_html=True)

# ✅ 제목 및 설명
st.title("🎥 유튜브 채널 추천기 (하이브리드 모델 기반)")
st.markdown("관심 있는 키워드를 입력하면 머신러닝 + 유사도 기반으로 유튜브 채널을 추천해드립니다!")

# ✅ 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("youtube_channels_500+.csv")
    df = df.dropna(subset=["설명"]).copy()
    df["설명"] = df["설명"].astype(str)
    return df

# ✅ 벡터화 함수
@st.cache_data
def vectorize_descriptions(descriptions):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(descriptions)
    return X, vectorizer

# ✅ 추천 함수
def hybrid_recommend(df, vectorizer, X, user_input, alpha=0.6, top_n=5):
    df = df.copy()
    user_vec = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_vec, X).flatten()

    df["label"] = df["설명"].apply(lambda x: 1 if user_input.lower() in x.lower() else 0)
    if df["label"].sum() > 0:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, df["label"])
        pred_proba = clf.predict_proba(X)[:, 1]
    else:
        pred_proba = [0] * len(df)

    df["유사도"] = cosine_sim
    df["모델점수"] = pred_proba
    df["최종점수"] = alpha * df["모델점수"] + (1 - alpha) * df["유사도"]
    return df.sort_values(by="최종점수", ascending=False).head(top_n)

# ✅ 데이터 준비
df = load_data()
X, vectorizer = vectorize_descriptions(df["설명"])

# ✅ 사용자 입력
user_input = st.text_input("🔍 키워드를 입력하세요 (예: 영화 리뷰, 고양이, 브이로그 등):")
alpha = st.slider("🤖 머신러닝 가중치 (alpha)", 0.0, 1.0, 0.6, step=0.05)
top_n = st.slider("📌 추천 채널 개수", 1, 10, 5)

if user_input:
    recommendations = hybrid_recommend(df, vectorizer, X, user_input, alpha=alpha, top_n=top_n)
    st.subheader("📌 추천 채널 목록")

    for _, row in recommendations.iterrows():
        score_percent = int(row["최종점수"] * 100)

        st.markdown(f"### 🔗 [{row['채널명']}](https://www.youtube.com/channel/{row['채널ID']})")
        st.markdown(f"- **카테고리:** {row['카테고리']}")
        st.markdown(f"- **구독자 수:** {row['구독자 수']}")
        st.markdown(f"- **영상 수:** {row['영상 수']}")
        st.markdown(f"- **총 조회수:** {row['총 조회수']}")
        st.markdown(f"- **설명:** {row['설명']}")
        st.markdown(f"- **✅ 모델 점수:** {row['모델점수']:.3f}")
        st.markdown(f"- **📎 유사도 점수:** {row['유사도']:.3f}")
        st.markdown("**🎯 최종점수:**")
        st.markdown(f"""
        <div class="bar-container">
            <div class="bar-fill" style="width: {score_percent}%;">{score_percent}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    # ✅ 맨 위로 이동 링크
    st.markdown("[🔝 맨 위로 이동](#top)")
