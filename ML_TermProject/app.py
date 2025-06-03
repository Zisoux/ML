import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import datetime
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from imblearn.over_sampling import SMOTE  # SMOTE를 사용하여 불균형 처리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ 페이지 설정
st.set_page_config(page_title="유튜브 채널 추천 및 관리", layout="wide")

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

# ✅ 제목 및 설명
st.title("🎥 유튜브 채널 추천기 & 관리 서비스")
st.markdown("유튜브 채널 추천과 관리 기능을 제공합니다. 관심 있는 키워드를 입력하고 구독한 채널을 관리해보세요!")

# OAuth 2.0 설정
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']

# ✅ 인증 함수
def authenticate():
    """OAuth 2.0 인증을 통한 사용자 인증"""
    creds = None
    # 토큰 파일이 존재하면 자격 증명을 로드
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # 자격 증명이 없거나 만료된 경우 새로 인증
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'client_secrets.json', SCOPES)
                creds = flow.run_local_server(port=8080)  # 포트 변경
                print("Authentication successful!")
            except Exception as e:
                print("Error during authentication:", e)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds

# 인증 후 YouTube API 클라이언트 생성
creds = authenticate()
youtube = build('youtube', 'v3', credentials=creds)

# ✅ 데이터 로드 (유튜브 채널 추천용)
@st.cache_data
def load_data():
    df = pd.read_csv("youtube_channels_500+.csv")
    df = df.dropna(subset=["설명"]).copy()
    df["설명"] = df["설명"].astype(str)
    return df

# ✅ 벡터화 함수 (유튜브 채널 추천용)
@st.cache_data
def vectorize_descriptions(descriptions):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(descriptions)
    return X, vectorizer

# ✅ 추천 함수 (유튜브 채널 추천용)
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

# ✅ 사용자가 구독한 채널 및 시청 여부 확인 함수
def get_subscribed_channels():
    channels = []
    request = youtube.subscriptions().list(
        part="snippet",
        mine=True,
        maxResults=50
    )
    response = request.execute()
    for item in response["items"]:
        channel = item["snippet"]["title"]
        channel_id = item["snippet"]["resourceId"]["channelId"]
        channels.append((channel, channel_id))
    return channels

# ✅ 사용자가 시청한 영상 목록을 기반으로 보지 않은 채널을 분석하는 함수
def get_unwatched_channels(days):
    unwatched_channels = set()  # 집합으로 변경하여 중복을 방지
    last_viewed_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # 구독한 채널 목록을 불러와 보지 않은 채널을 추려냅니다.
    channels = get_subscribed_channels()
    for channel, channel_id in channels:
        # 해당 채널의 영상을 가져오고, 사용자가 마지막으로 본 영상을 확인합니다.
        request = youtube.activities().list(
            part="snippet",
            channelId=channel_id,
            maxResults=5
        )
        response = request.execute()
        
        for activity in response["items"]:
            video_date = activity["snippet"]["publishedAt"]
            # fromisoformat을 사용하여 타임존을 포함한 날짜를 처리합니다.
            video_date = datetime.datetime.fromisoformat(video_date)
            
            # 타임존 정보 제거 (naive datetime으로 변환)
            video_date = video_date.replace(tzinfo=None)
            
            if video_date < last_viewed_date:
                unwatched_channels.add(channel)  # 집합에 추가하여 중복 방지
    
    return list(unwatched_channels)  # 리스트로 변환하여 반환


# ✅ 모델 학습 및 하이퍼파라미터 튜닝
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # 모델들 정의
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1, solver='liblinear'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10),
        'XGBoost': xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=6)
    }

    # 하이퍼파라미터 그리드 설정 (랜덤 포레스트, XGBoost 예시)
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9]
        }
    }

    best_models = {}
    best_scores = {}

    # StratifiedKFold 사용하여 데이터가 불균형할 때도 비율을 유지하도록 설정
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)  # cv=2로 변경

    # 성능을 기록하기 위해 출력된 모델 성능을 저장
    st.subheader("모델 성능 평가")
    
    for model_name, model in models.items():
        st.write(f"Training {model_name}...")
        
        # 하이퍼파라미터 튜닝
        grid_search = GridSearchCV(model, param_grids[model_name], cv=cv, scoring='accuracy')  # StratifiedKFold로 교차 검증
        grid_search.fit(X_train, y_train)
        
        # 최적 모델
        best_models[model_name] = grid_search.best_estimator_
        
        # 예측 및 평가
        y_pred = best_models[model_name].predict(X_test)
        
        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        best_scores[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        st.write(f"**{model_name} 성능**:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")
        st.markdown("---")
        
    return best_models, best_scores


# ✅ 사용자 인터페이스 구성
col1, col2 = st.columns(2)

# 왼쪽: 유튜브 채널 추천 서비스
with col1:
    st.subheader("📌 유튜브 채널 추천")
    user_input = st.text_input("🔍 키워드를 입력하세요 (예: 영화 리뷰, 고양이, 브이로그 등):")
    alpha = st.slider("🤖 머신러닝 가중치 (alpha)", 0.0, 1.0, 0.6, step=0.05)
    top_n = st.slider("📌 추천 채널 개수", 1, 10, 5)
    
    if user_input:
        df = load_data()
        X, vectorizer = vectorize_descriptions(df["설명"])
        recommendations = hybrid_recommend(df, vectorizer, X, user_input, alpha=alpha, top_n=top_n)
        for _, row in recommendations.iterrows():
            score_percent = int(row["최종점수"] * 100)
            st.markdown(f"### 🔗 [{row['채널명']}](https://www.youtube.com/channel/{row['채널ID']})")
            st.markdown(f"- **카테고리:** {row['카테고리']}")
            st.markdown(f"- **구독자 수:** {row['구독자 수']}")
            st.markdown(f"- **영상 수:** {row['영상 수']}")

# 오른쪽: 잘 보지 않는 채널 정리 서비스
with col2:
    st.subheader("🗑️ 잘 보지 않는 채널 정리")
    days = st.slider("잘 보지 않은 채널을 판단할 기준 기간을 선택하세요 (최소 3일, 최대 30일):", 3, 30, 7)
    unwatched_channels = get_unwatched_channels(days)  # 선택한 날짜를 기준으로 잘 보지 않은 채널 추출
    if unwatched_channels:
        st.markdown("**구독 취소를 추천하는 채널들**:") 
        for channel in unwatched_channels:
            st.markdown(f"- {channel}")
    else:
        st.markdown("**모든 채널을 잘 보고 있습니다!**")

# 데이터 준비
df_channels = pd.DataFrame({
    '구독자 수': [1000, 1500, 2000, 800],
    '영상 수': [100, 50, 200, 10],
    '라벨': [1, 0, 1, 0]  # 예시 라벨: 1=잘 보고 있는 채널, 0=잘 보지 않은 채널
})

X = df_channels[['구독자 수', '영상 수']]  # 피처
y = df_channels['라벨']  # 라벨

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습 및 평가
best_models, best_scores = train_and_evaluate_model(X_train, X_test, y_train, y_test)

# 최적 모델 선택하여 출력
best_model_name = max(best_scores, key=lambda x: best_scores[x]['Accuracy'])  # 정확도가 가장 높은 모델 선택
best_model = best_models[best_model_name]

st.write(f"**선택된 모델**: {best_model_name} - 정확도: {best_scores[best_model_name]['Accuracy']:.2f}")
