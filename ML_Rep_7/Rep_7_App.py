import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 페이지 설정
st.set_page_config(page_title="Sensor Activity Classifier", layout="wide")
st.title("📊 Sensor Activity Classification App")

st.markdown("""
이 앱은 가속도 센서 데이터를 기반으로 고소작업자 상태 (걷기, 뛰기, 정지)를 예측하는 머신러닝 모델을 비교합니다.
""")

# 데이터 업로드
uploaded_walk = st.file_uploader("걷기 데이터 업로드", type="csv")
uploaded_run = st.file_uploader("뛰기 데이터 업로드", type="csv")
uploaded_stop = st.file_uploader("정지 데이터 업로드", type="csv")

if uploaded_walk and uploaded_run and uploaded_stop:
    # CSV 읽기
    walk_df = pd.read_csv(uploaded_walk)
    walk_df['label'] = 'walk'
    run_df = pd.read_csv(uploaded_run)
    run_df['label'] = 'run'
    stop_df = pd.read_csv(uploaded_stop)
    stop_df['label'] = 'stop'

    # 병합
    df = pd.concat([walk_df, run_df, stop_df], ignore_index=True)

    # 전처리
    feature_cols = [
        'Linear Acceleration x (m/s^2)',
        'Linear Acceleration y (m/s^2)',
        'Linear Acceleration z (m/s^2)'
    ]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=feature_cols)

    # 이상치 제거
    from scipy.stats import zscore
    z_scores = np.abs(zscore(df[feature_cols]))
    df = df[(z_scores < 3).all(axis=1)]

    # 시각화
    st.subheader("📈 Feature Pairplot")
    fig = sns.pairplot(df[feature_cols + ['label']], hue='label', diag_kind="hist")
    st.pyplot(fig)

    # 학습 준비
    X = df[feature_cols]
    y = df['label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델
    models = {
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    results = []
    st.subheader("📊 Model Performance")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })

        st.markdown(f"### {name} Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig_cm)

    results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
    st.subheader("🏁 Summary Table")
    st.dataframe(results_df, use_container_width=True)

else:
    st.info("모든 센서 CSV 파일(걷기, 뛰기, 정지)을 업로드해주세요.")
