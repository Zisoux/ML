import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

st.set_page_config(layout="wide")
st.title("👷 고소 작업자 상태 분류 ML 시스템")
st.write("센서 데이터를 업로드하여 여러 ML 모델을 평가하고 예측합니다.")

# 파일 업로드
train_file = st.file_uploader("📥 학습용 CSV 업로드 (x, y, z, label 포함)", type="csv")
test_file = st.file_uploader("📥 예측용 CSV 업로드 (x, y, z)", type="csv")

if train_file:
    df = pd.read_csv(train_file)

    # 데이터 확인
    if not {'x', 'y', 'z', 'label'}.issubset(df.columns):
        st.error("❌ 학습 데이터에는 'x', 'y', 'z', 'label' 컬럼이 필요합니다.")
    else:
        # 전처리
        X = df[['x', 'y', 'z']]
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 정의
        models = {
            'KNN': KNeighborsClassifier(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }

        results = []
        predictions_dict = {}

        st.subheader("📊 모델 학습 및 평가 결과")

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            predictions_dict[name] = model  # 저장해두기

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            results.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1
            })

        results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
        st.dataframe(results_df.style.highlight_max(axis=0))

        # Confusion Matrix 시각화
        st.subheader("📉 Confusion Matrix")
        selected_for_cm = st.selectbox("🔎 Confusion Matrix 보기: 모델 선택", list(models.keys()))
        model = models[selected_for_cm]
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{selected_for_cm} - Confusion Matrix")
        st.pyplot(fig)

        # 예측용 데이터 처리
        if test_file:
            test_df = pd.read_csv(test_file)
            if not {'x', 'y', 'z'}.issubset(test_df.columns):
                st.error("❌ 예측용 CSV에는 'x', 'y', 'z' 컬럼이 필요합니다.")
            else:
                st.subheader("🔍 예측 실행")
                selected_model = st.selectbox("📌 사용할 모델 선택", results_df['Model'].tolist())
                best_model = predictions_dict[selected_model]
                X_input = test_df[['x', 'y', 'z']]
                X_scaled = scaler.transform(X_input)
                pred_labels = best_model.predict(X_scaled)
                test_df['예측된 상태'] = pred_labels

                st.success("✅ 예측 완료")
                st.dataframe(test_df)

                st.write("📈 상태 분포")
                st.bar_chart(test_df['예측된 상태'].value_counts())

