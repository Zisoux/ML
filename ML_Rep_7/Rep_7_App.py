import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 페이지 설정
st.set_page_config(page_title="Sensor Activity Classifier", layout="wide")
st.title("📊 Sensor Activity Classification App")

st.markdown("""
이 앱은 센서 데이터를 기반으로 고소작업자의 상태 (걷기, 뛰기, 정지)를 예측하기 위해 머신러닝 모델을 학습하고,
최적의 모델을 선정하여 예측 서비스를 제공합니다.
스마트폰 센서를 통해 수집한 데이터를 기반으로, 실시간 상태 판별은 아니지만 파일 업로드 기반으로 작동합니다.
""")

# 데이터 업로드
uploaded_walk = st.file_uploader("걷기 데이터 업로드", type="csv")
uploaded_run = st.file_uploader("뛰기 데이터 업로드", type="csv")
uploaded_stop = st.file_uploader("정지 데이터 업로드", type="csv")

if uploaded_walk and uploaded_run and uploaded_stop:
    walk_df = pd.read_csv(uploaded_walk)
    walk_df['label'] = 'walk'
    run_df = pd.read_csv(uploaded_run)
    run_df['label'] = 'run'
    stop_df = pd.read_csv(uploaded_stop)
    stop_df['label'] = 'stop'

    df = pd.concat([walk_df, run_df, stop_df], ignore_index=True)

    feature_cols = [
        'Linear Acceleration x (m/s^2)',
        'Linear Acceleration y (m/s^2)',
        'Linear Acceleration z (m/s^2)'
    ]

    # 전처리
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=feature_cols)

    from scipy.stats import zscore
    z_scores = np.abs(zscore(df[feature_cols]))
    df = df[(z_scores < 3).all(axis=1)]

    # 라벨 인코딩
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    # 시각화
    st.subheader("📈 Feature Pairplot")
    fig = sns.pairplot(df[feature_cols + ['label']], hue='label', diag_kind="hist")
    fig.fig.set_size_inches(2.5, 2)
    st.pyplot(fig)

    # 축 간 관계 시각화
    for (x, y) in [(0, 1), (0, 2), (1, 2)]:
        fig, ax = plt.subplots(figsize=(2.5, 2))
        sns.scatterplot(data=df, x=feature_cols[x], y=feature_cols[y], hue='label', ax=ax)
        ax.set_title(f"{feature_cols[x]} vs {feature_cols[y]}", fontsize=10)
        ax.tick_params(labelsize=8)
        st.pyplot(fig)

    # 학습용 데이터 구성
    X = df[feature_cols]
    y = df['label_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 하이퍼파라미터 튜닝 포함 모델들
    st.subheader("🔍 모델 성능 및 하이퍼파라미터 튜닝")
    model_grid = {
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5]
            }
        }
    }

    best_model = None
    best_score = 0
    best_name = ""
    results = []

    for name, cfg in model_grid.items():
        clf = GridSearchCV(cfg['model'], cfg['params'], cv=3, scoring='f1_macro')
        clf.fit(X_train_scaled, y_train)
        score = clf.best_score_

        results.append({
            'Model': name,
            'Best Params': clf.best_params_,
            'CV F1 Score': score
        })

        if score > best_score:
            best_score = score
            best_model = clf.best_estimator_
            best_name = name

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # 테스트 성능 출력
    y_pred = best_model.predict(X_test_scaled)
    st.markdown(f"### ✅ 최종 선택된 모델: {best_name}")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots(figsize=(2.5, 2))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig_cm)

    # 예측 CSV 업로드
    st.subheader("📥 예측용 센서 CSV 업로드")
    uploaded_predict = st.file_uploader("예측할 센서 데이터 업로드 (CSV, x/y/z 컬럼 포함)", type="csv")

    if uploaded_predict:
        pred_df = pd.read_csv(uploaded_predict)
        for col in feature_cols:
            pred_df[col] = pd.to_numeric(pred_df[col], errors='coerce')
        pred_df = pred_df.dropna(subset=feature_cols)

        pred_scaled = scaler.transform(pred_df[feature_cols])
        pred_labels = best_model.predict(pred_scaled)
        pred_df['predicted_label'] = le.inverse_transform(pred_labels)

        st.subheader("📤 예측 결과")
        st.dataframe(pred_df[[*feature_cols, 'predicted_label']])

        csv = pred_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("예측 결과 CSV 다운로드", csv, "prediction_results.csv", "text/csv")

else:
    st.info("모든 센서 CSV 파일(걷기, 뛰import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 페이지 설정
st.set_page_config(page_title="Sensor Activity Classifier", layout="wide")
st.title("📊 Sensor Activity Classification App")

st.markdown("""
이 앱은 센서 데이터를 기반으로 고소작업자의 상태 (걷기, 뛰기, 정지)를 예측하기 위해 머신러닝 모델을 학습하고,
최적의 모델을 선정하여 예측 서비스를 제공합니다.
스마트폰 센서를 통해 수집한 데이터를 기반으로, 실시간 상태 판별은 아니지만 파일 업로드 기반으로 작동합니다.
""")

# 데이터 업로드
uploaded_walk = st.file_uploader("걷기 데이터 업로드", type="csv")
uploaded_run = st.file_uploader("뛰기 데이터 업로드", type="csv")
uploaded_stop = st.file_uploader("정지 데이터 업로드", type="csv")

if uploaded_walk and uploaded_run and uploaded_stop:
    walk_df = pd.read_csv(uploaded_walk)
    walk_df['label'] = 'walk'
    run_df = pd.read_csv(uploaded_run)
    run_df['label'] = 'run'
    stop_df = pd.read_csv(uploaded_stop)
    stop_df['label'] = 'stop'

    df = pd.concat([walk_df, run_df, stop_df], ignore_index=True)

    feature_cols = [
        'Linear Acceleration x (m/s^2)',
        'Linear Acceleration y (m/s^2)',
        'Linear Acceleration z (m/s^2)'
    ]

    # 전처리
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=feature_cols)

    from scipy.stats import zscore
    z_scores = np.abs(zscore(df[feature_cols]))
    df = df[(z_scores < 3).all(axis=1)]

    # 라벨 인코딩
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    # 시각화
    st.subheader("📈 Feature Pairplot")
    fig = sns.pairplot(df[feature_cols + ['label']], hue='label', diag_kind="hist")
    fig.fig.set_size_inches(6, 4)
    for ax in fig.axes.flatten():
        if ax:
            ax.tick_params(labelsize=6)
            ax.set_xlabel(ax.get_xlabel(), fontsize=7)
            ax.set_ylabel(ax.get_ylabel(), fontsize=7)
    fig.fig.tight_layout()
    st.pyplot(fig)

    # 축 간 관계 시각화
    for (x, y) in [(0, 1), (0, 2), (1, 2)]:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.scatterplot(data=df, x=feature_cols[x], y=feature_cols[y], hue='label', ax=ax)
        ax.set_title(f"{feature_cols[x]} vs {feature_cols[y]}", fontsize=9)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        st.pyplot(fig)

    # 학습용 데이터 구성
    X = df[feature_cols]
    y = df['label_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 하이퍼파라미터 튜닝 포함 모델들
    st.subheader("🔍 모델 성능 및 하이퍼파라미터 튜닝")
    model_grid = {
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5]
            }
        }
    }

    best_model = None
    best_score = 0
    best_name = ""
    results = []

    for name, cfg in model_grid.items():
        clf = GridSearchCV(cfg['model'], cfg['params'], cv=3, scoring='f1_macro')
        clf.fit(X_train_scaled, y_train)
        score = clf.best_score_

        results.append({
            'Model': name,
            'Best Params': clf.best_params_,
            'CV F1 Score': score
        })

        if score > best_score:
            best_score = score
            best_model = clf.best_estimator_
            best_name = name

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # 테스트 성능 출력
    y_pred = best_model.predict(X_test_scaled)
    st.markdown(f"### ✅ 최종 선택된 모델: {best_name}")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, annot_kws={"size": 7})
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.tick_params(labelsize=6)
    fig_cm.tight_layout()
    st.pyplot(fig_cm)

    # 예측 CSV 업로드
    st.subheader("📥 예측용 센서 CSV 업로드")
    uploaded_predict = st.file_uploader("예측할 센서 데이터 업로드 (CSV, x/y/z 컬럼 포함)", type="csv")

    if uploaded_predict:
        pred_df = pd.read_csv(uploaded_predict)
        for col in feature_cols:
            pred_df[col] = pd.to_numeric(pred_df[col], errors='coerce')
        pred_df = pred_df.dropna(subset=feature_cols)

        pred_scaled = scaler.transform(pred_df[feature_cols])
        pred_labels = best_model.predict(pred_scaled)
        pred_df['predicted_label'] = le.inverse_transform(pred_labels)

        st.subheader("📤 예측 결과")
        st.dataframe(pred_df[[*feature_cols, 'predicted_label']])

        csv = pred_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("예측 결과 CSV 다운로드", csv, "prediction_results.csv", "text/csv")

else:
    st.info("모든 센서 CSV 파일(걷기, 뛰기, 정지)을 업로드해주세요.")
기, 정지)을 업로드해주세요.")