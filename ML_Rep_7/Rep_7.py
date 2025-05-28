import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier  # 사용 시 주석 해제

# 1. 데이터 불러오기 및 라벨 추가
walk_df = pd.read_csv("걷기데이터.csv")
walk_df["label"] = "walk"

run_df = pd.read_csv("뛰기데이터.csv")
run_df["label"] = "run"

stop_df = pd.read_csv("정지데이터.csv")
stop_df["label"] = "stop"

# 병합
df = pd.concat([walk_df, run_df, stop_df], ignore_index=True)

# 사용할 컬럼 지정
feature_cols = [
    'Linear Acceleration x (m/s^2)',
    'Linear Acceleration y (m/s^2)',
    'Linear Acceleration z (m/s^2)'
]

# 2. 전처리: 수치형 변환 → 결측치 제거 → 이상치 제거
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=feature_cols)
z_scores = np.abs(zscore(df_clean[feature_cols]))
df_clean = df_clean[(z_scores < 3).all(axis=1)]

# 3. 시각화 (pairplot)
sns.pairplot(df_clean[feature_cols + ['label']], hue='label', diag_kind="hist")
plt.suptitle("센서 데이터 상태별 분포 (전처리 후)", y=1.02)
plt.show()

# 4. 학습 데이터 구성
X = df_clean[feature_cols]
y = df_clean['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 모델 정의 및 학습
models = {
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # y를 숫자로 바꿔야 사용 가능
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1 Score': f1_score(y_test, y_pred, average='macro')
    })

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["walk", "run", "stop"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=["walk", "run", "stop"], yticklabels=["walk", "run", "stop"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# 6. 성능 비교표 출력
results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
print("\n📊 모델 성능 비교:")
print(results_df)
