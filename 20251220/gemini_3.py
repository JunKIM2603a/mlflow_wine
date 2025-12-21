import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# 환경 설정
warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set(style="whitegrid")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def load_and_preprocess(filepath):
    """데이터 로드 및 기본적인 이상치 제거"""
    df = pd.read_csv(filepath, sep=None, engine='python')
    df.columns = [c.replace(' ', '_').lower() for c in df.columns]
    
    # 이상치 제거 (IQR)
    for col in ['volatile_acidity', 'residual_sugar', 'chlorides']:
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
    return df

def apply_domain_fe(df):
    """도메인 지식 기반 피처 엔지니어링"""
    df_fe = df.copy()
    # 1. 분자 이산화황: 살균/산화 방지 효율
    df_fe['molecular_so2'] = df_fe['free_sulfur_dioxide'] / (1 + 10**(df_fe['ph'] - 1.81))
    # 2. 산도 비율: 고정 산도 대비 휘발성 산도 (신선도 지표)
    df_fe['acidity_ratio'] = df_fe['fixed_acidity'] / (df_fe['volatile_acidity'] + 1e-5)
    # 3. 바디감 지수: 알코올과 당도의 조화
    df_fe['body_index'] = df_fe['alcohol'] * df_fe['residual_sugar']
    # 4. 결합 이산화황: 총량에서 자유량을 뺀 결합 상태
    df_fe['bound_so2'] = df_fe['total_sulfur_dioxide'] - df_fe['free_sulfur_dioxide']
    return df_fe

def run_experiment():
    # MLflow 설정
    mlflow.set_experiment("Wine_Final_Pipeline_Project")
    mlflow.sklearn.autolog(log_models=True) # Pipeline도 자동 기록됨

    # 1. 데이터 준비
    data = load_and_preprocess("../wine_data_homework.csv")
    X = data.drop("quality", axis=1)
    y = data["quality"]

    # 최종 평가용 테스트 세트 분할 (20%)
    # 품질 분포를 고려하여 stratify 적용 (회귀지만 정수 등급이므로 가능)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. ElasticNet Pipeline + RepeatedKFold 튜닝
    print("Step 1: Tuning ElasticNet with Pipeline & RepeatedKFold...")
    
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    alphas = [0.05, 0.1, 0.5]
    l1_ratios = [0.1, 0.5, 0.9]
    
    best_en_pipe = None
    best_en_score = -np.inf

    for a in alphas:
        for l in l1_ratios:
            run_name = f"EN_Pipeline_a{a}_l{l}"
            with mlflow.start_run(run_name=run_name, nested=True):
                # Pipeline 구성: 스케일링 -> 모델
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('en', ElasticNet(alpha=a, l1_ratio=l, random_state=42))
                ])
                
                # 교차 검증 (R2 기준)
                scores = cross_val_score(pipe, X_train, y_train, cv=rkf, scoring='r2')
                avg_r2 = np.mean(scores)
                
                mlflow.log_metric("avg_cv_r2", avg_r2)
                
                # 전체 트레인 셋 학습
                pipe.fit(X_train, y_train)
                
                # 변수 중요도 (ElasticNet Coefficients) 확인
                coefs = pipe.named_steps['en'].coef_
                selected_feat_count = np.sum(coefs != 0)
                mlflow.log_param("selected_features", selected_feat_count)
                
                if avg_r2 > best_en_score:
                    best_en_score = avg_r2
                    best_en_pipe = pipe

    # 3. XGBRegressor + Feature Engineering 튜닝
    print("\nStep 2: Tuning XGBRegressor with Feature Engineering...")
    
    # FE 적용 데이터 생성
    data_fe = apply_domain_fe(data)
    X_fe = data_fe.drop("quality", axis=1)
    
    X_train_fe, X_test_fe, _, _ = train_test_split(
        X_fe, y, test_size=0.2, random_state=42, stratify=y
    )

    xgb_configs = [
        {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 5},
        {'n_estimators': 250, 'learning_rate': 0.03, 'max_depth': 7}
    ]
    
    best_xgb_model = None
    best_xgb_score = -np.inf

    for cfg in xgb_configs:
        run_name = f"XGB_FE_depth{cfg['max_depth']}_lr{cfg['learning_rate']}"
        with mlflow.start_run(run_name=run_name, nested=True):
            model = XGBRegressor(**cfg, random_state=42)
            
            # 교차 검증 (FE 데이터셋 기준)
            scores = cross_val_score(model, X_train_fe, y_train, cv=rkf, scoring='r2')
            avg_r2 = np.mean(scores)
            
            mlflow.log_metric("avg_cv_r2", avg_r2)
            model.fit(X_train_fe, y_train)
            
            if avg_r2 > best_xgb_score:
                best_xgb_score = avg_r2
                best_xgb_model = model

    # 4. 최종 테스트 세트 추론 및 비교
    print("\nStep 3: Final Inference on Test Set and Visualization...")
    
    # Best ElasticNet 추론
    en_final_preds = best_en_pipe.predict(X_test)
    en_res = eval_metrics(y_test, en_final_preds)
    
    # Best XGBoost 추론
    xgb_final_preds = best_xgb_model.predict(X_test_fe)
    xgb_res = eval_metrics(y_test, xgb_final_preds)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ElasticNet 결과
    sns.scatterplot(x=y_test, y=en_final_preds, ax=axes[0], alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_title(f"ElasticNet Baseline (R2: {en_res[2]:.3f})")
    axes[0].set_xlabel("Actual Quality")
    axes[0].set_ylabel("Predicted Quality")

    # XGBoost 결과
    sns.scatterplot(x=y_test, y=xgb_final_preds, ax=axes[1], alpha=0.5, color='teal')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_title(f"XGBoost + Domain FE (R2: {xgb_res[2]:.3f})")
    axes[1].set_xlabel("Actual Quality")
    axes[1].set_ylabel("Predicted Quality")

    plt.tight_layout()
    plt.savefig("final_comparison_plot.png")
    
    # 결과 요약 로그
    with mlflow.start_run(run_name="Final_Evaluation_Summary"):
        mlflow.log_metric("best_en_test_r2", en_res[2])
        mlflow.log_metric("best_xgb_test_r2", xgb_res[2])
        mlflow.log_artifact("final_comparison_plot.png")
        
        print("\n[Final Summary]")
        print(f"ElasticNet (Baseline) R2: {en_res[2]:.4f}")
        print(f"XGBoost (Advanced) R2:   {xgb_res[2]:.4f}")
        print(f"Improvement: {((xgb_res[2] - en_res[2]) / en_res[2]) * 100:.2f}%")

if __name__ == "__main__":
    run_experiment()