#! /usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import mlflow
import mlflow.sklearn

def analyze_and_preprocess_data(csv_path, n_features=7):
    """
    와인 데이터를 분석하고 전처리하여 최적의 피처만 선택
    
    Args:
        csv_path: CSV 파일 경로
        n_features: 선택할 최종 피처 개수 (기본값: 7)
    
    Returns:
        X_processed: 전처리된 피처
        y: 타겟 변수 (quality)
        selected_features: 선택된 피처 이름 리스트
    """
    # 1. 데이터 로드
    df = pd.read_csv(csv_path)
    
    # 2. 불필요한 인덱스 컬럼 제거
    # CSV에서 자동 생성된 인덱스는 예측에 도움이 안됨
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    print(f"원본 데이터 크기: {df.shape}")
    print(f"결측치 개수:\n{df.isnull().sum()}")
    
    # 3. 결측치 처리
    # 평균값으로 대체 (결측치가 적을 경우 안전한 방법)
    df = df.fillna(df.mean(numeric_only=True))
    
    # 4. 이상치 제거 (IQR 방법)
    # 극단적인 값들은 모델 성능을 저하시킬 수 있음
    # 각 피처에 대해 IQR(Inter-Quartile Range) 기반 이상치 탐지
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # 이상치 범위: Q1 - 1.5*IQR ~ Q3 + 1.5*IQR
    # 이 범위를 벗어나는 행은 제거
    outlier_mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    df_clean = df[outlier_mask]
    print(f"이상치 제거 후 데이터 크기: {df_clean.shape} ({df.shape[0] - df_clean.shape[0]}개 행 제거)")
    
    # 5. 피처와 타겟 분리
    X = df_clean.drop('quality', axis=1)
    y = df_clean['quality']
    
    # 6. 고도로 상관된 피처 제거 (다중공선성 문제 해결)
    # 예: density와 residual sugar는 상관관계가 높을 가능성
    # fixed acidity와 pH도 높은 음의 상관관계 가능
    corr_matrix = X.corr().abs()
    
    # 상관계수 0.85 이상인 피처 쌍 찾기
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # 높은 상관관계를 가진 피처 중 하나 제거
    to_drop = [column for column in upper_tri.columns 
               if any(upper_tri[column] > 0.85)]
    
    if to_drop:
        print(f"높은 상관관계로 인해 제거되는 피처: {to_drop}")
        X = X.drop(columns=to_drop)
    
    # 7. 피처 스케일링
    # RandomForest는 스케일링이 필수는 아니지만,
    # 피처 중요도 계산 시 더 공정한 비교를 위해 적용
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # 8. 통계적 피처 선택 (F-검정 기반)
    # F-통계량이 높은 피처들만 선택 (타겟과의 선형 관계가 강한 피처)
    # n_features개의 최상위 피처만 유지하여 차원 축소
    selector = SelectKBest(score_func=f_regression, k=min(n_features, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    
    # 선택된 피처 이름 추출
    selected_mask = selector.get_support()
    selected_features = X_scaled.columns[selected_mask].tolist()
    
    # F-점수 출력 (피처 중요도 참고)
    f_scores = selector.scores_[selected_mask]
    feature_scores = list(zip(selected_features, f_scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n선택된 {len(selected_features)}개 피처 (F-점수 기준):")
    for feat, score in feature_scores:
        print(f"  - {feat}: {score:.2f}")
    
    # 9. 최종 피처 데이터프레임 생성
    X_final = pd.DataFrame(
        X_selected,
        columns=selected_features,
        index=X.index
    )
    
    return X_final, y, selected_features, scaler, selector

def run_mlflow_experiment(data_path, n_features=7):
    """
    MLflow를 사용한 와인 품질 예측 실험
    
    Args:
        data_path: 데이터 파일 경로
        n_features: 선택할 피처 개수
    """
    # 데이터 전처리 및 피처 선택
    X, y, selected_features, scaler, selector = analyze_and_preprocess_data(
        data_path, 
        n_features=n_features
    )
    
    # Train/Test 분리
    # stratify는 회귀 문제에서 사용 불가하므로 제외
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"\n학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

    with mlflow.start_run():
        # 하이퍼파라미터 설정
        # Feature 수가 줄어들었으므로 트리 개수를 늘려 성능 보완
        n_estimators = 200  # 100 -> 200 증가
        max_depth = 8       # 6 -> 8 증가 (더 복잡한 패턴 학습)
        min_samples_split = 5  # 과적합 방지
        min_samples_leaf = 2   # 과적합 방지
        random_state = 42
        
        # 모델 학습
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # 모든 CPU 코어 사용
        )
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 메트릭 계산
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)  # RMSE 추가 (해석이 더 직관적)
        
        # MLflow 로깅
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("selected_features", selected_features)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("random_state", random_state)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # 피처 중요도 로깅
        feature_importance = dict(zip(selected_features, model.feature_importances_))
        for feat, imp in feature_importance.items():
            mlflow.log_metric(f"importance_{feat}", imp)
        
        mlflow.sklearn.log_model(model, "model")
        
        # 결과 출력
        print("\n=== 모델 성능 ===")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        print("\n=== 피처 중요도 ===")
        sorted_importance = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_importance:
            print(f"  {feat}: {imp:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='와인 품질 예측 모델 (피처 선택 및 전처리 강화)'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='wine_data_homework.csv',
        help='와인 데이터 CSV 파일 경로'
    )
    parser.add_argument(
        '--n_features',
        type=int,
        default=7,
        help='선택할 피처 개수 (기본값: 7, 원본: 11)'
    )
    args = parser.parse_args()
    
    run_mlflow_experiment(args.data, args.n_features)
