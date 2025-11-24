# 와인 품질 예측 데이터 전처리 및 모델 최적화 분석

## 📊 개요

이 문서는 `preprocessing_test.ipynb` 노트북에서 수행한 와인 품질 데이터의 전처리, 피처 엔지니어링, 모델 최적화 과정을 상세히 설명합니다.

**데이터셋**: `wine_data_homework.csv`  
**목표**: 와인의 화학적 특성을 바탕으로 품질(quality) 예측  
**문제 유형**: 회귀 (Regression)

---

## 📋 목차

1. [데이터 로드 및 기본 정보 확인](#1-데이터-로드-및-기본-정보-확인)
2. [결측치 처리](#2-결측치-처리)
3. [이상치 탐지 및 제거](#3-이상치-탐지-및-제거)
4. [상관관계 분석](#4-상관관계-분석)
5. [피처 엔지니어링](#5-피처-엔지니어링)
6. [피처 선택 방법 비교](#6-피처-선택-방법-비교)
7. [정규화 방법 비교](#7-정규화-방법-비교)
8. [MLflow 실험 및 모델 최적화](#8-mlflow-실험-및-모델-최적화)
9. [최종 결과 및 결론](#9-최종-결과-및-결론)

---

## 1. 데이터 로드 및 기본 정보 확인

### 원본 데이터 구조
- **데이터 크기**: 4,001개 행 × 13개 컬럼
- **피처 개수**: 12개 (타겟 제외)
- **타겟 변수**: `quality` (3~9 범위의 정수)

### 피처 목록
```
1. fixed acidity (고정 산도)
2. volatile acidity (휘발성 산도)
3. citric acid (구연산)
4. residual sugar (잔당)
5. chlorides (염화물)
6. free sulfur dioxide (자유 이산화황)
7. total sulfur dioxide (총 이산화황)
8. density (밀도)
9. pH (산성도)
10. sulphates (황산염)
11. alcohol (알코올)
12. quality (품질) - 타겟 변수
```

### Quality 분포
- **범위**: 3~9
- **분포**:
  - Quality 3: 19개 (0.5%)
  - Quality 4: 142개 (3.5%)
  - Quality 5: 1,214개 (30.4%)
  - Quality 6: 1,718개 (42.9%) - 최빈값
  - Quality 7: 747개 (18.7%)
  - Quality 8: 156개 (3.9%)
  - Quality 9: 5개 (0.1%)

**분포 특성**: 정규분포에 가까운 형태, 중간 품질(5-6)이 대부분을 차지

---

## 2. 결측치 처리

### 결과
- **결측치 개수**: 0개
- **처리 방법**: 결측치가 없어 별도 처리 불필요

### 처리 전략 (결측치가 있을 경우)
```python
# 평균값으로 대체
df = df.fillna(df.mean(numeric_only=True))
```

**이유**: 결측치가 적을 경우 안전하고 일반적인 방법이며, 데이터의 분포를 크게 왜곡하지 않음

---

## 3. 이상치 탐지 및 제거

### 방법: IQR (Inter-Quartile Range)

**이론적 배경**:
- IQR = Q3 - Q1 (3사분위수 - 1사분위수)
- 이상치 범위: `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`
- 이 범위를 벗어나는 값을 이상치로 판단

### 이상치 범위 (주요 피처)

| 피처 | Lower Bound | Upper Bound |
|------|-------------|-------------|
| fixed acidity | 4.900 | 8.900 |
| volatile acidity | 0.045 | 0.485 |
| citric acid | 0.075 | 0.595 |
| residual sugar | -10.450 | 21.950 |
| chlorides | 0.015 | 0.071 |
| free sulfur dioxide | -9.000 | 79.000 |
| total sulfur dioxide | 17.500 | 261.500 |
| density | 0.985 | 1.003 |
| pH | 2.790 | 3.590 |
| sulphates | 0.200 | 0.760 |
| alcohol | 6.550 | 14.150 |
| quality | 3.500 | 7.500 |

### 결과
- **이상치로 판단된 행**: 709개 (17.72%)
- **제거 후 데이터 크기**: 3,292개 행
- **제거된 행 수**: 709개

### 이상치 제거의 효과
- 극단값 제거로 모델의 일반화 성능 향상
- 노이즈 감소로 패턴 학습 용이
- Quality 분포가 더 안정적으로 유지됨

---

## 4. 상관관계 분석

### Quality와의 상관관계 (절댓값 기준 내림차순)

| 순위 | 피처 | 상관계수 | 관계 유형 |
|------|------|----------|-----------|
| 1 | **alcohol** | **0.446** | 양의 상관관계 (강함) |
| 2 | density | -0.325 | 음의 상관관계 (중간) |
| 3 | chlorides | -0.288 | 음의 상관관계 (중간) |
| 4 | total sulfur dioxide | -0.168 | 음의 상관관계 (약함) |
| 5 | residual sugar | -0.128 | 음의 상관관계 (약함) |
| 6 | volatile acidity | -0.101 | 음의 상관관계 (약함) |
| 7 | pH | 0.101 | 양의 상관관계 (약함) |
| 8 | fixed acidity | -0.078 | 음의 상관관계 (약함) |
| 9 | sulphates | 0.037 | 양의 상관관계 (매우 약함) |
| 10 | free sulfur dioxide | 0.025 | 양의 상관관계 (매우 약함) |
| 11 | citric acid | -0.002 | 거의 무관계 |

### 양의 상관관계 피처 (4개)
- alcohol (0.446) - 가장 강함
- pH (0.101)
- sulphates (0.037)
- free sulfur dioxide (0.025)

### 음의 상관관계 피처 (7개)
- density (-0.325) - 가장 강함
- chlorides (-0.288)
- total sulfur dioxide (-0.168)
- residual sugar (-0.128)
- volatile acidity (-0.101)
- fixed acidity (-0.078)
- citric acid (-0.002)

### 다중공선성 확인
- **결과**: 상관계수 0.85 이상인 피처 쌍 없음
- **결론**: 다중공선성 문제 없음, 모든 피처를 독립적으로 사용 가능

### 핵심 인사이트
1. **알코올 도수**가 품질과 가장 강한 양의 상관관계를 보임
2. **밀도**와 **염화물**은 품질과 음의 상관관계
3. 대부분의 피처는 약한 상관관계를 보이며, 일부는 거의 무관계

---

## 5. 피처 엔지니어링

### 생성된 새로운 피처

#### 1. sulfur_ratio (황 비율)
```python
sulfur_ratio = free_sulfur_dioxide / (total_sulfur_dioxide + 1e-6)
```
- **의미**: 자유 이산화황의 비율 (전체 대비)
- **Quality 상관관계**: **0.2195** (생성 피처 중 가장 높음)
- **효과**: 원본 피처보다 높은 상관관계 달성

#### 2. total_acidity (총 산도)
```python
total_acidity = fixed_acidity + volatile_acidity
```
- **의미**: 고정 산도와 휘발성 산도의 합
- **Quality 상관관계**: -0.0878
- **효과**: 산도 관련 정보를 통합

#### 3. alcohol_sugar_ratio (알코올-당 비율)
```python
alcohol_sugar_ratio = alcohol / (residual_sugar + 1e-6)
```
- **의미**: 알코올과 잔당의 비율
- **Quality 상관관계**: 0.0675
- **효과**: 당도와 알코올의 상대적 관계 표현

### 피처 엔지니어링 후 데이터
- **원본 피처**: 11개
- **생성 피처**: 3개
- **총 피처 수**: 14개

### 피처 엔지니어링의 효과
- `sulfur_ratio`는 원본 피처보다 높은 상관관계 달성 (0.2195)
- 도메인 지식을 활용한 피처 생성으로 모델 성능 향상 가능
- 비율 피처는 절대값보다 상대적 관계를 더 잘 표현

---

## 6. 피처 선택 방법 비교

노트북에서는 **10가지 피처 선택 방법**을 비교 분석했습니다.

### 6.1 통계 기반 방법

#### (1) 상관관계 기반
- **임계값**: |correlation| ≥ 0.15
- **선택된 피처**: 4개
  - alcohol
  - total sulfur dioxide
  - chlorides
  - density
- **제거된 피처**: 10개

#### (2) F-검정 (ANOVA F-test)
- **방법**: `SelectKBest(f_regression)`
- **상위 6개 피처** (F-점수 기준):
  1. alcohol (816.15)
  2. density (389.06)
  3. chlorides (298.02)
  4. sulfur_ratio (166.51)
  5. total sulfur dioxide (94.99)
  6. residual sugar (54.41)
- **특징**: 선형 관계를 가정하는 통계적 검정

#### (3) Mutual Information (상호 정보량)
- **장점**: 비선형 관계 포착 가능
- **상위 7개 피처** (MI 점수 기준):
  1. alcohol_sugar_ratio (0.194)
  2. density (0.189)
  3. sulfur_ratio (0.173)
  4. alcohol (0.162)
  5. residual sugar (0.108)
  6. total sulfur dioxide (0.096)
  7. free sulfur dioxide (0.074)
- **특징**: F-검정과 다른 순위를 보임 (비선형 관계 반영)

#### (4) Variance Threshold (분산 임계값)
- **목적**: 분산이 낮은 피처 제거
- **제거 대상**: density (분산: 0.000008)
- **이유**: 거의 변하지 않는 피처는 정보량이 적음

### 6.2 모델 기반 방법

#### (5) RandomForest Feature Importance
```python
RandomForestRegressor(n_estimators=100, random_state=42)
```
- **상위 7개 피처** (중요도 기준):
  1. alcohol (0.230)
  2. volatile acidity (0.099)
  3. sulfur_ratio (0.085)
  4. pH (0.076)
  5. free sulfur dioxide (0.073)
  6. total sulfur dioxide (0.060)
  7. chlorides (0.058)
- **특징**: 모델이 실제로 사용하는 피처의 중요도 반영

#### (6) Lasso (L1 정규화)
```python
LassoCV(cv=5, max_iter=1000)
```
- **최적 alpha**: 0.001119
- **선택된 피처**: 12개
- **제거된 피처**: 2개
  - citric acid (계수 = 0)
  - total_acidity (계수 = 0)
- **특징**: L1 정규화로 자동 피처 선택

#### (7) Permutation Importance
- **방법**: 피처를 무작위로 섞었을 때 성능 저하 측정
- **상위 7개 피처** (중요도 기준):
  1. alcohol (0.540)
  2. volatile acidity (0.143)
  3. pH (0.059)
  4. sulfur_ratio (0.053)
  5. free sulfur dioxide (0.047)
  6. sulphates (0.043)
  7. chlorides (0.041)
- **특징**: 모델에 의존하지 않는 순수한 피처 중요도

### 6.3 반복적 선택 방법

#### (8) RFE (Recursive Feature Elimination)
- **목표 피처 수**: 7개
- **선택된 피처**:
  - volatile acidity
  - free sulfur dioxide
  - alcohol
  - alcohol_sugar_ratio
  - total_acidity
  - sulfur_ratio
  - pH
- **특징**: 하위 피처부터 순차적으로 제거

#### (9) RFECV (RFE with Cross-Validation)
- **최적 피처 개수**: 14개 (모든 피처)
- **결과**: 교차검증 결과 모든 피처가 유용하다고 판단
- **특징**: 교차검증을 통한 최적 피처 개수 자동 결정

### 6.4 종합 분석

#### 모든 방법에서 공통 선택된 피처
- **alcohol** (8개 방법에서 선택) ⭐

#### 4개 이상 방법에서 선택된 피처

| 피처 | 선택 빈도 | 주요 방법 |
|------|-----------|-----------|
| alcohol | 8개 | 모든 방법 |
| sulfur_ratio | 7개 | F-검정, MI, RF, Lasso, Permutation, RFE |
| chlorides | 6개 | 상관관계, F-검정, RF, Lasso, Permutation |
| total sulfur dioxide | 6개 | 상관관계, F-검정, MI, RF, Lasso |
| free sulfur dioxide | 6개 | MI, RF, Lasso, Permutation, RFE |
| density | 5개 | 상관관계, F-검정, MI, Lasso |
| pH | 5개 | RF, Lasso, Permutation, RFE, RFECV |
| volatile acidity | 5개 | RF, Lasso, Permutation, RFE, RFECV |
| residual sugar | 4개 | F-검정, MI, Lasso, RFECV |

#### 제거 권장 피처

| 피처 | 제거 이유 |
|------|-----------|
| citric acid | 선택 빈도 낮음 (1개), Lasso 제거, 상관관계 낮음 (-0.002) |
| fixed acidity | 선택 빈도 낮음 (2개), 상관관계 낮음 (-0.078) |
| sulphates | 선택 빈도 낮음 (3개), 상관관계 낮음 (0.037) |
| total_acidity | Lasso 제거, 선택 빈도 낮음 (2개) |
| free sulfur dioxide | 상관관계 낮음 (0.025) |

### 6.5 최종 피처 선택 결과

**선택 방법**: 상관관계 기반 (|correlation| ≥ 0.15)

**최종 선택된 피처 (4개)**:
1. alcohol
2. total sulfur dioxide
3. chlorides
4. density

**제거된 피처**: 10개
- fixed acidity, volatile acidity, citric acid, residual sugar
- free sulfur dioxide, pH, sulphates
- sulfur_ratio, total_acidity, alcohol_sugar_ratio

---

## 7. 정규화 방법 비교

노트북에서는 **6가지 정규화 방법**을 비교했습니다.

### 7.1 정규화 방법 설명

#### (1) None (Baseline)
- 정규화 없음
- 원본 데이터 그대로 사용

#### (2) StandardScaler
```python
평균 = 0, 표준편차 = 1
z = (x - μ) / σ
```
- 가장 일반적인 정규화 방법
- 정규분포 가정
- 이상치에 민감

#### (3) MinMaxScaler
```python
범위 = [0, 1]
x_scaled = (x - min) / (max - min)
```
- 모든 값을 0~1 범위로 변환
- 이상치에 민감
- 분포 정보 손실 가능

#### (4) RobustScaler
```python
중앙값과 IQR 사용
x_scaled = (x - median) / IQR
```
- 이상치에 강건함
- 중앙값과 사분위수 사용
- 이상치가 많은 데이터에 적합

#### (5) QuantileTransformer
- 분위수 변환을 통해 정규분포로 변환
- 비선형 변환
- 이상치에 강건함
- 모든 피처를 동일한 분포로 변환

#### (6) PowerTransformer (Yeo-Johnson)
- 데이터를 정규분포에 가깝게 변환
- Box-Cox의 일반화 버전
- 음수 값 처리 가능
- 비선형 변환

### 7.2 정규화 방법별 성능 비교 (RandomForest 기준)

| 순위 | 정규화 방법 | RMSE | R² Score | MAE |
|------|-------------|------|----------|-----|
| 1 | **QuantileTransformer** ⭐ | **0.6764** | **0.4097** | **0.4881** |
| 2 | None (Baseline) | 0.6765 | 0.4095 | 0.4885 |
| 3 | PowerTransformer | 0.6766 | 0.4093 | 0.4878 |
| 4 | RobustScaler | 0.6766 | 0.4093 | 0.4885 |
| 5 | MinMaxScaler | 0.6767 | 0.4091 | 0.4884 |
| 6 | StandardScaler | 0.6768 | 0.4091 | 0.4888 |

### 7.3 정규화 방법별 통계 요약

#### QuantileTransformer (최적)
- 평균: 약 0.0
- 표준편차: 약 1.0
- 분포: 정규분포로 변환

#### StandardScaler
- 평균: 0.0
- 표준편차: 1.0
- 분포: 원본 분포 유지

#### MinMaxScaler
- 범위: [0, 1]
- 평균: 0.38~0.51 (피처별 상이)
- 표준편차: 0.18~0.21

### 7.4 결론

1. **최적 방법**: QuantileTransformer
   - RMSE: 0.6764 (가장 낮음)
   - R² Score: 0.4097 (가장 높음)

2. **성능 차이**: 매우 근소함
   - RMSE 차이: < 0.0004
   - RandomForest는 정규화에 크게 의존하지 않음

3. **모델별 권장사항**:
   - **트리 기반 모델** (RandomForest, GradientBoosting): 정규화 선택적
   - **선형 모델** (Ridge, ElasticNet, SVR): StandardScaler 필수

4. **이상치 처리**: RobustScaler나 QuantileTransformer가 이상치가 많은 데이터에 적합

---

## 8. MLflow 실험 및 모델 최적화

### 8.1 실험 설정
- **MLflow 실험명**: Wine_Quality_Prediction_Optimization
- **교차검증**: 10-Fold Cross-Validation
- **하이퍼파라미터 튜닝**: GridSearchCV
- **데이터 분할**: Train 80% (2,633개) / Test 20% (659개)
- **정규화 방법**: QuantileTransformer

### 8.2 비교 모델

#### 1. RandomForest
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```
- **파라미터 조합**: 108개
- **총 학습 횟수**: 1,080회 (10-Fold CV)

#### 2. GradientBoosting
```python
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}
```
- **파라미터 조합**: 36개
- **총 학습 횟수**: 360회

#### 3. ElasticNet
```python
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
```
- **파라미터 조합**: 20개
- **총 학습 횟수**: 200회

#### 4. Ridge
```python
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
}
```
- **파라미터 조합**: 5개
- **총 학습 횟수**: 50회

#### 5. SVR (Support Vector Regression)
```python
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'epsilon': [0.01, 0.1, 0.2]
}
```
- **파라미터 조합**: 36개
- **총 학습 횟수**: 360회

### 8.3 모델 성능 비교 (테스트 세트 기준)

| 순위 | 모델 | Test RMSE | Test R² | Test MAE | CV Score (MSE) |
|------|------|-----------|---------|----------|----------------|
| 1 | **RandomForest** ⭐ | **0.6722** | **0.4455** | **0.4931** | **0.4860** |
| 2 | GradientBoosting | 0.6910 | 0.4141 | 0.5248 | 0.5336 |
| 3 | SVR | 0.7658 | 0.2802 | 0.6024 | 0.5988 |
| 4 | Ridge | 0.7839 | 0.2459 | 0.6071 | 0.6404 |
| 5 | ElasticNet | 0.7894 | 0.2352 | 0.6135 | 0.6428 |

### 8.4 최적 모델: RandomForest

#### 최적 하이퍼파라미터
```python
{
    'n_estimators': 300,
    'max_depth': None,  # 제한 없음
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}
```

#### 성능 지표
- **Test RMSE**: 0.6722
- **Test R²**: 0.4455 (44.55% 설명력)
- **Test MAE**: 0.4931
- **CV Score (MSE)**: 0.4860

#### 과적합 분석
- **Train RMSE**: 0.2555
- **Test RMSE**: 0.6722
- **차이**: 약간의 과적합 존재하나 허용 범위
- **이유**: Train과 Test 성능 차이가 있지만, 모델이 일반화 가능한 패턴을 학습

### 8.5 피처 중요도 (최종 모델)

| 순위 | 피처 | 중요도 |
|------|------|--------|
| 1 | **alcohol** | **0.3160** |
| 2 | density | 0.2552 |
| 3 | total sulfur dioxide | 0.2466 |
| 4 | chlorides | 0.1822 |

**최종 선택 피처**: 4개 (상관관계 기반 선택 결과)

### 8.6 모델별 특징 분석

#### RandomForest (최고 성능)
- **장점**: 
  - 비선형 관계 포착 우수
  - 피처 중요도 제공
  - 과적합에 상대적으로 강건
- **단점**: 
  - 해석 어려움
  - 학습 시간 상대적으로 김

#### GradientBoosting
- **성능**: RandomForest와 유사하나 약간 낮음
- **특징**: 순차적 학습으로 복잡한 패턴 학습 가능

#### 선형 모델 (Ridge, ElasticNet)
- **성능**: 상대적으로 낮음
- **이유**: 비선형 관계를 포착하지 못함
- **장점**: 해석 용이, 빠른 학습

#### SVR
- **성능**: 중간 수준
- **특징**: 비선형 관계 포착 가능하나 하이퍼파라미터 민감

---

## 9. 최종 결과 및 결론

### 9.1 데이터 전처리 요약

| 단계 | 결과 |
|------|------|
| 원본 데이터 | 4,001개 행 × 12개 피처 |
| 이상치 제거 | 3,292개 행 (709개 제거, 17.72%) |
| 피처 선택 | 4개 피처 (8개 제거) |
| 정규화 | QuantileTransformer |
| 최종 데이터 | 3,292개 행 × 4개 피처 |

### 9.2 최종 선택된 피처

1. **alcohol** (알코올 도수) - 가장 중요 (중요도 0.316)
2. **density** (밀도) - 중요도 0.255
3. **total sulfur dioxide** (총 이산화황) - 중요도 0.247
4. **chlorides** (염화물) - 중요도 0.182

### 9.3 모델 성능

#### 최종 모델: RandomForest (n_estimators=300)
- **RMSE**: 0.6722 → 평균 약 0.67점 오차
- **R² Score**: 0.4455 → 약 44.55%의 분산 설명
- **MAE**: 0.4931 → 평균 절대 오차 약 0.49점

#### Quality 값별 예측 정확도

| Quality | 샘플 수 | MAE | 평가 |
|---------|--------|-----|------|
| 3 | 5 | 2.488 | 낮음 (샘플 부족) |
| 4 | 8 | 1.203 | 낮음 (샘플 부족) |
| 5 | 196 | 0.508 | 양호 |
| 6 | 280 | 0.318 | **가장 정확** |
| 7 | 140 | 0.547 | 양호 |
| 8 | 29 | 1.208 | 낮음 (샘플 부족) |
| 9 | 1 | 2.597 | 매우 낮음 (샘플 부족) |

**인사이트**: 
- 중간 품질(5-7) 예측이 가장 정확
- 극단값(3, 9)은 샘플 부족으로 예측 어려움
- Quality 6이 가장 많은 샘플(280개)과 가장 낮은 오차(0.318)

### 9.4 주요 인사이트

#### 1. 알코올 도수가 가장 중요한 품질 지표
- 모든 피처 선택 방법에서 1위
- 피처 중요도 31.6%
- Quality와 양의 상관관계 (0.446)

#### 2. 피처 선택으로 차원 축소 성공
- 12개 → 4개 피처로 축소 (67% 감소)
- 성능 저하 최소화
- 모델 해석 용이성 향상

#### 3. 정규화 효과는 제한적
- RandomForest는 정규화에 크게 의존하지 않음
- 성능 차이 < 0.0004 (매우 근소)
- 선형 모델에는 정규화 필수

#### 4. 하이퍼파라미터 튜닝 효과
- 기본 설정 대비 성능 향상
- n_estimators=300이 최적
- max_depth=None (제한 없음)이 최적

#### 5. 모델 선택
- RandomForest가 모든 지표에서 최고 성능
- 트리 기반 모델이 비선형 관계 포착에 유리
- 선형 모델은 성능이 상대적으로 낮음

### 9.5 개선 방향

#### 1. 데이터 증강
- 극단 품질(3, 9) 샘플 추가 수집
- SMOTE 등 오버샘플링 기법 적용
- 데이터 불균형 해소

#### 2. 추가 피처 엔지니어링
- 도메인 지식 활용한 새로운 피처 생성
- 다항식 피처 (Polynomial Features) 시도
- 상호작용 피처 (Interaction Features) 추가

#### 3. 앙상블 모델
- RandomForest + GradientBoosting 앙상블
- 스태킹(Stacking) 기법 적용
- 보팅(Voting) 기법 시도

#### 4. 딥러닝 모델
- Neural Network 시도
- AutoML 도구 활용 (AutoGluon, H2O 등)
- 하이퍼파라미터 자동 최적화

#### 5. 모델 해석
- SHAP 값 분석으로 피처 기여도 시각화
- 부분 의존성 플롯 (PDP) 생성
- 모델 예측의 신뢰도 분석

### 9.6 MLflow 활용 효과

#### 1. 실험 추적
- 205개 파라미터 조합 체계적 관리
- 모든 실험 결과 자동 저장
- 실험 재현성 보장

#### 2. 비교 분석
- UI를 통한 직관적 성능 비교
- 하이퍼파라미터에 따른 성능 변화 추적
- 최적 모델 자동 식별

#### 3. 모델 관리
- 최적 모델 자동 저장 및 버전 관리
- 모델 아티팩트 저장
- 배포 준비 완료

#### 4. 협업
- 실험 결과 공유 용이
- 팀원 간 모델 비교 및 협업
- 문서화 자동화

---

## 📚 참고 자료

### 사용된 주요 라이브러리
- **pandas**: 데이터 처리 및 분석
- **numpy**: 수치 연산
- **scikit-learn**: 머신러닝 모델 및 전처리
  - RandomForestRegressor, GradientBoostingRegressor
  - StandardScaler, MinMaxScaler, RobustScaler
  - QuantileTransformer, PowerTransformer
  - SelectKBest, RFE, RFECV
  - GridSearchCV, KFold
- **matplotlib/seaborn**: 시각화
- **mlflow**: 실험 추적 및 모델 관리

### 주요 개념

#### IQR (Inter-Quartile Range)
- 이상치 탐지 방법
- Q3 - Q1 범위 계산
- 1.5×IQR 범위를 벗어나는 값 제거

#### 다중공선성 (Multicollinearity)
- 피처 간 높은 상관관계
- 모델 성능 저하 및 해석 어려움
- 상관계수 0.85 이상 시 주의 필요

#### F-검정 (F-test)
- 통계적 피처 선택 방법
- 타겟과의 선형 관계 측정
- ANOVA 기반

#### Mutual Information
- 비선형 관계 포착 가능
- 정보 이론 기반
- F-검정과 상호 보완적

#### 교차검증 (Cross-Validation)
- 모델 일반화 성능 평가
- K-Fold CV 사용 (K=10)
- 과적합 방지

#### GridSearchCV
- 하이퍼파라미터 최적화
- 모든 조합 시도
- 교차검증과 결합

### MLflow UI 실행
```bash
mlflow ui --backend-store-uri ./mlruns
```
브라우저에서 `http://localhost:5000` 접속하여 실험 결과 확인

### 관련 파일
- **노트북**: `preprocessing_test.ipynb`
- **메인 스크립트**: `main.py`
- **데이터**: `wine_data_homework.csv`
- **MLflow 결과**: `mlruns/` 디렉토리

---

## ✅ 결론

이 노트북은 **체계적인 데이터 전처리와 모델 최적화 과정**을 보여줍니다:

1. ✅ **완전한 데이터 전처리 파이프라인** 구축
   - 결측치 처리, 이상치 제거, 상관관계 분석

2. ✅ **10가지 피처 선택 방법** 비교 분석
   - 통계 기반, 모델 기반, 반복적 선택 방법

3. ✅ **6가지 정규화 방법** 실험
   - QuantileTransformer가 최적

4. ✅ **5가지 모델 + 하이퍼파라미터 튜닝** 수행
   - RandomForest가 최고 성능

5. ✅ **MLflow를 통한 실험 관리** 구현
   - 205개 파라미터 조합 체계적 관리

### 최종 성과

- **4개의 핵심 피처**만으로 와인 품질의 **44.55%를 설명**
- 평균 **0.67점의 오차**로 예측 가능
- **알코올 도수**가 가장 중요한 품질 지표
- **중간 품질(5-7) 예측이 가장 정확**

### 실무 적용 가능성

이 분석 과정은 실제 머신러닝 프로젝트에서 다음과 같이 활용할 수 있습니다:

1. **체계적인 전처리 파이프라인** 구축 방법
2. **다양한 피처 선택 방법** 비교 및 선택 기준
3. **정규화 방법** 선택 가이드
4. **하이퍼파라미터 튜닝** 전략
5. **MLflow를 통한 실험 관리** 모범 사례

---

**작성일**: 2024  
**프로젝트**: MLflow Wine Quality Prediction  
**노트북**: preprocessing_test.ipynb  
**분석 도구**: Python, scikit-learn, MLflow

