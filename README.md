# MLflow 와인 품질 예측 프로젝트

이 프로젝트는 MLflow를 활용하여 와인 품질을 예측하는 머신러닝 모델을 개발하고 실험을 관리하는 프로젝트입니다.

## 📋 목차

- [MLflow란?](#mlflow란)
- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [데이터 전처리](#데이터-전처리)
- [모델 정보](#모델-정보)
- [MLflow 실험 추적](#mlflow-실험-추적)

## MLflow란?

**MLflow**는 머신러닝 라이프사이클을 관리하기 위한 오픈소스 플랫폼입니다. MLflow는 다음과 같은 주요 기능을 제공합니다:

### 1. **실험 추적 (Experiment Tracking)**
   - 하이퍼파라미터, 메트릭, 모델 아티팩트를 자동으로 기록
   - 여러 실험을 비교하고 최적의 모델을 찾기 용이
   - 웹 UI를 통해 실험 결과를 시각적으로 확인 가능

### 2. **모델 관리 (Model Registry)**
   - 모델 버전 관리 및 스테이징(Staging, Production 등)
   - 모델 배포 및 서빙을 위한 표준화된 형식 제공

### 3. **프로젝트 패키징 (Projects)**
   - 재현 가능한 ML 프로젝트를 위한 표준 형식
   - 다른 환경에서도 동일한 결과를 얻을 수 있도록 보장

### 4. **모델 서빙 (Model Serving)**
   - REST API를 통한 모델 배포
   - 다양한 플랫폼에서 모델 사용 가능

이 프로젝트에서는 주로 **실험 추적** 기능을 활용하여 다양한 피처 선택 및 하이퍼파라미터 조합을 실험하고 비교합니다.

## 프로젝트 개요

이 프로젝트는 와인의 화학적 특성(알코올, 산도, 당도 등)을 기반으로 와인의 품질(quality)을 예측하는 회귀 모델을 개발합니다. 

- **목표**: 와인 품질 예측 (회귀 문제)
- **모델**: Random Forest Regressor
- **평가 지표**: MSE, RMSE, R² Score
- **실험 관리**: MLflow를 통한 실험 추적 및 비교

## 주요 기능

### 1. 데이터 전처리
- 결측치 처리 (평균값 대체)
- 이상치 제거 (IQR 방법)
- 다중공선성 해결 (높은 상관관계 피처 제거)
- 피처 스케일링 (StandardScaler)
- 통계적 피처 선택 (F-검정 기반 SelectKBest)

### 2. 모델 학습 및 평가
- Random Forest Regressor를 사용한 회귀 모델 학습
- 교차 검증을 통한 모델 성능 평가
- 피처 중요도 분석

### 3. MLflow 실험 추적
- 하이퍼파라미터 자동 로깅
- 메트릭 자동 추적 (MSE, RMSE, R² Score)
- 피처 중요도 메트릭 기록
- 모델 아티팩트 저장

## 프로젝트 구조

```
mlflow_wine/
├── main.py                      # 메인 실행 스크립트
├── preprocessing_test.ipynb     # 데이터 전처리 탐색 노트북
├── wine_data_homework.csv       # 와인 데이터셋
├── requirements.txt             # 프로젝트 의존성
├── mlruns/                      # MLflow 실험 결과 저장 디렉토리
│   ├── 0/                       # 실험 ID별 실행 결과
│   └── models/                  # 저장된 모델
└── README.md                    # 프로젝트 문서
```

## 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd mlflow_wine
```

### 2. 가상환경 생성 및 활성화 (권장)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

주요 패키지:
- `mlflow==3.6.0`: 실험 추적 및 모델 관리
- `scikit-learn==1.7.2`: 머신러닝 모델 및 전처리
- `pandas==2.3.3`: 데이터 처리
- `numpy==2.3.5`: 수치 연산

## 사용 방법

### 기본 실행
```bash
python main.py
```

기본 설정으로 실행하면:
- 데이터 파일: `wine_data_homework.csv`
- 선택 피처 수: 7개

### 커스텀 옵션 실행
```bash
# 다른 데이터 파일 사용
python main.py --data your_data.csv

# 피처 개수 변경
python main.py --n_features 5

# 두 옵션 동시 사용
python main.py --data wine_data_homework.csv --n_features 10
```

### MLflow UI 실행
실험 결과를 웹 UI로 확인하려면:
```bash
mlflow ui
```

브라우저에서 `http://localhost:5000`으로 접속하여 실험 결과를 확인할 수 있습니다.

## 데이터 전처리

프로젝트는 다음과 같은 전처리 단계를 거칩니다:

1. **데이터 로드**: CSV 파일에서 와인 데이터 읽기
2. **인덱스 컬럼 제거**: 불필요한 인덱스 컬럼 제거
3. **결측치 처리**: 평균값으로 결측치 대체
4. **이상치 제거**: IQR(Inter-Quartile Range) 방법으로 이상치 탐지 및 제거
5. **다중공선성 해결**: 상관계수 0.85 이상인 피처 쌍 중 하나 제거
6. **피처 스케일링**: StandardScaler를 사용한 정규화
7. **피처 선택**: F-검정 기반 SelectKBest로 최상위 N개 피처 선택

자세한 전처리 과정은 `preprocessing_test.ipynb` 노트북에서 확인할 수 있습니다.

## 모델 정보

### Random Forest Regressor
- **n_estimators**: 200 (트리 개수)
- **max_depth**: 8 (최대 깊이)
- **min_samples_split**: 5 (분할을 위한 최소 샘플 수)
- **min_samples_leaf**: 2 (리프 노드의 최소 샘플 수)
- **random_state**: 42 (재현성을 위한 시드)

### 평가 지표
- **MSE (Mean Squared Error)**: 평균 제곱 오차
- **RMSE (Root Mean Squared Error)**: 평균 제곱근 오차
- **R² Score**: 결정계수 (1에 가까울수록 좋음)

## MLflow 실험 추적

MLflow는 다음 정보를 자동으로 추적합니다:

### Parameters (하이퍼파라미터)
- `n_features`: 선택된 피처 개수
- `selected_features`: 선택된 피처 목록
- `n_estimators`: 트리 개수
- `max_depth`: 최대 깊이
- `min_samples_split`: 분할 최소 샘플 수
- `min_samples_leaf`: 리프 최소 샘플 수
- `random_state`: 랜덤 시드

### Metrics (평가 지표)
- `mse`: 평균 제곱 오차
- `rmse`: 평균 제곱근 오차
- `r2_score`: 결정계수
- `importance_{feature_name}`: 각 피처의 중요도

### Artifacts (아티팩트)
- 학습된 모델이 `mlruns/` 디렉토리에 저장됩니다.

### 실험 비교
MLflow UI를 통해:
- 여러 실험의 성능을 비교
- 하이퍼파라미터 조합에 따른 성능 변화 확인
- 최적의 모델 선택 및 다운로드

## 예제 출력

실행 시 다음과 같은 정보가 출력됩니다:

```
원본 데이터 크기: (4001, 12)
이상치 제거 후 데이터 크기: (3500, 12) (501개 행 제거)
높은 상관관계로 인해 제거되는 피처: ['density']

선택된 7개 피처 (F-점수 기준):
  - alcohol: 245.32
  - volatile acidity: 189.45
  - total sulfur dioxide: 156.78
  ...

학습 데이터: (2800, 7), 테스트 데이터: (700, 7)

=== 모델 성능 ===
MSE: 0.4523
RMSE: 0.6725
R² Score: 0.4234

=== 피처 중요도 ===
  alcohol: 0.3245
  volatile acidity: 0.2134
  ...
```

## 참고사항

- 데이터 파일(`wine_data_homework.csv`)이 프로젝트 루트 디렉토리에 있어야 합니다.
- MLflow 실험 결과는 `mlruns/` 디렉토리에 저장됩니다.
- 여러 실험을 실행하면 각각 다른 run ID로 저장되어 비교 가능합니다.

## 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

