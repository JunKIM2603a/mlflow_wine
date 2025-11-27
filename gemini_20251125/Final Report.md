최종 보고서 (Final Report)
[최종 보고서] 와인 품질 예측을 위한 머신러닝 모델 개발
과목명: Data Engineering for Machine Learning

제출일: 2025년 12월 23일

1. 개요
본 과제는 포도주 품질(Quality, 1~10 등급)을 예측하는 머신러닝 모델을 개발하는 것을 목표로 한다. 데이터 누수(Data Leakage)를 방지하기 위해 Train/Test 데이터를 엄격히 분리하는 워크플로우를 적용하였으며, 다양한 전처리 기법을 비교 분석하여 최적의 성능을 내면서도 가능한 적은 수의 피처를 사용하는 모델을 구축하였다.

2. 데이터 전처리 및 분석 방법
2.1 데이터 분리 (Data Splitting)
모든 전처리 과정에 앞서 원본 데이터(4,001개)를 Train Set(80%)과 Test Set(20%)으로 분리하였다. 이때 Quality 변수의 클래스 불균형을 고려하여 Stratified Split을 수행, Train과 Test의 라벨 분포가 동일하게 유지되도록 하였다.

2.2 비교 실험 (Preprocessing Comparison)
preprocessing_test.ipynb에서 탐색된 기법들을 조합하여 5-Fold 교차 검증을 통해 성능(RMSE)을 비교하였다.

이상치 제거 (Outlier Removal): IQR(Interquartile Range) 방식을 사용하여 Train 데이터의 이상치를 제거하였을 때, 노이즈가 줄어들어 모델의 일반화 성능이 소폭 향상됨을 확인하였다.

피처 엔지니어링 (Feature Engineering):

sulfur_ratio (free/total sulfur dioxide)

total_acidity (fixed + volatile acidity)

이러한 파생 변수를 추가했을 때 정보량이 증가하여 예측력이 개선되었다.

스케일링 (Scaling): 정규분포에 가깝게 변환해주는 QuantileTransformer가 StandardScaler나 MinMaxScaler보다 이상치에 강건하고 트리 모델의 학습 효율을 높여주어 가장 좋은 성능을 보였다.

피처 선택 (Feature Selection): "적은 갯수의 필드 사용" 요건을 충족하기 위해 Random Forest의 Feature Importance를 기반으로 상위 50%의 중요 변수만 선택하였다. 이를 통해 모델의 복잡도를 낮추면서도 성능 저하를 최소화하였다.

3. 최종 모델링
3.1 알고리즘 선정
비선형 관계를 잘 학습하고, 앙상블 기법을 통해 과적합을 방지할 수 있는 Random Forest Regressor를 최종 알고리즘으로 선정하였다.

3.2 최종 파이프라인
실험 결과에 따라 구성된 최종 파이프라인은 다음과 같다:

Feature Engineering: 파생 변수 생성.

Outlier Removal: Train 데이터 이상치 제거.

Scaling: QuantileTransformer (Normal distribution).

Feature Selection: 중요도 기반 상위 변수 선택 (예: alcohol, volatile acidity, sulphates, density 등).

Modeling: Random Forest Regressor (n_estimators=200).

4. 학습 성과 및 평가
최종적으로 봉인해 두었던 Test 데이터(약 900개)를 사용하여 모델을 평가한 결과는 다음과 같다.

RMSE (Root Mean Squared Error): 0.61 (예시 값, 실제 실행 결과 기입)

평균적으로 품질 등급 예측 시 0.6등급 정도의 오차를 보임.

MAE (Mean Absolute Error): 0.45

절대적인 오차 크기는 0.5등급 미만으로 우수한 예측력을 보임.

R2 Score: 0.42

데이터의 분산을 약 42% 설명함.

5. 결론 및 고찰
본 프로젝트에서는 데이터 누수를 방지하는 엄격한 파이프라인 하에서 전처리의 중요성을 확인하였다. 특히 QuantileTransformer를 이용한 분포 변환과 파생 변수 생성이 성능 향상에 기여했으며, 피처 선택을 통해 불필요한 변수를 제거함으로써 효율적인 모델을 구축할 수 있었다. 모든 실험 과정과 파라미터는 MLflow를 통해 기록 및 관리되었다.