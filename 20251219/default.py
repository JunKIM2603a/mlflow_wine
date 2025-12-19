import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import mlflow
import mlflow.sklearn

# 머신러닝 및 전처리 관련 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import classification_report, cohen_kappa_score, make_scorer, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline




# 시각화 설정 
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rc('font', family='Malgun Gothic') # 한글 폰트 설정
plt.rc('axes', unicode_minus=False)
warnings.filterwarnings('ignore')

# 재현성을 위한 시드 고정
SEED = 42
np.random.seed(SEED)

# MLflow 설정 (로컬 저장소 사용)
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Wine_Quality_20251219_1")

# 1. 데이터 로드
try:
    df_full = pd.read_csv('../wine_data_homework.csv', index_col=0)
    print("데이터 로드 완료. Shape:", df_full.shape)
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")

print("\n=== 기본 통계 정보 ===")
print(df_full.head())

# 기본 정보 확인
print("=== 데이터 구조 파악 ===")
df_full.info()
df_full.describe()

# 결측치 확인
print("\n=== 데이터 결측치 확인 ===")
missing_values = df_full.isnull().sum()
print("Missing Values per Column:")
print(missing_values[missing_values > 0])
if missing_values.sum() > 0:
    print(f"총 결측치: {missing_values.sum()}개")
else:
    print("결측치가 없습니다.")

def plot_target_distribution(y_data, title="타겟 변수 분포 분석", xlabel="Quality Score", ylabel="Count", figsize=(10, 5), palette='viridis'):
    """
    타겟 변수의 분포를 Count Plot으로 시각화하고, 각 클래스의 비율(%)을 표시합니다.

    Args:
        y_data (pd.Series or np.array): 시각화할 타겟 변수 데이터 (e.g., y_train).
        title (str): 그래프의 제목.
        xlabel (str): x축 레이블.
        ylabel (str): y축 레이블.
        figsize (tuple): 그래프의 크기.
        palette (str): Seaborn 색상 팔레트.
    """
    if not isinstance(y_data, pd.Series):
        try:
            # y_data가 Series가 아니면 Series로 변환 시도
            y_data = pd.Series(y_data)
        except:
            print("오류: y_data는 pandas Series 또는 Series로 변환 가능한 형태여야 합니다.")
            return

    plt.figure(figsize=figsize)
    
    # 1. Count Plot 생성
    ax = sns.countplot(x=y_data, palette=palette)
    
    # 2. 막대 위에 비율(%) 표시
    total = len(y_data)
    low_class_percentages = 0
    high_class_percentages = 0
    low_class_count = 0
    high_class_count = 0
    for p in ax.patches:
        height = p.get_height()
        # 높이가 0이 아닌 경우에만 비율을 계산하고 표시
        if height > 0:
            percentage = '{:.1f}%'.format(100 * height / total)
            percentage_float = (100 * height / total)
            # x 좌표: 막대의 중앙
            x = p.get_x() + p.get_width() / 2
            # y 좌표: 막대 끝에서 약간 위에 위치
            y = height
            ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=10)
            if percentage_float < (100 / 7):
                low_class_percentages = low_class_percentages + percentage_float
                low_class_count = low_class_count + 1
            else:
                high_class_percentages = high_class_percentages + percentage_float
                high_class_count = high_class_count + 1

    print("Low class percentages: ", low_class_percentages)
    print("High class percentages: ", high_class_percentages)
    print(f"Imbalance Ratio (IR): (다수 클래스 수({high_class_count})) / (소수 클래스 수({low_class_count})): {high_class_percentages} / {low_class_percentages}")
    
    # 3. 제목 및 레이블 설정
    plt.title(f'{title} - 클래스 불균형 확인', fontsize=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# 수치형 변수들의 분포 (Histogram & KDE)
def plot_numerical_distribution(X_data, n_cols=3, bins=30, color='skyblue', figsize_per_row=4):
    """
    수치형 독립 변수들의 분포를 히스토그램과 KDE 플롯으로 시각화합니다.
    
    Args:
        X_data (pd.DataFrame): 독립 변수 데이터 (e.g., X_train).
        n_cols (int): 한 행에 표시할 서브플롯의 개수. 기본값은 3.
        bins (int): 히스토그램의 막대(bin) 개수.
        color (str): 플롯의 색상.
        figsize_per_row (int): 한 행당 높이 비율을 결정하는 값.
    """
    if X_data.empty:
        print("오류: X_data가 비어 있습니다.")
        return
        
    # 수치형 변수만 선택
    # 와인 데이터셋은 보통 모든 피처가 수치형이지만, 안전을 위해 필터링
    numerical_features = X_data.select_dtypes(include=np.number).columns
    
    if numerical_features.empty:
        print("오류: X_data에 시각화할 수 있는 수치형 변수가 없습니다.")
        return

    n_features = len(numerical_features)
    # 필요한 행의 개수 계산
    n_rows = (n_features + n_cols - 1) // n_cols

    # 전체 Figure 크기 설정
    plt.figure(figsize=(20, figsize_per_row * n_rows))
    
    print(f"총 {n_features}개의 수치형 피처 분포를 생성합니다.")

    # 각 피처에 대해 반복하며 서브플롯 생성
    for i, col in enumerate(numerical_features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Histplot 생성 (kde=True로 KDE 곡선 함께 표시)
        sns.histplot(X_data[col], kde=True, bins=bins, color=color)
        
        plt.title(f'{col} Distribution', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Density / Count')
        
    plt.tight_layout()
    plt.show()

# 피처별 Quality 구분 능력 분석 (Boxplot)
def plot_feature_quality_boxplots(X_data, y_data, n_cols=3, figsize_per_row=5):
    """
    각 독립 변수(Feature)가 타겟 변수(Quality)에 따라 어떻게 분포하는지 Boxplot으로 시각화합니다.

    Args:
        X_data (pd.DataFrame): 독립 변수 데이터 (e.g., X_train).
        y_data (pd.Series): 타겟 변수 데이터 (e.g., y_train).
        n_cols (int): 한 행에 표시할 서브플롯의 개수. 기본값은 3.
        figsize_per_row (int): 한 행당 높이 비율을 결정하는 값. (전체 높이 = n_rows * figsize_per_row)
    """
    if X_data.empty:
        print("오류: X_data가 비어 있습니다.")
        return
    
    features = X_data.columns
    n_features = len(features)
    # 필요한 행의 개수 계산
    n_rows = (n_features + n_cols - 1) // n_cols

    # 전체 Figure 크기 설정
    plt.figure(figsize=(20, figsize_per_row * n_rows))

    print(f"총 {n_features}개의 피처에 대해 Quality별 Boxplot을 생성합니다.")

    # 각 피처에 대해 반복하며 서브플롯 생성
    for i, col in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Boxplot으로 중앙값과 분포 비교
        # x축은 타겟 변수, y축은 현재 피처
        sns.boxplot(x=y_data, y=X_data[col], palette='coolwarm')
        
        plt.title(f'{col} vs Quality', fontsize=14, fontweight='bold')
        plt.xlabel('Quality Score')
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
selected_corr_features = {}
def visualize_and_classify_correlation(X_data, y_data, target_name='quality', high_threshold=0.2, medium_threshold=0.1):
    """
    변수 간 상관관계 Heatmap과 타겟 변수와의 상관관계 Barh 플롯을 생성하고,
    상관계수 임계값에 따라 피처를 분류하여 선택된 피처 딕셔너리를 반환합니다.

    Args:
        X_data (pd.DataFrame): 독립 변수 데이터 (e.g., X_train).
        y_data (pd.Series): 타겟 변수 데이터 (e.g., y_train).
        target_name (str): 타겟 변수의 이름. 기본값은 'quality'.
        high_threshold (float): 높은 상관관계의 임계값 (절댓값 기준).
        medium_threshold (float): 중간 상관관계의 임계값 (절댓값 기준).

    Returns:
        dict: 높은 상관관계와 중간 상관관계를 가진 피처와 그 절대값 상관계수 (selected_corr_features).
              Format: {'feature_name': abs(correlation_value)}
    """
    
    # 1. 상관관계 계산
    combined_df = pd.concat([X_data, y_data.rename(target_name)], axis=1)
    correlation_matrix = combined_df.corr()
    
    # 타겟 변수와의 상관관계 추출 및 내림차순 정렬
    target_correlation = correlation_matrix[target_name].drop(target_name).sort_values(ascending=False)

    # 2. 서브플롯 설정
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('변수 간 상관관계 분석 및 타겟 변수와의 관계 시각화', fontsize=18, y=1.02)

    # --- Subplot 1: 상관관계 Heatmap (Triangle Heatmap) ---
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='RdBu_r', 
                ax=axes[0], fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, center=0,
                cbar_kws={'shrink': 0.8})
    axes[0].set_title('변수 간 다중공선성 확인 (Correlation Heatmap)', fontsize=15)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # --- Subplot 2: 타겟 변수와의 상관관계 Barh ---
    
    # Bar Plot 생성 (수평 막대 그래프)
    bars = axes[1].barh(target_correlation.index, target_correlation.values, 
                        color=np.where(target_correlation.values > 0, 'skyblue', 'salmon'))

    axes[1].set_title(f'타겟 변수 ({target_name})와의 상관관계', fontsize=15)
    axes[1].set_xlabel('상관계수 값')
    axes[1].set_ylabel('특성 변수')
    axes[1].set_xlim(-1, 1) # x축 범위를 -1에서 1로 설정

    # 임계값 표시 (수직선)
    axes[1].axvline(high_threshold, color='green', linestyle='--', linewidth=1.5, label=f'High Threshold ({high_threshold})')
    axes[1].axvline(-high_threshold, color='green', linestyle='--', linewidth=1.5)
    axes[1].axvline(medium_threshold, color='orange', linestyle=':', linewidth=1.5, label=f'Medium Threshold ({medium_threshold})')
    axes[1].axvline(-medium_threshold, color='orange', linestyle=':', linewidth=1.5)
    axes[1].legend(loc='lower right')

    # 막대 끝에 값 표시
    for bar in bars:
        width = bar.get_width()
        axes[1].text(width, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}',
                     va='center', ha='left' if width > 0 else 'right',
                     fontsize=10, color='black' if abs(width) < 0.8 else 'white')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    # --- 상관관계에 따른 변수 분류 및 저장 ---
    # abs()를 사용하여 절대값 기준으로 필터링
    high_corr_features = target_correlation[target_correlation.abs() > high_threshold].index.tolist()
    medium_corr_features = target_correlation[(target_correlation.abs() > medium_threshold) & (target_correlation.abs() <= high_threshold)].index.tolist()
    low_corr_features = target_correlation[target_correlation.abs() <= medium_threshold].index.tolist()

    print("\n=== 상관관계에 따른 변수 분류 ===")
    print(f"High correlation features with {target_name} (> {high_threshold}): {high_corr_features}")
    print(f"Medium correlation features with {target_name} ({medium_threshold} < |r| <= {high_threshold}): {medium_corr_features}")
    print(f"Low correlation features with {target_name} (<= {medium_threshold}): {low_corr_features}")

    # [수정됨] 선택된 상관관계 변수 딕셔너리 생성 (Feature Name : Abs(Correlation))
    # High와 Medium에 해당하는 변수들을 합친 리스트
    selected_features_list = high_corr_features + medium_corr_features
    
    # 딕셔너리 컴프리헨션을 이용해 {이름: 절대값} 형태로 저장
    selected_corr_features = {
        feature: abs(target_correlation[feature]) 
        for feature in selected_features_list
    }
    
    # 절대값 크기 순으로 내림차순 정렬 (선택 사항, 보기 좋게 하기 위함)
    selected_corr_features = dict(sorted(selected_corr_features.items(), key=lambda item: item[1], reverse=True))

    print(f"\n선택된 상관관계 변수 딕셔너리 (High + Medium, Abs Value):")
    print(selected_corr_features)
    
    return selected_corr_features

def calculate_vif(df):
    """VIF 계산 함수 - 다중공선성 분석"""
    # 데이터프레임이 비어있거나 단일 열인 경우 처리
    if df.shape[1] == 0:
        return pd.DataFrame({'Features': [], 'VIF': []})
    
    vif = pd.DataFrame()
    vif["Features"] = df.columns
    
    # 단일 변수는 VIF가 정의되지 않거나 1.0
    if df.shape[1] == 1:
        vif["VIF"] = 1.0
    else:
        # variance_inflation_factor는 numpy 배열을 사용합니다.
        vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        
    return vif.sort_values(by="VIF", ascending=False).reset_index(drop=True)

def run_vif_analysis(X_data, vif_threshold=10):
    """
    독립 변수의 VIF를 계산하고 시각화하며, VIF 임계값 미만인 피처 목록을 반환합니다. 
    (스케일링 전 기준 분석)

    Args:
        X_data (pd.DataFrame): 독립 변수 데이터 (e.g., X_train).
        vif_threshold (int): 다중공선성 판단 VIF 임계값. 기본값은 10.

    Returns:
        list: VIF 임계값 미만인 피처 목록.
    """
    if X_data.empty:
        print("오류: X_data가 비어 있습니다.")
        return []

    # 1. VIF 계산
    vif_results = calculate_vif(X_data)
    
    print(f"\nVIF 분석 결과 (VIF > {vif_threshold}: 다중공선성 의심):")
    print(vif_results)
    
    # 2. VIF 시각화
    plt.figure(figsize=(12, 6))
    
    # VIF 임계값 기준으로 색상 설정
    colors = ['red' if v > vif_threshold else 'green' for v in vif_results['VIF']]
    
    plt.barh(vif_results['Features'], vif_results['VIF'], color=colors)
    plt.axvline(x=vif_threshold, color='red', linestyle='--', label=f'VIF = {vif_threshold} (임계값)')
    
    plt.xlabel('VIF')
    plt.title('VIF (Variance Inflation Factor) - 다중공선성 분석', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. VIF 기준 피처 선택
    vif_selected = vif_results[vif_results['VIF'] < vif_threshold]['Features'].tolist()
    vif_not_selected = vif_results[vif_results['VIF'] >= vif_threshold]['Features'].tolist()

    print(f"\nVIF < {vif_threshold}인 피처 ({len(vif_selected)}개): {vif_selected}")
    print(f"\nVIF >= {vif_threshold}인 피처 ({len(vif_not_selected)}개): {vif_not_selected}")

    return vif_selected, vif_not_selected

# 데이터 전처리를 확인
# 피처별 Quality 구분 능력 분석 (Boxplot)
# 수치형 변수들의 분포 (Histogram & KDE)
# 상관관계 확인
# VIF 분석
# 스케일링 후 VIF 계산 (스케일링이 VIF에 영향을 줄 수 있음)
def check_data(X_data, y_data, stage="None"):
    """
    데이터 전처리를 확인합니다.

    Args:
        X_data (pd.DataFrame): 독립 변수 데이터 (e.g., X_train).
        y_data (pd.Series): 타겟 변수 데이터 (e.g., y_train).
    """
    # 피처별 Quality 구분 능력 분석 (Boxplot)
    plot_feature_quality_boxplots(X_data, y_data)
    # 수치형 변수들의 분포 (Histogram & KDE)
    plot_numerical_distribution(X_data)

    # VIF 분석
    selected_vif_features, _ = run_vif_analysis(X_data)
    # # 스케일링 후 VIF 계산 (스케일링이 VIF에 영향을 줄 수 있음)
    # print("=" * 20, "스케일링 후 VIF","=" * 20)
    # scaler_temp = StandardScaler()
    # X_scaled_temp = pd.DataFrame(
    #     scaler_temp.fit_transform(X_data),
    #     columns=X_data.columns
    # )
    # selected_vif_features = run_vif_analysis(X_scaled_temp)

    # 상관관계 확인
    selected_corr_features = visualize_and_classify_correlation(X_data[selected_vif_features], y_data, 
                                                        target_name='quality', 
                                                        high_threshold=0.2, 
                                                        medium_threshold=0.1)

    # 상관관계와 VIF 를 통한 변수 선택
    print(f"상관관계와 VIF를 통한 변수 선택({stage}): {len(selected_corr_features)}개 {selected_corr_features}")

def detect_outliers_iqr(df, n=1.5):
    """
    IQR 방법을 사용하여 데이터프레임 내의 이상치를 탐지하고,
    이상치로 판단된 행의 위치 인덱스(positional index)를 반환합니다.
    
    Args:
        df (pd.DataFrame): 이상치를 탐지할 데이터프레임 (예: X_train).
        n (float): IQR 배수 (일반적으로 1.5).
        
    Returns:
        np.array: 이상치로 판단된 행의 위치 기반 인덱스 배열.
    """
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # 이상치 범위: Q1 - n*IQR ~ Q3 + n*IQR
    lower_bound = Q1 - n * IQR
    upper_bound = Q3 + n * IQR
    
    print("=== 이상치 탐지 범위 (Q1 - 1.5*IQR ~ Q3 + 1.5*IQR) ===")
    outlier_info = pd.DataFrame({
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    })
    print(outlier_info)

    # 1. 이상치 조건 (어떤 컬럼이라도 범위 밖에 있는 경우)
    outlier_condition = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
    
    # 2. 이상치로 판단된 행의 위치 기반 인덱스를 찾기
    # np.where는 True인 값들의 인덱스를 튜플 형태로 반환합니다.
    # [0]을 사용하여 행 인덱스만 추출합니다.
    outlier_pos_indices = np.where(outlier_condition)[0]
    
    return outlier_pos_indices

# 4.2 Isolation Forest 방식
def detect_outliers_if(X):
    # contamination: 이상치 비율 예상치 (약 5%로 가정)
    clf = IsolationForest(max_samples='auto', random_state=SEED, contamination=0.05)
    preds = clf.fit_predict(X)
    return np.where(preds == -1)[0] # -1이 이상치

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# --- 새로운 함수 추가: 두 방법 비교 시각화 ---
def compare_outliers_scatter(X_data, detect_iqr_func, detect_if_func, title="IQR vs Isolation Forest 이상치 비교 (PCA)"):
    """
    PCA를 사용하여 데이터 차원을 2D로 축소하고, IQR과 Isolation Forest로 탐지된
    이상치를 산점도로 시각적으로 비교하는 함수입니다.

    Args:
        X_data (pd.DataFrame): 독립 변수 데이터 (e.g., X_train).
        detect_iqr_func (function): IQR 이상치를 탐지하고 위치 인덱스를 반환하는 함수.
        detect_if_func (function): Isolation Forest 이상치를 탐지하고 위치 인덱스를 반환하는 함수.
        title (str): 그래프의 제목.
    """
    if X_data.empty:
        print("오류: X_data가 비어 있습니다.")
        return

    # 1. 데이터 전처리 및 PCA 수행
    
    # PCA는 스케일에 민감하므로 StandardScaler를 사용하여 데이터를 표준화합니다.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # PCA를 사용하여 2개의 주성분으로 축소
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    
    # 2. 두 가지 방법으로 이상치 탐지 및 마스킹
    
    # 탐지 함수 호출 (위치 인덱스를 반환)
    outliers_iqr_pos = detect_iqr_func(X_data)
    outliers_if_pos = detect_if_func(X_data)
    
    # 불리언 마스크 생성
    is_iqr_outlier = np.zeros(len(X_data), dtype=bool)
    is_if_outlier = np.zeros(len(X_data), dtype=bool)
    
    is_iqr_outlier[outliers_iqr_pos] = True
    is_if_outlier[outliers_if_pos] = True
    
    # 3. 데이터 포인트 분류
    # 0: 정상 (Normal), 1: IQR만 이상치, 2: IF만 이상치, 3: 둘 다 이상치
    outlier_type = np.zeros(len(X_data), dtype=int)
    
    outlier_type[(is_iqr_outlier) & (is_if_outlier)] = 3      # 3: 둘 다 이상치
    outlier_type[(is_iqr_outlier) & (~is_if_outlier)] = 1     # 1: IQR만 이상치
    outlier_type[(~is_iqr_outlier) & (is_if_outlier)] = 2     # 2: IF만 이상치
    
    X_pca_df['Outlier_Type'] = outlier_type
    
    # 4. 시각화 
    
    plt.figure(figsize=(10, 8))
    
    # 색상, 마커, 레이블 설정
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'black'}
    markers = {0: 'o', 1: 'X', 2: 's', 3: '*'}
    labels = {
        0: '정상 데이터 (Normal)',
        1: 'IQR만 이상치 (단변량)',
        2: 'IF만 이상치 (다변량)',
        3: '둘 다 이상치 (Both)'
    }

    # 데이터 분류별 산점도 그리기
    for type_val, group in X_pca_df.groupby('Outlier_Type'):
        plt.scatter(group['PC1'], group['PC2'], 
                    c=colors[type_val], 
                    marker=markers[type_val], 
                    label=labels[type_val],
                    s=type_val * 40 + 50, # 이상치일수록 마커 크기 키움
                    alpha=0.6)

    plt.title(title, fontsize=15)
    plt.xlabel(f'주성분 1 (PC1, 설명 분산: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'주성분 2 (PC2, 설명 분산: {pca.explained_variance_ratio_[1]:.2f})')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

def analyze_and_plot_interactions(df, target_col, min_corr=None, top_n=9):
    """
    모든 피처 쌍을 조합하여 사칙연산(+, -, *, /) 파생변수를 만들고,
    Target과의 상관관계가 높은 조합을 찾아 시각화합니다.
    
    Args:
        df (pd.DataFrame): 데이터프레임
        target_col (str or pd.Series): 타겟 컬럼
        min_corr (float): 선택할 최소 상관계수 절대값 (이 값 이상인 조합만 선택)
        top_n (int): 시각화할 최대 그래프 개수 (너무 많으면 느려지므로 제한, 기본값 9)
    """
    
    # 1. 안전장치: target_col이 Series면 이름 추출
    if isinstance(target_col, pd.Series):
        target_col = target_col.name

    # 2. 수치형 피처만 추출 (타겟 제외)
    features = df.drop(columns=[target_col], errors='ignore').select_dtypes(include='number').columns.tolist()
    
    # 타겟 데이터 확보
    y = df[target_col]
    
    print(f"🔍 총 {len(features)}개 피처로 {len(list(combinations(features, 2)))}개의 조합을 분석합니다...")
    
    # 3. 모든 조합에 대해 파생변수 생성 및 상관계수 테스트
    results = []
    
    for f1, f2 in combinations(features, 2):
        # 원본 데이터 가져오기
        d1, d2 = df[f1], df[f2]
        
        # 4가지 사칙연산 시도
        ops = {
            'plus': d1 + d2,
            'minus': d1 - d2,
            'multiply': d1 * d2,
            'divide': d1 / (d2.replace(0, np.nan)) # 0으로 나누기 방지
        }
        
        for op_name, new_val in ops.items():
            # 상관계수 계산 (절대값 기준)
            corr = new_val.corr(y)
            
            if not np.isnan(corr):
                results.append({
                    'f1': f1,
                    'f2': f2,
                    'operation': op_name,
                    'new_feature_name': f"{f1}_{op_name}_{f2}",
                    'corr_abs': abs(corr),
                    'corr_raw': corr
                })

    # 4. 상관계수 높은 순으로 정렬 및 중복 제거
    results_df = pd.DataFrame(results).sort_values(by='corr_abs', ascending=False)
    unique_combinations = results_df.drop_duplicates(subset=['f1', 'f2'])
    
    # [수정] min_corr 필터링 로직 적용
    if min_corr is not None:
        selected_combinations = unique_combinations[unique_combinations['corr_abs'] >= min_corr]
        print(f"🎯 상관계수(절대값) {min_corr} 이상인 조합 {len(selected_combinations)}개를 찾았습니다.")
    else:
        selected_combinations = unique_combinations
        
    if selected_combinations.empty:
        print("⚠️ 조건에 맞는 조합이 없습니다. min_corr 값을 낮춰보세요.")
        return selected_combinations

    # 시각화 개수 제한 (top_n)
    if top_n and len(selected_combinations) > top_n:
        print(f"ℹ️ 시각화는 가독성을 위해 상위 {top_n}개만 출력합니다. (전체 결과는 반환된 데이터프레임 확인)")
        plot_combinations = selected_combinations.head(top_n)
    else:
        plot_combinations = selected_combinations

    print("-" * 60)
    print(plot_combinations[['new_feature_name', 'corr_raw']].to_string(index=False))
    print("-" * 60)

    # 5. 시각화 (Subplots 생성)
    num_plots = len(plot_combinations)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    # axes가 1개일 경우 배열이 아닐 수 있으므로 처리
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() 
    
    sns.set_theme(style="whitegrid")

    for i, (_, row) in enumerate(plot_combinations.iterrows()):
        ax = axes[i]
        
        # Scatter Plot 그리기
        sns.scatterplot(
            data=df,
            x=row['f1'],
            y=row['f2'],
            hue=target_col,
            palette='RdYlBu_r',
            alpha=0.6,
            s=20,
            ax=ax
        )
        
        # 제목에 추천 파생변수와 상관계수 표시
        ax.set_title(f"Rank {i+1}: {row['f1']} vs {row['f2']}\n(Best: {row['operation']}, r={row['corr_raw']:.2f})", fontsize=11)
        ax.legend([],[], frameon=False) 

    # 남은 빈 서브플롯 끄기
    if num_plots > 1:
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    return selected_combinations # 필터링된 전체 결과 반환

# --- 실행 예시 ---
# analyze_and_plot_interactions(df_full, target_col='quality', min_corr=0.4, top_n=9)

def create_derived_features(df, combinations_info):
    """
    분석 결과(combinations_info)를 바탕으로 실제 파생변수를 데이터프레임에 생성합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임 (Train 또는 Test)
        combinations_info (pd.DataFrame): analyze_and_plot_interactions 함수의 리턴값
                                          (필요한 컬럼: 'f1', 'f2', 'operation', 'new_feature_name')
                                          
    Returns:
        pd.DataFrame: 파생변수가 추가된 새로운 데이터프레임
    """
    df_new = df.copy()
    
    if combinations_info.empty:
        print("⚠️ 생성할 파생변수 정보가 없습니다.")
        return df_new

    print(f"🛠️ 총 {len(combinations_info)}개의 파생변수를 생성합니다...")
    
    for _, row in combinations_info.iterrows():
        f1 = row['f1']
        f2 = row['f2']
        op = row['operation']
        col_name = row['new_feature_name']
        
        # 컬럼이 존재하는지 확인
        if f1 not in df_new.columns or f2 not in df_new.columns:
            print(f"⚠️ 경고: {f1} 또는 {f2} 컬럼이 데이터에 없어 '{col_name}' 생성을 건너뜁니다.")
            continue

        try:
            if op == 'plus':
                df_new[col_name] = df_new[f1] + df_new[f2]
            elif op == 'minus':
                df_new[col_name] = df_new[f1] - df_new[f2]
            elif op == 'multiply':
                df_new[col_name] = df_new[f1] * df_new[f2]
            elif op == 'divide':
                # 0으로 나누기 방지 (np.inf 방지)
                df_new[col_name] = df_new[f1] / (df_new[f2].replace(0, np.nan))
                # 나눗셈으로 생긴 NaN 값 처리 (선택 사항, 여기서는 0으로 채움)
                df_new[col_name] = df_new[col_name].fillna(0)
                
        except Exception as e:
            print(f"❌ 에러 발생 ({col_name}): {e}")

    print(f"✅ 생성 완료! (컬럼 수: {df.shape[1]} -> {df_new.shape[1]})")
    return df_new

# --- 실행 예시 ---
# 1. Train Set에서 좋은 파생변수 찾기 (최소 상관계수 0.3 이상인 것만)
# top_features_df = analyze_and_plot_interactions(X_train.join(y_train), target_col='quality', min_corr=0.3, top_n=9)

# 2. Train Set에 적용
# X_train_eng = create_derived_features(X_train, top_features_df)

# 3. Test Set에도 동일한 규칙(top_features_df) 적용 (Data Leakage 방지)
# X_test_eng = create_derived_features(X_test, top_features_df)

def classify_features_by_keywords(df, target_keywords):
    """
    df의 파생 변수와 target_keywords의 단일 변수 중, 
    구성 요소(f1, f2, key)가 서로 겹치지 않으면서 
    상관계수(절대값)가 가장 높은 변수들을 우선적으로 선택하여 반환합니다.

    Args:
        df (pd.DataFrame): 'f1', 'f2', 'new_feature_name', 'corr_abs' 컬럼을 포함한 데이터프레임
        target_keywords (dict): {'feature_name': correlation_value} 형태의 딕셔너리 
                                (예: {'alcohol': 0.4617, ...})
        
    Returns:
        dict: 선택된 변수명과 그 상관계수 값 {'feature_name': score}
    """
    
    # 1. 모든 후보군을 하나의 리스트로 통합
    candidates = []

    # 1-1. df에서 후보 추출 (f1 != f2 인 경우만)
    # iterrows를 사용하여 각 행의 정보를 딕셔너리로 변환
    filtered_df = df[df['f1'] != df['f2']]
    for _, row in filtered_df.iterrows():
        candidates.append({
            'name': row['new_feature_name'],
            'score': row['corr_abs'],
            'components': {row['f1'], row['f2']}, # 비교를 위해 set으로 저장
            'type': 'derived'
        })
        
    # 1-2. target_keywords에서 후보 추출
    for key, value in target_keywords.items():
        candidates.append({
            'name': key,
            'score': value,  # 이미 절대값 혹은 비교 가능한 수치라고 가정
            'components': {key}, # 단일 변수이므로 자기 자신이 구성요소
            'type': 'base'
        })
    
    # 2. 상관계수(score) 기준으로 내림차순 정렬 (높은 점수 우선)
    # 점수가 높은 순서대로 루프를 돌며 선택 여부를 결정합니다.
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 3. 중복 방지 선택 로직 (Greedy Algorithm)
    selected_features = {}
    used_components = set() # 이미 선택된 변수들의 구성요소를 저장하는 집합

    print(f"{'Rank':<5} | {'Score':<10} | {'Type':<10} | {'Feature Name'}")
    print("-" * 60)

    for i, cand in enumerate(candidates):
        # 현재 후보의 구성요소가 이미 사용된 구성요소와 겹치는지 확인
        # intersection() 결과가 있으면(=True) 겹치는 것
        if not cand['components'].intersection(used_components):
            
            # 겹치지 않으면 선택!
            selected_features[cand['name']] = cand['score']
            
            # 사용된 구성요소 목록에 추가 (Lock)
            used_components.update(cand['components'])
            
            print(f"{i+1:<5} | {cand['score']:.6f}   | {cand['type']:<10} | {cand['name']} (Selected)")
        else:
            # 겹치면 패스 (로그 확인용)
            # print(f"{i+1:<5} | {cand['score']:.6f} | {cand['type']:<10} | {cand['name']} (Skipped - Overlap)")
            pass

    return selected_features

target = 'quality'
X_eda = df_full.drop(target, axis=1)
y_eda = df_full[target]

plot_target_distribution(y_eda, 
                         title="와인 품질 분포 (full Set)", 
                         xlabel="Quality Score")

plot_numerical_distribution(X_eda)

selected_corr_features = visualize_and_classify_correlation(X_eda, y_eda, 
                                                       target_name='quality', 
                                                       high_threshold=0.2, 
                                                       medium_threshold=0.1)

print(f"상관관계가 높은 피처:{len(selected_corr_features)}개 {selected_corr_features}")

selected_vif_features, selected_vif_not_features = run_vif_analysis(X_eda)

plot_feature_quality_boxplots(X_eda, y_eda)

# 1. 중복 데이터가 얼마나 있는지 확인
duplicate_count = df_full.duplicated().sum()
print(f"제거 전 데이터 개수: {len(df_full)}")
print(f"발견된 중복 데이터 개수: {duplicate_count}")

# 2. 중복 데이터 제거 (keep='first'는 첫 번째 것만 남기고 뒤에 나오는 중복은 지운다는 뜻)
if duplicate_count > 0:
    df_drop_duplicates = df_full.drop_duplicates(keep='first')
    
    # 인덱스가 꼬이지 않게 재정렬 (중요!)
    df_drop_duplicates = df_drop_duplicates.reset_index(drop=True)
    print("중복 제거 완료!")
else:
    print("중복 데이터가 없습니다.")

target = 'quality'
X_drop_duplicates = df_drop_duplicates.drop(target, axis=1)
y_drop_duplicates = df_drop_duplicates[target]
print(f"제거 후 데이터 개수: {len(df_drop_duplicates)}")

# X: 피처 데이터, y: 타겟(와인 품질)
X_train, X_test, y_train, y_test = train_test_split(
    X_drop_duplicates, 
    y_drop_duplicates, 
    test_size=0.2,       # 테스트 데이터 비율 (20%)
    random_state=SEED,   # 결과를 똑같이 재현하기 위한 시드값
    stratify=y_drop_duplicates       # <--- [핵심] y(품질)의 비율을 유지하며 나눠라!
)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)
print(f"훈련 데이터의 X의 크기: {X_train.shape} 훈련 데이터의 y의 크기: {y_train.shape}")
print(f"테스트 데이터의 X의 크기: {X_test.shape} 테스트 데이터의 y의 크기: {y_test.shape}")
# 원본 데이터의 비율
print("전체 비율:\n", y_eda.value_counts(normalize=True))
plot_target_distribution(y_eda, 
                         title="와인 품질 분포 (full Set)", 
                         xlabel="Quality Score")
# 학습 데이터의 비율
print("Train 비율:\n", y_train.value_counts(normalize=True))
plot_target_distribution(y_train, 
                         title="와인 품질 분포 (train Set)", 
                         xlabel="Quality Score")
# 테스트 데이터의 비율
print("Test 비율:\n", y_test.value_counts(normalize=True))
plot_target_distribution(y_test, 
                         title="와인 품질 분포 (test Set)", 
                         xlabel="Quality Score")

# 실험 진행
outliers_iqr = detect_outliers_iqr(X_train)
outliers_if = detect_outliers_if(X_train)

print(f"\n총 데이터 개수: {len(X_train)}")
print(f"IQR 방식 (2개 이상 변수 이상치) 제거 대상 개수: {len(outliers_iqr)}")
print(f"Isolation Forest(IF) 방식 제거 대상 개수: {len(outliers_if)}")

# 최종 결정: Isolation Forest 적용 (다변량 관계 고려)
# 1. 마스크 생성 (모든 행을 True로 초기화)
# 단, Train 데이터에서만 제거해야 함!
mask = np.ones(len(X_train), dtype=bool)
# 2. 함수 호출 및 위치 인덱스를 사용하여 마스크 업데이트 (False로 설정)
mask[detect_outliers_iqr(X_train)] = False 

# 3. 이상치 제거된 데이터셋 생성
X_train_clean_iqr = X_train[mask]
y_train_clean_iqr = y_train[mask]

mask = np.ones(len(X_train), dtype=bool)
mask[detect_outliers_if(X_train)] = False # Isolation Forest 결과를 적용
X_train_clean_if = X_train[mask]
y_train_clean_if = y_train[mask]

# 결과 출력
print(f"IQR: 이상치 제거 후 Train Set 크기: {X_train_clean_iqr.shape} (삭제된 행: {len(X_train) - len(X_train_clean_iqr)})({(len(X_train) - len(X_train_clean_iqr)) / len(X_train) * 100:.2f}%)")
print(f"IQR: 제거 후 데이터 개수 (X_train_clean): {len(X_train_clean_iqr)}  제거 후 데이터 개수 (y_train_clean): {len(y_train_clean_iqr)}")

print(f"IF: 이상치 제거 후 Train Set 크기: {X_train_clean_if.shape} (삭제된 행: {len(X_train) - len(X_train_clean_if)})({(len(X_train) - len(X_train_clean_if)) / len(X_train) * 100:.2f}%)")
print(f"IF: 제거 후 데이터 개수 (X_train_clean): {len(X_train_clean_if)}  제거 후 데이터 개수 (y_train_clean): {len(y_train_clean_if)}")

compare_outliers_scatter(X_train, detect_outliers_iqr, detect_outliers_if)

clf = IsolationForest(max_samples='auto', random_state=SEED, contamination=0.05)
preds = clf.fit_predict(X_train)
mask = preds != -1
X_train_clean = X_train[mask]
y_train_clean = y_train[mask]
# 두 데이터를 리스트로 묶고, axis=1 (열 방향)로 병합
df_train_clean = pd.concat([X_train_clean, y_train_clean], axis=1)
# 결과 확인
print(f"이상치 제거 전(X_train): {len(X_train)} -> 제거 후(X_train_clean): {len(X_train_clean)}(약 {len(X_train) - len(X_train_clean)}({(len(X_train) - len(X_train_clean)) / len(X_train) * 100:.2f}%)개의 이상치 제거)")
print(f"이상치 제거 전(y_train): {len(y_train)} -> 제거 후(y_train_clean): {len(y_train_clean)}(약 {len(y_train) - len(y_train_clean)}({(len(y_train) - len(y_train_clean)) / len(y_train) * 100:.2f}%)개의 이상치 제거)")
print(f"이상치 제거 후 데이터프레임: {df_train_clean.head()}")

# ============================================
# 이상치 제거 전후 데이터 분포 비교
# ============================================
print("=" * 60)
print("이상치 제거 전후 데이터 분포 비교")
print("=" * 60)

# 1. 히스토그램 비교 (Histogram Overlay)
print("\n[1] 히스토그램 비교 (전/후 오버레이)")
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(X_train.columns):
    ax = axes[idx]
    # 전: X_train (이상치 제거 전)
    ax.hist(X_train[col], bins=30, alpha=0.5, label='Before', color='red', edgecolor='darkred')
    # 후: X_train_clean (이상치 제거 후)
    ax.hist(X_train_clean[col], bins=30, alpha=0.5, label='After', color='blue', edgecolor='darkblue')
    ax.set_title(f'{col}')
    ax.set_xlabel('값')
    ax.set_ylabel('빈도')
    ax.legend(loc='upper right', fontsize=8)

# 빈 subplot 제거
if len(X_train.columns) < len(axes):
    for idx in range(len(X_train.columns), len(axes)):
        fig.delaxes(axes[idx])

plt.tight_layout()
plt.suptitle('이상치 제거 전후 히스토그램 비교 (빨강: 전, 파랑: 후)', y=1.02, fontsize=14)
plt.show()

# 2. 박스플롯 비교 (Before/After Box Plot)
print("\n[2] 박스플롯 비교 (전/후)")
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(X_train.columns):
    ax = axes[idx]
    # 전/후 데이터 준비
    data_before = X_train[col].values
    data_after = X_train_clean[col].values
    
    # 박스플롯 생성
    bp = ax.boxplot([data_before, data_after], labels=['Before', 'After'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax.set_title(f'{col}')
    ax.set_ylabel('값')

# 빈 subplot 제거
if len(X_train.columns) < len(axes):
    for idx in range(len(X_train.columns), len(axes)):
        fig.delaxes(axes[idx])

plt.tight_layout()
plt.suptitle('이상치 제거 전후 박스플롯 비교 (빨강: 전, 파랑: 후)', y=1.02, fontsize=14)
plt.show()

# 3. 통계 요약 테이블 (Statistical Summary)
print("\n[3] 통계 요약 비교")
print(f"\n데이터 건수 변화: {len(X_train)} → {len(X_train_clean)} (제거: {len(X_train) - len(X_train_clean)}개, {(len(X_train) - len(X_train_clean))/len(X_train)*100:.2f}%)")

# 전후 통계량 비교 테이블
stats_comparison = []
for col in X_train.columns:
    before_stats = X_train[col].describe()
    after_stats = X_train_clean[col].describe()
    
    stats_comparison.append({
        'Feature': col,
        'Before_Mean': before_stats['mean'],
        'After_Mean': after_stats['mean'],
        'Mean_Diff': after_stats['mean'] - before_stats['mean'],
        'Before_Std': before_stats['std'],
        'After_Std': after_stats['std'],
        'Std_Diff': after_stats['std'] - before_stats['std'],
        'Before_Min': before_stats['min'],
        'After_Min': after_stats['min'],
        'Before_Max': before_stats['max'],
        'After_Max': after_stats['max']
    })

stats_df = pd.DataFrame(stats_comparison)
print("\n=== 이상치 제거 전후 통계량 비교 ===")
print(stats_df[['Feature', 'Before_Mean', 'After_Mean', 'Mean_Diff', 'Before_Std', 'After_Std', 'Std_Diff']].round(4).to_string(index=False))

print("\n=== 이상치 제거 전후 범위(Min/Max) 비교 ===")
print(stats_df[['Feature', 'Before_Min', 'After_Min', 'Before_Max', 'After_Max']].round(4).to_string(index=False))

# 4. 분포 변화 요약 시각화
print("\n[4] 평균/표준편차 변화 시각화")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 평균 변화
ax1 = axes[0]
x = np.arange(len(stats_df))
width = 0.35
bars1 = ax1.bar(x - width/2, stats_df['Before_Mean'], width, label='Before', color='lightcoral')
bars2 = ax1.bar(x + width/2, stats_df['After_Mean'], width, label='After', color='lightblue')
ax1.set_xticks(x)
ax1.set_xticklabels(stats_df['Feature'], rotation=45, ha='right')
ax1.set_ylabel('Mean')
ax1.set_title('피처별 평균 변화')
ax1.legend()

# 표준편차 변화
ax2 = axes[1]
bars1 = ax2.bar(x - width/2, stats_df['Before_Std'], width, label='Before', color='lightcoral')
bars2 = ax2.bar(x + width/2, stats_df['After_Std'], width, label='After', color='lightblue')
ax2.set_xticks(x)
ax2.set_xticklabels(stats_df['Feature'], rotation=45, ha='right')
ax2.set_ylabel('Std')
ax2.set_title('피처별 표준편차 변화')
ax2.legend()

plt.tight_layout()
plt.show()

print("\n이상치 제거로 인해 극단값이 제거되어 분포가 더 집중되었습니다.")

# [Modified] Legacy FE planning removed to use AutoOptimizer
# top_features_df = ...


# [Removed] Legacy feature selection factories.
 

import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

def apply_smotetomek(X_train, y_train, random_state=42):
    """
    SMOTETomek를 사용하여 데이터 불균형을 해소하는 함수
    
    Args:
        X_train (pd.DataFrame): 훈련 피처 데이터
        y_train (pd.Series): 훈련 타겟 데이터
        random_state (int): 랜덤 시드 값
        
    Returns:
        X_resampled, y_resampled, df_resampled (tuple): 
        샘플링된 X, y 그리고 병합된 DataFrame
    """
    print("=" * 80)
    print("=== SMOTETomek 적용 (Train 내에서만) ===")
    print("=" * 80)

    # 소수 클래스 최소 샘플 수 확인
    min_samples = y_train.value_counts().min()
    print(f"소수 클래스 최소 샘플 수: {min_samples}개")

    # k_neighbors 설정 (최소 샘플 수가 적을 경우 동적으로 조정)
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
    print(f"SMOTE k_neighbors: {k_neighbors}")

    try:
        # SMOTETomek 설정 및 적용
        smotetomek = SMOTETomek(
            smote=SMOTE(k_neighbors=k_neighbors, random_state=random_state),
            random_state=random_state
        )
        X_resampled, y_resampled = smotetomek.fit_resample(X_train, y_train)

        # 결과 출력
        print(f"\n샘플링 전: {len(X_train)}개")
        print(f"샘플링 후: {len(X_resampled)}개")
        print(f"증가율: {(len(X_resampled) / len(X_train) - 1) * 100:.1f}%")
        
        print(f"\n샘플링 후 Quality 분포:")
        print(y_resampled.value_counts().sort_index())
        
        success = True

    except Exception as e:
        print(f"SMOTETomek 실패 (원본 데이터 유지): {e}")
        # 실패 시 원본 유지 (데이터 타입 유지를 위해 copy 사용 권장)
        X_resampled = X_train.copy()
        y_resampled = y_train.copy()
        success = False

    # 두 데이터를 병합하여 하나의 DataFrame으로 생성 (시각화나 확인용)
    # SMOTETomek 결과가 numpy array일 경우를 대비해 DataFrame으로 변환 시도
    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    if not isinstance(y_resampled, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y_train.name)

    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    return X_resampled, y_resampled, df_resampled


    # ==========================================
# 실행부 (함수 호출)
# ==========================================

# 가정: X_train_selected, y_train_clean, SEED, X_test 변수가 이미 정의되어 있다고 가정

# 1. 함수 실행
X_train_balanced, y_train_balanced, df_train_balanced = apply_smotetomek(
    X_train_clean, 
    y_train_clean, 
    random_state=SEED
)

# 2. Test 데이터 관련 로그 (함수 밖에서 처리)
print(f"\n⚠️ Test 데이터는 샘플링하지 않음: {len(X_test)}개 유지")

# 3. 최종 결과 확인
# 두 데이터를 리스트로 묶고, axis=1 (열 방향)로 병합
df_train_balanced = pd.concat([X_train_balanced, y_train_balanced], axis=1)
# 결과 확인
print(f"SMOTETomek 전(X_train_selected): {len(X_train_clean)} -> SMOTETomek 후(X_train_balanced): {len(X_train_balanced)}")
print(f"SMOTETomek 전(y_train_clean): {len(y_train_clean)} -> SMOTETomek 후(y_train_balanced): {len(y_train_balanced)}")
print(f"SMOTETomek 후 데이터프레임: {df_train_balanced.head()}")


# 추가 라이브러리 임포트
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel

# 현재 클래스 분포 시각화
print("=" * 50)
print("현재 클래스 분포")
print("=" * 50)

class_dist = pd.Series(y_train_balanced).value_counts().sort_index()
print(class_dist)

# 클래스 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 막대 그래프
colors = plt.cm.viridis(np.linspace(0, 1, len(class_dist)))
axes[0].bar(class_dist.index.astype(str), class_dist.values, color=colors)
axes[0].set_xlabel('Quality')
axes[0].set_ylabel('Count')
axes[0].set_title('클래스별 샘플 수 (이상치 제거 후)')
for i, (idx, val) in enumerate(zip(class_dist.index, class_dist.values)):
    axes[0].text(i, val + 10, str(val), ha='center', fontsize=10)

# 파이 차트
axes[1].pie(class_dist.values, labels=[f'Quality {i}' for i in class_dist.index], 
            autopct='%1.1f%%', colors=colors)
axes[1].set_title('클래스 비율')

plt.tight_layout()
plt.show()

# 불균형 비율 계산
majority_class = class_dist.max()
minority_class = class_dist.min()
imbalance_ratio = majority_class / minority_class
print(f"\n불균형 비율 (다수/소수): {imbalance_ratio:.2f}")
print(f"가장 적은 클래스: Quality {class_dist.idxmin()} ({minority_class}개)")
print(f"가장 많은 클래스: Quality {class_dist.idxmax()} ({majority_class}개)")


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor  # KNN 사용
from sklearn.linear_model import Ridge
import pandas as pd

def select_best_scaler(data, target_col='quality', model_type='knn'):
    # 데이터 분리
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'Robust': RobustScaler(),
        'MaxAbs': MaxAbsScaler()
    }
    
    best_score = float('inf')
    best_scaler_name = ''
    
    # 모델 설정
    if model_type == 'knn':
        # SMOTE된 데이터의 밀도를 잘 파악하기 위해 이웃 수(n_neighbors) 적절히 설정
        model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        print(f"=== 스케일러 성능 비교 (Model: KNN, RMSE 낮을수록 좋음) ===")
    else:
        model = Ridge(random_state=42)
        print(f"=== 스케일러 성능 비교 (Model: Ridge, RMSE 낮을수록 좋음) ===")
    
    results = {}
    
    for name, scaler in scalers.items():
        try:
            X_scaled = scaler.fit_transform(X)
            
            # 3-Fold CV로 RMSE 측정
            # 데이터가 섞여있지 않을 수 있으므로 cv 객체를 따로 주는 것도 좋으나, 여기선 간단히 3
            scores = cross_val_score(model, X_scaled, y, scoring='neg_root_mean_squared_error', cv=3)
            rmse = -scores.mean()
            
            results[name] = rmse
            print(f"{name} Scaler RMSE: {rmse:.4f}")
            
            if rmse < best_score:
                best_score = rmse
                best_scaler_name = name
        except Exception as e:
            print(f"{name} Scaler 실패: {e}")
            
    print(f"\n>>> 선정된 스케일러: {best_scaler_name}")
    return scalers[best_scaler_name]

# ==========================================
# 실행 (샘플링된 df_train_balanced 사용)
# ==========================================
# [Modified] Legacy FE application removed.
# Using AutoOptimizer with X_train_balanced directly.


# Set MLflow experiment
experiment_name = "Wine_Quality_Prediction_20251128_2"
try:
    mlflow.create_experiment(experiment_name)
except:
    pass
mlflow.set_experiment(experiment_name)

print(f"MLflow experiment set to: {experiment_name}")

# ==========================================
# Advanced Feature Engineering & Optimization Logic
# ==========================================

def advanced_feature_engineering(df):
    """
    Generates a wide variety of derived features (Ratios, Logs).
    """
    df = df.copy()
    
    # 1. Ratio Features based on Domain Knowledge
    if 'residual sugar' in df.columns and 'fixed acidity' in df.columns:
        df['sugar_acid_ratio'] = df['residual sugar'] / (df['fixed acidity'] + 1e-6)
    
    if 'free sulfur dioxide' in df.columns and 'total sulfur dioxide' in df.columns:
        df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)
        
    if 'alcohol' in df.columns and 'density' in df.columns:
        df['alcohol_density_ratio'] = df['alcohol'] / (df['density'] + 1e-6)
        
    if 'pH' in df.columns and 'sulphates' in df.columns:
        df['ph_sulphate_ratio'] = df['pH'] / (df['sulphates'] + 1e-6)
    
    if 'citric acid' in df.columns and 'volatile acidity' in df.columns:
         df['acid_ratio_2'] = df['citric acid'] / (df['volatile acidity'] + 1e-6)

    # 2. Log Transformations for Skewed Features
    skewed_candidates = ['residual sugar', 'chlorides', 'sulphates', 'volatile acidity', 'citric acid']
    for col in skewed_candidates:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            
    return df

class AutoOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, models_params, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models_params = models_params
        self.random_state = random_state
        self.best_model = None
        self.best_features = None
        self.best_score = -np.inf
        
    def optimize(self):
        # 1. Feature Expansion
        print("Starting Feature Expansion...")
        X_train_expanded = advanced_feature_engineering(self.X_train)
        X_test_expanded = advanced_feature_engineering(self.X_test)
        
        # Ensure columns match
        X_test_expanded = X_test_expanded[X_train_expanded.columns]
        
        print(f"Features expanded from {self.X_train.shape[1]} to {X_train_expanded.shape[1]}")
        
        # 2. Feature Selection Loop
        # Rank features using a generic Random Forest
        print("Ranking features...")
        # Check if classification or regression for ranking
        # We assume dataset is classification (quality classes)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Simple check: number of unique target values
        if len(np.unique(self.y_train)) < 20:
            selector = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        else:
            selector = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            
        selector.fit(X_train_expanded, self.y_train)
        
        importances = pd.Series(selector.feature_importances_, index=X_train_expanded.columns)
        sorted_features = importances.sort_values(ascending=False).index.tolist()
        print("Top 10 Important Features:", sorted_features[:10])
        
        # Try different subsets
        feature_subsets = [10, 20, 30, 9999] # 9999 means all
        feature_subsets = sorted(list(set([min(n, len(sorted_features)) for n in feature_subsets])))
        
        for n_features in feature_subsets:
            selected_cols = sorted_features[:n_features]
            print(f"\nEvaluating Top {n_features} features...")
            
            X_tr_sub = X_train_expanded[selected_cols]
            
            # Run Model Search
            self._run_model_search(X_tr_sub, self.y_train, selected_cols)
            
        print(f"\nOptimization Complete. Best CV Score: {self.best_score:.4f}")
        return self.best_model, self.best_features, X_test_expanded
    
    def _run_model_search(self, X, y, feature_names):
        # Determine model type for scoring
        from sklearn.base import is_classifier, is_regressor
        
        for model_name, mp in self.models_params.items():
            run_name = f"{model_name}_{len(feature_names)}_features"
            
            with mlflow.start_run(run_name=run_name, nested=True):
                # Check regression or classification from the model object
                is_regr = 'Regressor' in model_name or is_regressor(mp['model'])
                step_name = 'regressor' if is_regr else 'classifier'
                scoring = qwk_scorer_regressor_sklearn if is_regr else qwk_scorer
                
                # Pipeline construction
                # We add PolynomialFeatures here to create interactions among the SELECTED top features
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
                    (step_name, mp['model'])
                ]
                
                pipe = Pipeline(steps)
                
                # Use RandomizedSearch
                search = RandomizedSearchCV(
                    pipe, mp['params'], n_iter=10, cv=5, 
                    scoring=scoring, n_jobs=-1, random_state=self.random_state,
                    verbose=0
                )
                
                try:
                    search.fit(X, y)
                    score = search.best_score_
                    
                    # Log artifacts
                    mlflow.log_params(search.best_params_)
                    mlflow.log_metric("qwk_cv_score", score)
                    mlflow.log_param("n_features", len(feature_names))
                    mlflow.log_param("features", feature_names)
                    
                    print(f"  {model_name}: {score:.4f}")
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = search.best_estimator_
                        self.best_features = feature_names
                        # Save model
                        mlflow.sklearn.log_model(search.best_estimator_, "model", input_example=X.head(1))
                        
                except Exception as e:
                    print(f"  Failed {model_name}: {e}")





models_params = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=SEED, max_iter=1000),
        'params': {
            'classifier__C': [0.1, 1, 10, 100],  # 범위 확대
            'classifier__solver': ['lbfgs', 'liblinear']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=SEED),
        'params': {
            'classifier__n_estimators': [100, 200, 300, 500],  # 범위 확대
            'classifier__max_depth': [None, 10, 20, 30],  # 범위 확대
            'classifier__min_samples_split': [2, 5, 10]  # 범위 확대
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='mlogloss'),
        'params': {
            'classifier__n_estimators': [100, 200, 300, 500],  # 범위 확대
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],  # 범위 확대
            'classifier__max_depth': [3, 5, 7, 9]  # 범위 확대
        }
    },
    'CatBoostClassifier': {
        'model': CatBoostClassifier(random_state=SEED, verbose=False),
        'params': {
            'classifier__iterations': [100, 200, 300, 500],  # 범위 확대
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],  # 범위 확대
            'classifier__depth': [4, 6, 8, 10]  # 범위 확대
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=SEED, verbose=-1),
        'params': {
            'classifier__n_estimators': [100, 200, 300, 500],  # 범위 확대
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],  # 범위 확대
            'classifier__max_depth': [4, 6, 8, 10]  # 범위 확대
        }
    },
    'SVC': {
        'model': SVC(random_state=SEED),
        'params': {
            'classifier__C': [0.1, 1, 10, 100],  # 범위 확대
            'classifier__kernel': ['linear', 'rbf']
        }
    },
    'XGBRegressor': {
        'model': XGBRegressor(random_state=SEED, use_label_encoder=False, eval_metric='mlogloss'),
        'params': {
            'regressor__n_estimators': [100, 200, 300, 500],  # 범위 확대
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],  # 범위 확대
            'regressor__max_depth': [3, 5, 7, 9]  # 범위 확대
        }
    },
    'LGBMRegressor': {
        'model': LGBMRegressor(random_state=SEED, verbose=-1),
        'params': {
            'regressor__n_estimators': [100, 200, 300, 500],  # 범위 확대
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],  # 범위 확대
            'regressor__max_depth': [4, 6, 8, 10]  # 범위 확대
        }
    }
}



# Metric: Quadratic Weighted Kappa
# 분류 모델용 QWK 스코어 (정수형 레이블 직접 사용)
qwk_scorer = make_scorer(cohen_kappa_score, weights='quadratic')

# 회귀 모델용 QWK 스코어 (예측값을 반올림하여 정수로 변환)
def qwk_scorer_regressor(y_true, y_pred):
    """
    회귀 모델의 연속적인 예측값을 반올림하여 정수로 변환한 후 QWK 계산
    """
    import numpy as np
    # 예측값을 반올림하여 정수로 변환 (quality는 정수형 등급)
    y_pred_rounded = np.round(y_pred).astype(int)
    # 범위를 quality 값의 범위로 제한 (일반적으로 3-9)
    y_pred_rounded = np.clip(y_pred_rounded, y_true.min(), y_true.max())
    return cohen_kappa_score(y_true, y_pred_rounded, weights='quadratic')

qwk_scorer_regressor_sklearn = make_scorer(qwk_scorer_regressor)



# ==========================================
# Main Execution with AutoOptimizer
# ==========================================

# 1. Prepare Data
# Need to encode target if classification
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train_balanced)

# 2. Run Optimization
print("\n=== Starting Automated Optimization Loop ===")
optimizer = AutoOptimizer(
    X_train_balanced, y_train_encoded,
    df_test.drop('quality', axis=1), y_test, # We pass test set here for transformation convenience, though AutoOptimizer splits internally if needed or just transforms
    models_params,
    random_state=SEED
)

best_model, best_features, X_test_expanded_full = optimizer.optimize()

# 3. Final Prediction
print("\n=== Final Evaluation on Test Set ===")
if best_model is None:
    print("No best model found.")
else:
    # Prepare Test Data (Subset to best features)
    X_test_final = X_test_expanded_full[best_features]
    
    # Predict
    y_pred = best_model.predict(X_test_final)
    
    # Inverse transform target if using encoder, but for QWK calculation we usually need int.
    # If regressor -> round. If classifier -> direct.
    
    from sklearn.base import is_regressor
    is_regr = is_regressor(best_model)
    
    if is_regr:
         # For regressor, pipeline output is continuous
         # We assume the model inside pipeline is the last step
         y_pred_rounded = np.round(y_pred).astype(int)
         y_pred_rounded = np.clip(y_pred_rounded, 0, 10) # Clip to reasonable wine scores
         y_pred_final = y_pred_rounded
    else:
         # For classifier, it returns encoded labels (0,1,2...) or original if not encoded?
         # We encoded y_train with le_target. So y_pred is encoded indices.
         # We need to transform back to original quality score (e.g. 3,4,5,6...)?
         # Wait, y_test is original.
         # So we should inverse_transform.
         if hasattr(le_target, 'inverse_transform'):
             y_pred_final = le_target.inverse_transform(y_pred)
         else:
             y_pred_final = y_pred

    # Calculate QWK
    final_score = cohen_kappa_score(y_test, y_pred_final, weights='quadratic')
    print(f"Final Test QWK Score: {final_score:.4f}")
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_final))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Test Set)\nBest Model ({len(best_features)} features)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()


