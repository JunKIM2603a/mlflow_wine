import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import cohen_kappa_score, make_scorer, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.base import is_regressor, is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed
SEED = 42
np.random.seed(SEED)

# ==========================================
# 1. Helper Functions
# ==========================================

def qwk_scorer_regressor(y_true, y_pred):
    y_pred_rounded = np.round(y_pred).astype(int)
    y_pred_rounded = np.clip(y_pred_rounded, y_true.min(), y_true.max())
    return cohen_kappa_score(y_true, y_pred_rounded, weights='quadratic')

qwk_scorer_regressor_sklearn = make_scorer(qwk_scorer_regressor)
qwk_scorer = make_scorer(cohen_kappa_score, weights='quadratic')

# --- Outlier Detection Methods ---
def detect_outliers_iqr(df):
    outlier_indices = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if col != 'quality':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_indices.extend(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index)
    return list(set(outlier_indices))

def detect_outliers_isolation_forest(df):
    X = df.drop('quality', axis=1)
    X = X.fillna(X.median())
    clf = IsolationForest(random_state=SEED, contamination=0.05)
    preds = clf.fit_predict(X)
    return df.index[preds == -1].tolist()

# --- Feature Engineering ---
def generate_advanced_features(df, target_column, top_features=None):
    """
    Generates Log, Polynomial, and PCA features.
    Returns: (df_eng, top_features_used)
    """
    df_eng = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols: numeric_cols.remove(target_column)
    
    # 1. Log Transformation
    for col in numeric_cols:
        if df_eng[col].min() >= 0:
            df_eng[f"log_{col}"] = np.log1p(df_eng[col])
            
    # 2. Interaction Features (Arithmetic)
    if top_features is None:
        if target_column in df.columns:
            corrs = df[numeric_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
            top_n_for_poly = 5 
            top_features = corrs.head(top_n_for_poly).index.tolist()
        else:
            # Fallback if no target and no top_features provided (shouldn't happen in correct flow)
            top_features = numeric_cols[:5]
    
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            f1 = top_features[i]
            f2 = top_features[j]
            if f1 in df_eng.columns and f2 in df_eng.columns:
                df_eng[f"{f1}_mul_{f2}"] = df_eng[f1] * df_eng[f2]
                df_eng[f"{f1}_div_{f2}"] = df_eng[f1] / (df_eng[f2] + 1e-6)

    # 3. Polynomial Features (Degree 2)
    # Ensure we only use available columns
    valid_top_features = [f for f in top_features if f in df.columns]
    if valid_top_features:
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(df[valid_top_features].fillna(df[valid_top_features].median()))
        feature_names = poly.get_feature_names_out(valid_top_features)
        
        for i, name in enumerate(feature_names):
            if name not in df_eng.columns and name not in valid_top_features:
                clean_name = f"poly_{name.replace(' ', '_').replace('^', '_')}"
                df_eng[clean_name] = X_poly[:, i]

    # 4. PCA Features
    # Standardize first
    X_pca_input = df[numeric_cols].fillna(df[numeric_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca_input)
    
    pca = PCA(n_components=5, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    for i in range(5):
        df_eng[f"pca_{i+1}"] = X_pca[:, i]
        
    return df_eng, top_features

# --- Feature Selection ---
def select_features_correlation(df, target_col, threshold=0.1):
    corrs = df.corr()[target_col].abs()
    selected = corrs[corrs >= threshold].index.tolist()
    if target_col in selected:
        selected.remove(target_col)
    return selected

def select_features_vif(df, target_col, vif_threshold=10, max_features_for_vif=50):
    X = df.drop(target_col, axis=1)
    X = X.fillna(X.median())
    
    # Optimization: Pre-filter by correlation
    if X.shape[1] > max_features_for_vif:
        corrs = df.corr()[target_col].abs()
        top_features = corrs.sort_values(ascending=False).head(max_features_for_vif + 1).index.tolist()
        if target_col in top_features: top_features.remove(target_col)
        X = X[top_features]

    while True:
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.shape[1] == 0: break
        
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_numeric.columns
        vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
        
        max_vif = vif_data['VIF'].max()
        if max_vif > vif_threshold:
            feature_to_remove = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
            X = X.drop(feature_to_remove, axis=1)
        else:
            break
    return X.columns.tolist()

def select_features_pareto(df, target_col, top_n=20):
    """
    Selects features based on Pareto Optimality between Relevance (Correlation) and Redundancy (Multicollinearity).
    """
    features = [col for col in df.columns if col != target_col]
    X = df[features].fillna(df[features].median())
    y = df[target_col]
    
    # 1. Calculate Relevance (R): Absolute Correlation with Target
    relevance = df[features].corrwith(y).abs()
    
    # 2. Calculate Redundancy (D): Average Absolute Correlation with other features
    # This can be computationally expensive for many features.
    # We'll use a correlation matrix.
    corr_matrix = X.corr().abs()
    redundancy = (corr_matrix.sum(axis=1) - 1) / (len(features) - 1) # Avg corr with others
    
    # 3. Pareto Selection
    # We want to Maximize R and Minimize D.
    # Let's create a DataFrame
    metrics = pd.DataFrame({'R': relevance, 'D': redundancy})
    
    # Simple approach: Non-dominated sorting is ideal, but for feature selection, 
    # a scoring metric combining both is often practical and robust.
    # Score = R / (1 + D) -> Maximize this.
    metrics['Score'] = metrics['R'] / (1 + metrics['D'])
    
    selected = metrics.sort_values('Score', ascending=False).head(top_n).index.tolist()
    return selected

# ==========================================
# 2. Main Execution Flow
# ==========================================

if __name__ == "__main__":
    print("=== 1. Data Loading & Splitting ===")
    try:
        df_raw = pd.read_csv('wine_data_homework.csv')
    except FileNotFoundError:
        print("Error: 'wine_data_homework.csv' not found.")
        exit(1)

    if 'Unnamed: 0' in df_raw.columns:
        df_raw = df_raw.drop('Unnamed: 0', axis=1)
    elif df_raw.columns[0] == 'index':
         df_raw = df_raw.drop(df_raw.columns[0], axis=1)

    X = df_raw.drop('quality', axis=1)
    y = df_raw['quality']
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
    )
    
    df_train = X_train_raw.copy()
    df_train['quality'] = y_train
    
    print(f"Train Shape: {df_train.shape}")

    print("\n=== 2. Dynamic Preprocessing on Train Data ===")
    
    # --- A. Dynamic Outlier Detection ---
    print("\n[Comparison] Outlier Detection: IQR vs Isolation Forest")
    outlier_methods = {
        'IQR': detect_outliers_iqr,
        'IsolationForest': detect_outliers_isolation_forest
    }
    
    best_outlier_method = None
    best_outlier_score = -1
    
    for name, func in outlier_methods.items():
        indices = func(df_train)
        df_clean_temp = df_train.drop(indices)
        X_temp = df_clean_temp.drop('quality', axis=1)
        y_temp = df_clean_temp['quality']
        
        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTETomek(random_state=SEED, smote=SMOTE(k_neighbors=1))),
            ('model', RandomForestClassifier(random_state=SEED, n_jobs=-1))
        ])
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = cross_val_score(pipeline, X_temp, y_temp, cv=cv, scoring=qwk_scorer, n_jobs=-1)
        mean_score = scores.mean()
        print(f"  Method: {name}, CV Score: {mean_score:.4f}")
        
        if mean_score > best_outlier_score:
            best_outlier_score = mean_score
            best_outlier_method = name

    print(f"Winner: {best_outlier_method} (Score: {best_outlier_score:.4f})")
    
    winner_func = outlier_methods[best_outlier_method]
    train_outlier_indices = winner_func(df_train)
    df_train_clean = df_train.drop(train_outlier_indices).reset_index(drop=True)
    
    # --- B. Advanced Feature Engineering ---
    print("\nGenerating Advanced Features (Log, Poly, PCA)...")
    df_train_derived, top_features_used = generate_advanced_features(df_train_clean, 'quality')
    df_train_derived = df_train_derived.replace([np.inf, -np.inf], np.nan).fillna(df_train_derived.median())
    print(f"Total Features after Engineering: {df_train_derived.shape[1] - 1}") # -1 for target
    
    # --- C. Dynamic Feature Selection ---
    print("\n[Comparison] Feature Selection: Correlation vs VIF vs Pareto")
    
    fs_methods = {
        'Correlation': select_features_correlation,
        'VIF': select_features_vif,
        'Pareto': select_features_pareto
    }
    
    best_fs_method = None
    best_fs_score = -1
    final_selected_features = []
    
    for name, func in fs_methods.items():
        selected = func(df_train_derived, 'quality')
        if len(selected) == 0: continue
            
        X_temp = df_train_derived[selected]
        y_temp = df_train_derived['quality']
        
        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTETomek(random_state=SEED, smote=SMOTE(k_neighbors=1))),
            ('model', RandomForestClassifier(random_state=SEED, n_jobs=-1))
        ])
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = cross_val_score(pipeline, X_temp, y_temp, cv=cv, scoring=qwk_scorer, n_jobs=-1)
        mean_score = scores.mean()
        
        print(f"  Method: {name}, CV Score: {mean_score:.4f}, Features: {len(selected)}")
        
        if mean_score > best_fs_score:
            best_fs_score = mean_score
            best_fs_method = name
            final_selected_features = selected

    print(f"Winner: {best_fs_method} (Score: {best_fs_score:.4f})")
    
    X_train_final = df_train_derived[final_selected_features]
    y_train_final = df_train_derived['quality']
    
    print("\n=== 3. Model Tuning (Optuna) - Expanded Suite ===")
    
    def objective(trial, model_name, X, y):
        scaler_name = trial.suggest_categorical('scaler', ['minmax', 'standard', 'robust'])
        if scaler_name == 'minmax': scaler = MinMaxScaler()
        elif scaler_name == 'standard': scaler = StandardScaler()
        else: scaler = RobustScaler()
            
        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': SEED
            }
            model = RandomForestClassifier(**params)
            scoring = qwk_scorer
            
        elif model_name == 'XGBRegressor':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': SEED,
                'n_jobs': -1
            }
            model = XGBRegressor(**params)
            scoring = qwk_scorer_regressor_sklearn
            
        elif model_name == 'LGBMClassifier':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'random_state': SEED,
                'n_jobs': -1,
                'verbose': -1
            }
            model = LGBMClassifier(**params)
            scoring = qwk_scorer
            
        elif model_name == 'SVC':
            params = {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
                'random_state': SEED
            }
            model = SVC(**params)
            scoring = qwk_scorer
            
        elif model_name == 'LogisticRegression':
            params = {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
                'max_iter': 1000,
                'random_state': SEED
            }
            model = LogisticRegression(**params)
            scoring = qwk_scorer
            
        elif model_name == 'ElasticNet':
            params = {
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'random_state': SEED
            }
            model = ElasticNet(**params)
            scoring = qwk_scorer_regressor_sklearn

        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler),
            ('smote', SMOTETomek(random_state=SEED, smote=SMOTE(k_neighbors=1))),
            ('model', model)
        ])
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return scores.mean()

    study_results = {}
    models_to_tune = ['RandomForest', 'XGBRegressor', 'LGBMClassifier', 'SVC', 'LogisticRegression', 'ElasticNet']
    
    for model_name in models_to_tune:
        print(f"Optimizing {model_name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model_name, X_train_final, y_train_final), n_trials=10)
        study_results[model_name] = study
        print(f"  Best QWK: {study.best_value:.4f}")

    best_model_name = max(study_results, key=lambda k: study_results[k].best_value)
    best_study = study_results[best_model_name]
    print(f"\nChampion Model: {best_model_name} (CV Score: {best_study.best_value:.4f})")
    
    # Retrain Champion
    best_params = best_study.best_params.copy()
    scaler_name = best_params.pop('scaler')
    if scaler_name == 'minmax': final_scaler = MinMaxScaler()
    elif scaler_name == 'standard': final_scaler = StandardScaler()
    else: final_scaler = RobustScaler()
    
    if best_model_name == 'RandomForest':
        final_model = RandomForestClassifier(**best_params, random_state=SEED)
    elif best_model_name == 'XGBRegressor':
        final_model = XGBRegressor(**best_params, random_state=SEED, n_jobs=-1)
    elif best_model_name == 'LGBMClassifier':
        final_model = LGBMClassifier(**best_params, random_state=SEED, n_jobs=-1, verbose=-1)
    elif best_model_name == 'SVC':
        final_model = SVC(**best_params, random_state=SEED)
    elif best_model_name == 'LogisticRegression':
        final_model = LogisticRegression(**best_params, random_state=SEED)
    elif best_model_name == 'ElasticNet':
        final_model = ElasticNet(**best_params, random_state=SEED)
        
    final_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', final_scaler),
        ('smote', SMOTETomek(random_state=SEED, smote=SMOTE(k_neighbors=1))),
        ('model', final_model)
    ])
    
    final_pipeline.fit(X_train_final, y_train_final)
    
    print("\n=== 4. Final Evaluation (Inference on Raw Test Data) ===")
    
    # Feature Engineering on Raw Test Data
    # We must replicate the EXACT feature generation process
    # Generate advanced features on test
    # Pass the top_features_used from training to ensure consistency
    df_test_for_gen = X_test_raw.copy()
    df_test_derived, _ = generate_advanced_features(df_test_for_gen, target_column='DUMMY_TARGET', top_features=top_features_used)
    
    # Select the SAME features
    X_test_final = df_test_derived[final_selected_features]
    X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan).fillna(X_test_final.median())
    
    y_pred = final_pipeline.predict(X_test_final)
    
    if is_regressor(final_model):
        y_pred = np.round(y_pred).astype(int)
        y_pred = np.clip(y_pred, y_encoded.min(), y_encoded.max())
        
    test_qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    print(f"Final Test QWK Score: {test_qwk:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le_target.classes_]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
