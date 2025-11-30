import json
import os

notebook_filename = "wine_quality_expert_analysis.ipynb"

cells = []

def add_markdown(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source]
    })

# --- RE-ADDING PREVIOUS SECTIONS FOR COMPLETENESS ---
# (Ideally I would append, but for the script I need to provide the full content)

add_markdown([
    "# Wine Quality Expert Analysis",
    "",
    "## 1. Introduction & Objectives",
    "This notebook presents a comprehensive analysis of the Wine Quality dataset. The goal is to predict wine quality based on physicochemical tests.",
    "",
    "**Objectives:**",
    "1.  **Advanced EDA**: Understand data distribution and relationships using statistical visualizations.",
    "2.  **Outlier Detection**: Identify and handle anomalies using robust statistical methods.",
    "3.  **Feature Engineering**: Create domain-specific features to enhance model performance.",
    "4.  **Preprocessing**: Build a robust pipeline including scaling and imbalance handling (SMOTE).",
    "5.  **Modeling & MLflow**: Train and tune models (ElasticNet, Random Forest, XGBoost) with experiment tracking.",
    "",
    "---"
])

add_code([
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import mlflow",
    "import mlflow.sklearn",
    "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler",
    "from sklearn.ensemble import RandomForestClassifier, IsolationForest",
    "from sklearn.linear_model import LogisticRegression",
    "from xgboost import XGBClassifier",
    "from imblearn.over_sampling import SMOTE",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score",
    "",
    "# Configuration",
    "sns.set(style='whitegrid')",
    "plt.rcParams['figure.figsize'] = (12, 6)",
    "import warnings",
    "warnings.filterwarnings('ignore')"
])

add_markdown([
    "## 2. Data Loading & Basic Inspection",
    "We load the dataset and perform a preliminary check of the structure and statistics."
])

add_code([
    "# Load dataset",
    "df = pd.read_csv('wine_data_homework.csv')",
    "",
    "# Check for index column and drop if necessary",
    "if 'Unnamed: 0' in df.columns:",
    "    df = df.drop(columns=['Unnamed: 0'])",
    "",
    "display(df.head())",
    "print(f'Shape: {df.shape}')"
])

add_code([
    "df.info()"
])

add_code([
    "df.describe().T"
])

add_markdown([
    "## 3. Advanced Exploratory Data Analysis (EDA)",
    "",
    "In this section, we utilize various visualization techniques to understand the univariate and multivariate distributions.",
    "",
    "### 3.1 Univariate Analysis",
    "We will examine the distribution of each feature using Histograms, KDE plots, Boxplots, and Violin plots.",
    "",
    "- **Histogram**: Shows the frequency distribution.",
    "- **KDE (Kernel Density Estimate)**: Estimates the probability density function.",
    "- **Boxplot**: Visualizes the five-number summary and potential outliers.",
    "- **Violin Plot**: Combines boxplot and KDE to show distribution shape."
])

add_code([
    "def plot_univariate(df, feature):",
    "    fig, axes = plt.subplots(2, 2, figsize=(16, 10))",
    "    ",
    "    # Histogram",
    "    sns.histplot(df[feature], kde=False, ax=axes[0, 0], color='skyblue')",
    "    axes[0, 0].set_title(f'{feature} - Histogram')",
    "    ",
    "    # KDE",
    "    sns.kdeplot(df[feature], shade=True, ax=axes[0, 1], color='orange')",
    "    axes[0, 1].set_title(f'{feature} - KDE')",
    "    ",
    "    # Boxplot",
    "    sns.boxplot(x=df[feature], ax=axes[1, 0], color='lightgreen')",
    "    axes[1, 0].set_title(f'{feature} - Boxplot')",
    "    ",
    "    # Violin Plot",
    "    sns.violinplot(x=df[feature], ax=axes[1, 1], color='purple')",
    "    axes[1, 1].set_title(f'{feature} - Violin Plot')",
    "    ",
    "    plt.tight_layout()",
    "    plt.show()",
    "",
    "# Example usage for 'alcohol'",
    "plot_univariate(df, 'alcohol')"
])

add_markdown([
    "### 3.2 Multivariate Analysis",
    "We use Pairplots and Correlation Heatmaps to understand relationships between variables.",
    "",
    "- **Pairplot**: Visualizes pairwise relationships, colored by 'quality'.",
    "- **Correlation Heatmap**: Shows the linear correlation coefficients between variables."
])

add_code([
    "# Correlation Heatmap",
    "plt.figure(figsize=(12, 10))",
    "correlation_matrix = df.corr()",
    "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)",
    "plt.title('Correlation Heatmap')",
    "plt.show()"
])

add_code([
    "# Pairplot (sampled for performance if dataset is large, but here it's small enough)",
    "# We focus on features with higher correlation with quality",
    "high_corr_features = correlation_matrix.index[abs(correlation_matrix['quality']) > 0.1].tolist()",
    "if 'quality' not in high_corr_features: high_corr_features.append('quality')",
    "",
    "sns.pairplot(df[high_corr_features], hue='quality', palette='viridis')",
    "plt.show()"
])

add_markdown([
    "## 4. Outlier Detection & Handling",
    "",
    "Outliers can significantly skew statistical models. We compare three methods:",
    "",
    "1.  **IQR (Interquartile Range)**: Removes data points outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.",
    "2.  **Z-Score**: Removes data points with a Z-score > 3 (assuming Gaussian distribution).",
    "3.  **Isolation Forest**: An unsupervised learning algorithm that isolates anomalies.",
    "",
    "We will visualize the impact of each method."
])

add_code([
    "from scipy import stats",
    "",
    "def detect_outliers_iqr(df, features):",
    "    outlier_indices = []",
    "    for col in features:",
    "        Q1 = df[col].quantile(0.25)",
    "        Q3 = df[col].quantile(0.75)",
    "        IQR = Q3 - Q1",
    "        outlier_step = 1.5 * IQR",
    "        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index",
    "        outlier_indices.extend(outlier_list_col)",
    "    return list(set(outlier_indices))",
    "",
    "def detect_outliers_zscore(df, features, threshold=3):",
    "    z_scores = np.abs(stats.zscore(df[features]))",
    "    return np.where(z_scores > threshold)[0]",
    "",
    "features_to_check = df.columns.drop('quality')",
    "",
    "# IQR",
    "outliers_iqr = detect_outliers_iqr(df, features_to_check)",
    "print(f'Outliers detected by IQR: {len(outliers_iqr)}')",
    "",
    "# Z-Score",
    "outliers_zscore = detect_outliers_zscore(df, features_to_check)",
    "print(f'Outliers detected by Z-Score: {len(outliers_zscore)}')",
    "",
    "# Isolation Forest",
    "iso = IsolationForest(contamination=0.05, random_state=42)",
    "y_pred_iso = iso.fit_predict(df[features_to_check])",
    "outliers_iso = df.index[y_pred_iso == -1].tolist()",
    "print(f'Outliers detected by Isolation Forest: {len(outliers_iso)}')"
])

add_markdown([
    "### Strategy Selection",
    "We will proceed with **Isolation Forest** or **IQR** depending on the overlap. For this analysis, we will create a clean dataframe by removing outliers detected by the chosen method (e.g., Isolation Forest for multivariate anomaly detection)."
])

add_code([
    "# Removing outliers using Isolation Forest for this pipeline",
    "df_clean = df.drop(outliers_iso).reset_index(drop=True)",
    "print(f'Original shape: {df.shape}, Cleaned shape: {df_clean.shape}')"
])

# --- NEW SECTIONS ---

add_markdown([
    "## 5. Feature Engineering",
    "",
    "We create new features based on domain knowledge to potentially improve model performance.",
    "",
    "- **Total Acidity**: Sum of fixed and volatile acidity.",
    "- **Acidity Ratio**: Ratio of fixed to volatile acidity.",
    "- **Sugar to Acidity Ratio**: Balance between sweetness and tartness.",
    "- **Alcohol Density**: Interaction between alcohol and density."
])

add_code([
    "df_fe = df_clean.copy()",
    "",
    "# 1. Total Acidity",
    "df_fe['total_acidity'] = df_fe['fixed acidity'] + df_fe['volatile acidity']",
    "",
    "# 2. Acidity Ratio",
    "df_fe['acidity_ratio'] = df_fe['fixed acidity'] / (df_fe['volatile acidity'] + 1e-6) # Avoid div by zero",
    "",
    "# 3. Sugar to Acidity",
    "df_fe['sugar_to_acidity'] = df_fe['residual sugar'] / (df_fe['total_acidity'] + 1e-6)",
    "",
    "# 4. Alcohol Density",
    "df_fe['alcohol_density'] = df_fe['alcohol'] * df_fe['density']",
    "",
    "# Check correlations with target",
    "new_features = ['total_acidity', 'acidity_ratio', 'sugar_to_acidity', 'alcohol_density', 'quality']",
    "print(df_fe[new_features].corr()['quality'].sort_values(ascending=False))"
])

add_markdown([
    "## 6. Preprocessing Pipeline",
    "",
    "We prepare the data for modeling:",
    "1.  **Split**: Separate Features (X) and Target (y).",
    "2.  **Train-Test Split**: 80/20 split.",
    "3.  **Scaling**: Standardize features using `StandardScaler`.",
    "4.  **Imbalance Handling**: Apply `SMOTE` (Synthetic Minority Over-sampling Technique) to the training set to address class imbalance."
])

add_code([
    "# 1. Split X and y",
    "X = df_fe.drop('quality', axis=1)",
    "y = df_fe['quality']",
    "",
    "# 2. Train-Test Split",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)",
    "",
    "# 3. Scaling",
    "scaler = StandardScaler()",
    "X_train_scaled = scaler.fit_transform(X_train)",
    "X_test_scaled = scaler.transform(X_test)",
    "",
    "# 4. SMOTE",
    "print(f'Before SMOTE: {y_train.value_counts().to_dict()}')",
    "smote = SMOTE(random_state=42)",
    "X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)",
    "print(f'After SMOTE: {y_train_resampled.value_counts().to_dict()}')"
])

add_markdown([
    "## 7. Modeling & MLflow Experiments",
    "",
    "We will train and evaluate three models:",
    "1.  **ElasticNet (Logistic Regression)**: Linear model with regularization.",
    "2.  **Random Forest**: Ensemble of decision trees.",
    "3.  **XGBoost**: Gradient boosting framework.",
    "",
    "We use **MLflow** to track experiments, parameters, and metrics."
])

add_code([
    "# Set MLflow experiment",
    "mlflow.set_experiment('Wine_Quality_Prediction')",
    "",
    "def train_evaluate_model(model, model_name, params=None):",
    "    with mlflow.start_run(run_name=model_name):",
    "        # Train",
    "        if params:",
    "            grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)",
    "            grid.fit(X_train_resampled, y_train_resampled)",
    "            best_model = grid.best_estimator_",
    "            best_params = grid.best_params_",
    "            print(f'Best Params for {model_name}: {best_params}')",
    "            mlflow.log_params(best_params)",
    "        else:",
    "            best_model = model",
    "            best_model.fit(X_train_resampled, y_train_resampled)",
    "            mlflow.log_params(best_model.get_params())",
    "        ",
    "        # Predict",
    "        y_pred = best_model.predict(X_test_scaled)",
    "        ",
    "        # Metrics",
    "        acc = accuracy_score(y_test, y_pred)",
    "        f1 = f1_score(y_test, y_pred, average='weighted')",
    "        ",
    "        print(f'--- {model_name} ---')",
    "        print(f'Accuracy: {acc:.4f}')",
    "        print(f'F1 Score: {f1:.4f}')",
    "        print(classification_report(y_test, y_pred))",
    "        ",
    "        # Log metrics",
    "        mlflow.log_metric('accuracy', acc)",
    "        mlflow.log_metric('f1_score', f1)",
    "        ",
    "        # Log model",
    "        mlflow.sklearn.log_model(best_model, model_name)",
    "        ",
    "        return best_model, acc, f1"
])

add_code([
    "# 1. Logistic Regression (ElasticNet equivalent)",
    "lr_params = {'C': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9], 'penalty': ['elasticnet'], 'solver': ['saga'], 'max_iter': [1000]}",
    "lr_model = LogisticRegression(multi_class='multinomial')",
    "best_lr, acc_lr, f1_lr = train_evaluate_model(lr_model, 'Logistic_ElasticNet', lr_params)"
])

add_code([
    "# 2. Random Forest",
    "rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}",
    "rf_model = RandomForestClassifier(random_state=42)",
    "best_rf, acc_rf, f1_rf = train_evaluate_model(rf_model, 'Random_Forest', rf_params)"
])

add_code([
    "# 3. XGBoost",
    "# Remap target to 0-indexed for XGBoost if necessary (usually handles it, but good practice)",
    "from sklearn.preprocessing import LabelEncoder",
    "le = LabelEncoder()",
    "y_train_resampled_enc = le.fit_transform(y_train_resampled)",
    "y_test_enc = le.transform(y_test)",
    "",
    "xgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]}",
    "xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)",
    "",
    "# Adjusted function for XGBoost to use encoded labels",
    "with mlflow.start_run(run_name='XGBoost'):",
    "    grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)",
    "    grid.fit(X_train_resampled_enc, y_train_resampled_enc)",
    "    best_xgb = grid.best_estimator_",
    "    mlflow.log_params(grid.best_params_)",
    "    y_pred_enc = best_xgb.predict(X_test_scaled)",
    "    acc_xgb = accuracy_score(y_test_enc, y_pred_enc)",
    "    f1_xgb = f1_score(y_test_enc, y_pred_enc, average='weighted')",
    "    print(f'--- XGBoost ---')",
    "    print(f'Accuracy: {acc_xgb:.4f}')",
    "    print(f'F1 Score: {f1_xgb:.4f}')",
    "    mlflow.log_metric('accuracy', acc_xgb)",
    "    mlflow.log_metric('f1_score', f1_xgb)",
    "    mlflow.sklearn.log_model(best_xgb, 'XGBoost')"
])

add_markdown([
    "## 8. Conclusion",
    "",
    "We have successfully analyzed the Wine Quality dataset, engineered features, and trained multiple models.",
    "",
    "**Summary of Results:**",
    "- **EDA**: Revealed key distributions and correlations.",
    "- **Outliers**: Removed anomalies to clean the data.",
    "- **Feature Engineering**: Added derived variables like `total_acidity`.",
    "- **Modeling**: Compared Logistic Regression, Random Forest, and XGBoost.",
    "",
    "Based on the F1-scores, the **Random Forest** (or XGBoost) model typically performs best for this tabular dataset due to its ability to capture non-linear relationships."
])

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(notebook_filename, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print(f"Notebook {notebook_filename} updated successfully.")
