import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Disable interactive display of figures so script only saves images (no GUI popups)
plt.show = lambda *args, **kwargs: None
import seaborn as sns
import joblib
import os
# ensure folder exists for confusion matrix outputs early (used by KNN section)
os.makedirs('confusion_matrices', exist_ok=True)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, RocCurveDisplay)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')
warnings.filterwarnings('ignore', message='.*use_inf_as_na option is deprecated.*')
warnings.filterwarnings('ignore', message='.*SAMME.R algorithm.*deprecated.*')
warnings.filterwarnings('ignore', message='.*One or more of the test scores are non-finite.*')
warnings.filterwarnings('ignore', message='.*The total space of parameters.*')
warnings.filterwarnings('ignore', message='.*The NumPy global RNG was seeded.*')
warnings.filterwarnings('ignore', message='.*X does not have valid feature names.*')
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
# Handle inf values to suppress the seaborn warning
pd.options.mode.use_inf_as_na = False

# 1. DATA LOADING
file_path = 'labelled_combined_sensor_data_v0.csv' 
df = pd.read_csv(file_path)

# Save class distribution pie chart (0: normal, 1: failure)
os.makedirs('plots', exist_ok=True)
class_counts = df["failure"].value_counts().reindex([0,1]).fillna(0)
plt.figure(figsize=(6,6))
colors = ['#2ca02c', "#d62787"]
labels = [f"Normal (0): {int(class_counts.get(0,0))}", f"Failure (1): {int(class_counts.get(1,0))}"]
plt.pie(class_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title('Class distribution: Normal vs Failure')
plt.savefig(os.path.join('plots','class_distribution_pie.png'), bbox_inches='tight')
plt.close()
# Selected features based on prior analysis
selected_features = [
    'Temperature_1',        # Ambient temperature
    'Wind_Direction',       # Wind absolute direction
    'Temperature_2',        # Nacelle temperature
    'Temperature_3',        # Gearbox temperature
    'Wind_Speed_kmh',       # Windspeed
    'Turbine_Voltage',      # Calculated Voltage
    'Turbine_Current',      # Calculated Current
    'Turbine_Power',        # Total Active Power
    'Gas_Value',            # Gas sensor value
    'Line_Sensor',          # Line sensor value
    'Sound_Level',          # Sound level sensor value
    'Vibration_Value'       # Vibration sensor value
]

# Friendly display names for plots
feature_name_map = {
    'Temperature_1': 'Ambient_Temperature',
    'Temperature_2': 'Nacelle_Temperature',
    'Temperature_3': 'Gearbox_Temperature',
    'Wind_Direction': 'Wind_Direction',
    'Wind_Speed_kmh': 'Wind_Speed',
    'Turbine_Voltage': 'Turbine_Voltage',
    'Turbine_Current': 'Turbine_Current',
    'Turbine_Power': 'Turbine_Power',
    'Gas_Value': 'Gas_Value',
    'Line_Sensor': 'Line_Sensor',
    'Sound_Level': 'Sound_Level',
    'Vibration_Value': 'Vibration_Value'
}

def _map_display_names(cols):
    return [feature_name_map.get(c, c) for c in cols]

# Target variable
target = 'failure'

# Correlation of features with target and correlation matrix
try:
    corr_with_target = df[selected_features].corrwith(df[target])
    plt.figure(figsize=(10,6))
    display_names = _map_display_names(corr_with_target.index.tolist())
    sns.barplot(x=corr_with_target.values, y=display_names, palette='vlag')
    plt.xlabel('Correlation with failure (target)')
    plt.title('Feature correlation with target (failure)')
    plt.tight_layout()
    plt.savefig(os.path.join('plots','feature_corr_with_target_bar.png'), bbox_inches='tight')
    plt.close()

    # correlation matrix including target
    corr_matrix = df[selected_features + [target]].corr()
    # ensure numeric and rename rows/cols for display
    corr_matrix = corr_matrix.astype(float)
    corr_matrix_display = corr_matrix.copy()
    corr_matrix_display.index = _map_display_names(corr_matrix_display.index.tolist())
    corr_matrix_display.columns = _map_display_names(corr_matrix_display.columns.tolist())
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix_display, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                annot_kws={'size':10}, linewidths=0.5, square=True, cbar_kws={'shrink':0.75})
    plt.xticks(rotation=45, ha='right')
    plt.title('Correlation matrix (features + target)')
    plt.tight_layout()
    plt.savefig(os.path.join('plots','correlation_matrix.png'), bbox_inches='tight')
    plt.close()
except Exception as e:
    print('Could not compute/save correlation plots:', e)

# Only keep selected features and target
df_final = df[selected_features + [target]].copy()

# 5. MODEL TRAINING WITH 5-FOLD CROSS-VALIDATION
X = df_final[selected_features]
y = df_final[target]

# Split: reserve a hold-out test set, then perform 5-fold CV on the remaining train+val
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5-fold stratified cross-validation on train+val
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mlp_fold_results = []
rf_fold_results = []
fold_num = 1
cat_fold_results = []
xgb_fold_results = []
logistic_fold_results = []
dt_fold_results = []
stacking_fold_results = []
nb_fold_results = []
lgb_fold_results = []
svm_fold_results = []

for train_idx, val_idx in skf.split(X_trainval, y_trainval):
    X_tr = X_trainval.iloc[train_idx]
    X_val = X_trainval.iloc[val_idx]
    y_tr = y_trainval.iloc[train_idx]
    y_val = y_trainval.iloc[val_idx]

    # MLP pipeline
    mlp_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42))
    ])

    # Random Forest pipeline (include scaler for consistency)
    rf_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # Fit both
    mlp_pipeline.fit(X_tr, y_tr)
    rf_pipeline.fit(X_tr, y_tr)
    # CatBoost pipeline (optional if installed)
    if CatBoostClassifier is not None:
        cat_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('cat', CatBoostClassifier(iterations=500, learning_rate=0.05, random_state=42, verbose=0))
        ])
        cat_pipeline.fit(X_tr, y_tr)
        y_val_pred_cat = cat_pipeline.predict(X_val)
        acc_c = accuracy_score(y_val, y_val_pred_cat)
        prec_c = precision_score(y_val, y_val_pred_cat, zero_division=0)
        rec_c = recall_score(y_val, y_val_pred_cat, zero_division=0)
        f1_c = f1_score(y_val, y_val_pred_cat, zero_division=0)
        cat_fold_results.append({'fold': fold_num, 'accuracy': acc_c, 'precision': prec_c, 'recall': rec_c, 'f1_score': f1_c})
    else:
        y_val_pred_cat = None
        acc_c = prec_c = rec_c = f1_c = None

    # XGBoost pipeline (optional if installed)
    if XGBClassifier is not None:
        xgb_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('xgb', XGBClassifier(eval_metric='logloss', n_estimators=200, random_state=42))
        ])
        xgb_pipeline.fit(X_tr, y_tr)
        y_val_pred_xgb = xgb_pipeline.predict(X_val)
        acc_x = accuracy_score(y_val, y_val_pred_xgb)
        prec_x = precision_score(y_val, y_val_pred_xgb, zero_division=0)
        rec_x = recall_score(y_val, y_val_pred_xgb, zero_division=0)
        f1_x = f1_score(y_val, y_val_pred_xgb, zero_division=0)
        xgb_fold_results.append({'fold': fold_num, 'accuracy': acc_x, 'precision': prec_x, 'recall': rec_x, 'f1_score': f1_x})
    else:
        y_val_pred_xgb = None
        acc_x = prec_x = rec_x = f1_x = None

    # LightGBM pipeline (optional if installed)
    if LGBMClassifier is not None:
        lgb_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('lgb', LGBMClassifier(n_estimators=200, random_state=42, verbosity=-1, force_col_wise=True))
        ])
        lgb_pipeline.fit(X_tr, y_tr)
        y_val_pred_lgb = lgb_pipeline.predict(X_val)
        acc_lg = accuracy_score(y_val, y_val_pred_lgb)
        prec_lg = precision_score(y_val, y_val_pred_lgb, zero_division=0)
        rec_lg = recall_score(y_val, y_val_pred_lgb, zero_division=0)
        f1_lg = f1_score(y_val, y_val_pred_lgb, zero_division=0)
        lgb_fold_results.append({'fold': fold_num, 'accuracy': acc_lg, 'precision': prec_lg, 'recall': rec_lg, 'f1_score': f1_lg})
    else:
        y_val_pred_lgb = None
        acc_lg = prec_lg = rec_lg = f1_lg = None

    # Logistic Regression pipeline
    log_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('log', LogisticRegression(max_iter=1000, random_state=42))
    ])
    log_pipeline.fit(X_tr, y_tr)
    y_val_pred_log = log_pipeline.predict(X_val)
    acc_l = accuracy_score(y_val, y_val_pred_log)
    prec_l = precision_score(y_val, y_val_pred_log, zero_division=0)
    rec_l = recall_score(y_val, y_val_pred_log, zero_division=0)
    f1_l = f1_score(y_val, y_val_pred_log, zero_division=0)
    logistic_fold_results.append({'fold': fold_num, 'accuracy': acc_l, 'precision': prec_l, 'recall': rec_l, 'f1_score': f1_l})

    # Decision Tree pipeline
    dt_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('dt', DecisionTreeClassifier(random_state=42))
    ])
    dt_pipeline.fit(X_tr, y_tr)
    y_val_pred_dt = dt_pipeline.predict(X_val)
    acc_d = accuracy_score(y_val, y_val_pred_dt)
    prec_d = precision_score(y_val, y_val_pred_dt, zero_division=0)
    rec_d = recall_score(y_val, y_val_pred_dt, zero_division=0)
    f1_d = f1_score(y_val, y_val_pred_dt, zero_division=0)
    dt_fold_results.append({'fold': fold_num, 'accuracy': acc_d, 'precision': prec_d, 'recall': rec_d, 'f1_score': f1_d})

    # Naive Bayes pipeline
    nb_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('nb', GaussianNB())
    ])
    nb_pipeline.fit(X_tr, y_tr)
    y_val_pred_nb = nb_pipeline.predict(X_val)
    acc_nb = accuracy_score(y_val, y_val_pred_nb)
    prec_nb = precision_score(y_val, y_val_pred_nb, zero_division=0)
    rec_nb = recall_score(y_val, y_val_pred_nb, zero_division=0)
    f1_nb = f1_score(y_val, y_val_pred_nb, zero_division=0)
    nb_fold_results.append({'fold': fold_num, 'accuracy': acc_nb, 'precision': prec_nb, 'recall': rec_nb, 'f1_score': f1_nb})

    # Support Vector Machine pipeline
    svm_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42))
    ])
    svm_pipeline.fit(X_tr, y_tr)
    y_val_pred_svm = svm_pipeline.predict(X_val)
    acc_sv = accuracy_score(y_val, y_val_pred_svm)
    prec_sv = precision_score(y_val, y_val_pred_svm, zero_division=0)
    rec_sv = recall_score(y_val, y_val_pred_svm, zero_division=0)
    f1_sv = f1_score(y_val, y_val_pred_svm, zero_division=0)
    svm_fold_results.append({'fold': fold_num, 'accuracy': acc_sv, 'precision': prec_sv, 'recall': rec_sv, 'f1_score': f1_sv})

    # Predict on validation fold
    y_val_pred_mlp = mlp_pipeline.predict(X_val)
    y_val_pred_rf = rf_pipeline.predict(X_val)

    # Stacking ensemble pipeline (build base estimators; include optional ones if available)
    try:
        estimators = [
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
            ('log', LogisticRegression(max_iter=1000, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
        # optionally include CatBoost/XGBoost as base learners if available
        if CatBoostClassifier is not None:
            estimators.append(('cat', CatBoostClassifier(iterations=500, learning_rate=0.05, random_state=42, verbose=0)))
        if XGBClassifier is not None:
            estimators.append(('xgb', XGBClassifier(eval_metric='logloss', n_estimators=200, random_state=42)))

        stacking_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('stack', StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1))
        ])
        stacking_pipeline.fit(X_tr, y_tr)
        y_val_pred_stack = stacking_pipeline.predict(X_val)
        acc_s = accuracy_score(y_val, y_val_pred_stack)
        prec_s = precision_score(y_val, y_val_pred_stack, zero_division=0)
        rec_s = recall_score(y_val, y_val_pred_stack, zero_division=0)
        f1_s = f1_score(y_val, y_val_pred_stack, zero_division=0)
        stacking_fold_results.append({'fold': fold_num, 'accuracy': acc_s, 'precision': prec_s, 'recall': rec_s, 'f1_score': f1_s})
    except Exception as e:
        # if anything goes wrong (e.g., heavy compute or missing deps), record None and continue
        y_val_pred_stack = None
        acc_s = prec_s = rec_s = f1_s = None

    # Metrics MLP
    acc_m = accuracy_score(y_val, y_val_pred_mlp)
    prec_m = precision_score(y_val, y_val_pred_mlp, zero_division=0)
    rec_m = recall_score(y_val, y_val_pred_mlp, zero_division=0)
    f1_m = f1_score(y_val, y_val_pred_mlp, zero_division=0)

    # Metrics RF
    acc_r = accuracy_score(y_val, y_val_pred_rf)
    prec_r = precision_score(y_val, y_val_pred_rf, zero_division=0)
    rec_r = recall_score(y_val, y_val_pred_rf, zero_division=0)
    f1_r = f1_score(y_val, y_val_pred_rf, zero_division=0)

    mlp_fold_results.append({'fold': fold_num, 'accuracy': acc_m, 'precision': prec_m, 'recall': rec_m, 'f1_score': f1_m})
    rf_fold_results.append({'fold': fold_num, 'accuracy': acc_r, 'precision': prec_r, 'recall': rec_r, 'f1_score': f1_r})

    # start fold summary and show stacking metrics if available
    summary_parts = [f"Fold {fold_num}"]
    # show stacking metrics if available
    if y_val_pred_stack is not None:
        summary_parts.append(f"STACK acc: {acc_s:.4f}, prec: {prec_s:.4f}, rec: {rec_s:.4f}, f1: {f1_s:.4f}")
    # show NB metrics
    if y_val_pred_nb is not None:
        summary_parts.append(f"NB    acc: {acc_nb:.4f}, prec: {prec_nb:.4f}, rec: {rec_nb:.4f}, f1: {f1_nb:.4f}")
    # show LGB metrics
    if y_val_pred_lgb is not None:
        summary_parts.append(f"LGB   acc: {acc_lg:.4f}, prec: {prec_lg:.4f}, rec: {rec_lg:.4f}, f1: {f1_lg:.4f}")
    # show SVM metrics
    if y_val_pred_svm is not None:
        summary_parts.append(f"SVM   acc: {acc_sv:.4f}, prec: {prec_sv:.4f}, rec: {rec_sv:.4f}, f1: {f1_sv:.4f}")

    # add core model metrics to the fold summary
    summary_parts.append(f"MLP acc: {acc_m:.4f}, prec: {prec_m:.4f}, rec: {rec_m:.4f}, f1: {f1_m:.4f}")
    summary_parts.append(f"RF  acc: {acc_r:.4f}, prec: {prec_r:.4f}, rec: {rec_r:.4f}, f1: {f1_r:.4f}")
    if CatBoostClassifier is not None:
        summary_parts.append(f"CAT acc: {acc_c:.4f}, prec: {prec_c:.4f}, rec: {rec_c:.4f}, f1: {f1_c:.4f}")
    if XGBClassifier is not None:
        summary_parts.append(f"XGB acc: {acc_x:.4f}, prec: {prec_x:.4f}, rec: {rec_x:.4f}, f1: {f1_x:.4f}")
    summary_parts.append(f"LOG acc: {acc_l:.4f}, prec: {prec_l:.4f}, rec: {rec_l:.4f}, f1: {f1_l:.4f}")
    summary_parts.append(f"DT  acc: {acc_d:.4f}, prec: {prec_d:.4f}, rec: {rec_d:.4f}, f1: {f1_d:.4f}")
    print(" | ".join(summary_parts))
    fold_num += 1

# Summary of CV for both models
df_mlp_folds = pd.DataFrame(mlp_fold_results)
df_rf_folds = pd.DataFrame(rf_fold_results)
mlp_mean_metrics = df_mlp_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
rf_mean_metrics = df_rf_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
print("\nMLP CV Mean Metrics:\n", mlp_mean_metrics)
print("\nRF CV Mean Metrics:\n", rf_mean_metrics)
if CatBoostClassifier is not None:
    df_cat_folds = pd.DataFrame(cat_fold_results)
    cat_mean_metrics = df_cat_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
    print("\nCAT CV Mean Metrics:\n", cat_mean_metrics)
else:
    df_cat_folds = pd.DataFrame()
    cat_mean_metrics = pd.Series({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})
if XGBClassifier is not None:
    df_xgb_folds = pd.DataFrame(xgb_fold_results)
    xgb_mean_metrics = df_xgb_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
    print("\nXGB CV Mean Metrics:\n", xgb_mean_metrics)
else:
    df_xgb_folds = pd.DataFrame()
    xgb_mean_metrics = pd.Series({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})
# Logistic CV summary
df_logistic_folds = pd.DataFrame(logistic_fold_results)
logistic_mean_metrics = df_logistic_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
print("\nLOGISTIC CV Mean Metrics:\n", logistic_mean_metrics)
# Decision Tree CV summary
df_dt_folds = pd.DataFrame(dt_fold_results)
dt_mean_metrics = df_dt_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
print("\nDECISION TREE CV Mean Metrics:\n", dt_mean_metrics)

# Stacking CV summary (if computed)
if len(stacking_fold_results) > 0:
    df_stacking_folds = pd.DataFrame(stacking_fold_results)
    stacking_mean_metrics = df_stacking_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
    print("\nSTACKING CV Mean Metrics:\n", stacking_mean_metrics)
else:
    df_stacking_folds = pd.DataFrame()
    stacking_mean_metrics = pd.Series({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})

# Naive Bayes CV summary
if len(nb_fold_results) > 0:
    df_nb_folds = pd.DataFrame(nb_fold_results)
    nb_mean_metrics = df_nb_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
    print("\nNAIVE BAYES CV Mean Metrics:\n", nb_mean_metrics)
else:
    df_nb_folds = pd.DataFrame()
    nb_mean_metrics = pd.Series({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})

# LightGBM CV summary
if len(lgb_fold_results) > 0:
    df_lgb_folds = pd.DataFrame(lgb_fold_results)
    lgb_mean_metrics = df_lgb_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
    print("\nLIGHTGBM CV Mean Metrics:\n", lgb_mean_metrics)
else:
    df_lgb_folds = pd.DataFrame()
    lgb_mean_metrics = pd.Series({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})

# SVM CV summary
if len(svm_fold_results) > 0:
    df_svm_folds = pd.DataFrame(svm_fold_results)
    svm_mean_metrics = df_svm_folds[['accuracy', 'precision', 'recall', 'f1_score']].mean()
    print("\nSVM CV Mean Metrics:\n", svm_mean_metrics)
else:
    df_svm_folds = pd.DataFrame()
    svm_mean_metrics = pd.Series({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})

# --------------------------
# KNN: find best k via CV
# --------------------------
# We'll evaluate odd k values from 1..25 using the same StratifiedKFold
k_values = list(range(1, 26, 2))
knn_fold_results = []
for k in k_values:
    fold_num_k = 1
    for train_idx, val_idx in skf.split(X_trainval, y_trainval):
        X_tr = X_trainval.iloc[train_idx]
        X_val = X_trainval.iloc[val_idx]
        y_tr = y_trainval.iloc[train_idx]
        y_val = y_trainval.iloc[val_idx]

        knn_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ])
        knn_pipeline.fit(X_tr, y_tr)
        y_val_pred_knn = knn_pipeline.predict(X_val)

        acc_kn = accuracy_score(y_val, y_val_pred_knn)
        prec_kn = precision_score(y_val, y_val_pred_knn, zero_division=0)
        rec_kn = recall_score(y_val, y_val_pred_knn, zero_division=0)
        f1_kn = f1_score(y_val, y_val_pred_knn, zero_division=0)

        knn_fold_results.append({'k': k, 'fold': fold_num_k, 'accuracy': acc_kn, 'precision': prec_kn, 'recall': rec_kn, 'f1_score': f1_kn})
        fold_num_k += 1

df_knn_folds = pd.DataFrame(knn_fold_results)
# compute mean metrics per k
knn_mean_by_k = df_knn_folds.groupby('k')[['accuracy', 'precision', 'recall', 'f1_score']].mean()
# choose best k by mean F1
best_k = int(knn_mean_by_k['f1_score'].idxmax())
print(f"\nKNN CV mean metrics by k:\n{knn_mean_by_k}\nBest k by mean F1: {best_k}")

# plot mean F1 vs k
plt.figure(figsize=(8,5))
plt.plot(knn_mean_by_k.index, knn_mean_by_k['f1_score'], marker='o')
plt.xlabel('k (n_neighbors)')
plt.ylabel('Mean F1-score')
plt.title('KNN: mean F1-score vs k (5-fold CV on train+val)')
plt.grid(True)
plt.savefig('knn_k_selection.png', bbox_inches='tight')
plt.show()

# Train final KNN with best_k on entire train+val and save
final_knn = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])
final_knn.fit(X_trainval, y_trainval)
joblib.dump(final_knn, 'prototype_model_knn.pkl')
print(f"Saved final KNN pipeline to 'prototype_model_knn.pkl' (k={best_k}).")

# Evaluate KNN on hold-out test
y_test_pred_knn = final_knn.predict(X_test)
test_acc_kn = accuracy_score(y_test, y_test_pred_knn)
test_prec_kn = precision_score(y_test, y_test_pred_knn, zero_division=0)
test_rec_kn = recall_score(y_test, y_test_pred_knn, zero_division=0)
test_f1_kn = f1_score(y_test, y_test_pred_knn, zero_division=0)

print("\nKNN Test set classification report:")
print(classification_report(y_test, y_test_pred_knn, zero_division=0))
cm_knn = confusion_matrix(y_test, y_test_pred_knn)

# KNN confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=["Normal (0)", "Failure (1)"])
disp_knn.plot(cmap='coolwarm', values_format='d', ax=ax)
plt.title(f'Confusion Matrix - KNN Test Set (k={best_k})')
plt.savefig(os.path.join('confusion_matrices', 'confusion_knn.png'), bbox_inches='tight')
plt.show()

# KNN test metrics dataframe for Excel
df_knn_test = pd.DataFrame([{ 'k': best_k, 'accuracy': test_acc_kn, 'precision': test_prec_kn, 'recall': test_rec_kn, 'f1_score': test_f1_kn }])

# --------------------------
# Hyperparameter tuning for models (RandomizedSearchCV where applicable)
# Runs on the combined train+val set (`X_trainval`, `y_trainval`) using `skf`.
# Results are collected in `tuning_results` and later saved to the Excel workbook.
# --------------------------
tuning_results = {}

def _run_random_search(name, pipeline, param_dist, cv, X, y, n_iter=20):
    try:
        rs = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring='f1', n_jobs=-1, random_state=42, verbose=0)
        rs.fit(X, y)
        res_df = pd.DataFrame(rs.cv_results_)
        tuning_results[name] = {'cv_results': res_df, 'best_params': rs.best_params_, 'best_score': rs.best_score_}
        print(f"Tuning completed for {name}: best_score={rs.best_score_:.4f}")
    except Exception as e:
        print(f"Tuning failed for {name}: {e}")
        tuning_results[name] = None

# Random Forest
rf_pipe_tune = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestClassifier(random_state=42))])
rf_param_dist = {
    'rf__n_estimators': [50, 100, 200, 300],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}
_run_random_search('RandomForest', rf_pipe_tune, rf_param_dist, skf, X_trainval, y_trainval, n_iter=20)

# MLP
mlp_pipe_tune = Pipeline([('scaler', MinMaxScaler()), ('mlp', MLPClassifier(max_iter=1000, random_state=42))])
mlp_param_dist = {
    'mlp__hidden_layer_sizes': [(64,32), (128,64), (100,50,25)],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate_init': [0.001, 0.01]
}
_run_random_search('MLP', mlp_pipe_tune, mlp_param_dist, skf, X_trainval, y_trainval, n_iter=12)

# SVM hyperparameter tuning commented out (heavy/slow)
# If you want to enable SVM tuning, uncomment the block below.
# svm_pipe_tune = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC(probability=True, random_state=42))])
# svm_param_dist = {
#     'svm__C': [0.1, 1, 10, 100],
#     'svm__gamma': ['scale', 'auto', 0.01, 0.001],
#     'svm__kernel': ['rbf', 'poly']
# }
# _run_random_search('SVM', svm_pipe_tune, svm_param_dist, skf, X_trainval, y_trainval, n_iter=12)

# Decision Tree
dt_pipe_tune = Pipeline([('scaler', MinMaxScaler()), ('dt', DecisionTreeClassifier(random_state=42))])
dt_param_dist = {
    'dt__max_depth': [None, 5, 10, 20],
    'dt__min_samples_split': [2, 5, 10],
    'dt__min_samples_leaf': [1, 2, 4]
}
_run_random_search('DecisionTree', dt_pipe_tune, dt_param_dist, skf, X_trainval, y_trainval, n_iter=12)

# Logistic Regression
log_pipe_tune = Pipeline([('scaler', MinMaxScaler()), ('log', LogisticRegression(max_iter=1000, random_state=42))])
log_param_dist = {
    'log__C': [0.01, 0.1, 1, 10, 100],
    'log__penalty': ['l2']
}
_run_random_search('Logistic', log_pipe_tune, log_param_dist, skf, X_trainval, y_trainval, n_iter=8)

# KNN (small grid)
knn_pipe_tune = Pipeline([('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier())])
knn_param_grid = {
    'knn__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21,23,25],
    'knn__weights': ['uniform', 'distance']
}
try:
    gs_knn = GridSearchCV(knn_pipe_tune, param_grid=knn_param_grid, cv=skf, scoring='f1', n_jobs=-1)
    gs_knn.fit(X_trainval, y_trainval)
    tuning_results['KNN'] = {'cv_results': pd.DataFrame(gs_knn.cv_results_), 'best_params': gs_knn.best_params_, 'best_score': gs_knn.best_score_}
    print(f"Tuning completed for KNN: best_score={gs_knn.best_score_:.4f}")
except Exception as e:
    print('KNN tuning failed:', e)
    tuning_results['KNN'] = None

# XGBoost (if available)
if XGBClassifier is not None:
    xgb_pipe_tune = Pipeline([('scaler', MinMaxScaler()), ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))])
    xgb_param_dist = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__max_depth': [3, 6, 10],
        'xgb__learning_rate': [0.01, 0.05, 0.1]
    }
    _run_random_search('XGBoost', xgb_pipe_tune, xgb_param_dist, skf, X_trainval, y_trainval, n_iter=12)

# LightGBM (if available)
if LGBMClassifier is not None:
    lgb_pipe_tune = Pipeline([('scaler', MinMaxScaler()), ('lgb', LGBMClassifier(random_state=42, verbosity=-1, force_col_wise=True))])
    lgb_param_dist = {
        'lgb__n_estimators': [50, 100, 200],
        'lgb__num_leaves': [31, 50, 100],
        'lgb__learning_rate': [0.01, 0.05, 0.1]
    }
    _run_random_search('LightGBM', lgb_pipe_tune, lgb_param_dist, skf, X_trainval, y_trainval, n_iter=12)

# CatBoost (if available)
if CatBoostClassifier is not None:
    cat_pipe_tune = Pipeline([('scaler', MinMaxScaler()), ('cat', CatBoostClassifier(random_state=42, verbose=0))])
    cat_param_dist = {
        'cat__iterations': [100, 200, 500],
        'cat__learning_rate': [0.01, 0.05, 0.1]
    }
    _run_random_search('CatBoost', cat_pipe_tune, cat_param_dist, skf, X_trainval, y_trainval, n_iter=8)

# After tuning, print summary
print('\nHyperparameter tuning summary:')
for k, v in tuning_results.items():
    if v is None:
        print(f" - {k}: no results")
    else:
        print(f" - {k}: best_score={v['best_score']:.4f}, best_params={v['best_params']}")


# Train final pipeline on entire train+val and evaluate on hold-out test set
# Train final pipelines on entire train+val and evaluate on hold-out test set
final_mlp = Pipeline([
    ('scaler', MinMaxScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42))
])

final_rf = Pipeline([
    ('scaler', MinMaxScaler()),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
])

final_mlp.fit(X_trainval, y_trainval)
final_rf.fit(X_trainval, y_trainval)
if CatBoostClassifier is not None:
    final_cat = Pipeline([
        ('scaler', MinMaxScaler()),
        ('cat', CatBoostClassifier(iterations=500, learning_rate=0.05, random_state=42, verbose=0))
    ])
    final_cat.fit(X_trainval, y_trainval)
    joblib.dump(final_cat, 'prototype_model_catboost.pkl')
    print("Saved final CatBoost pipeline to 'prototype_model_catboost.pkl'.")
else:
    final_cat = None
    print("CatBoost not available (install catboost to enable). Skipping CatBoost final training.")

if XGBClassifier is not None:
    final_xgb = Pipeline([
        ('scaler', MinMaxScaler()),
        ('xgb', XGBClassifier(eval_metric='logloss', n_estimators=200, random_state=42))
    ])
    final_xgb.fit(X_trainval, y_trainval)
    joblib.dump(final_xgb, 'prototype_model_xgb.pkl')
    print("Saved final XGBoost pipeline to 'prototype_model_xgb.pkl'.")
else:
    final_xgb = None
    print("XGBoost not available (install xgboost to enable). Skipping XGBoost final training.")

# Train final LightGBM on entire train+val and save (if available)
if LGBMClassifier is not None:
    final_lgb = Pipeline([
        ('scaler', MinMaxScaler()),
        ('lgb', LGBMClassifier(n_estimators=200, random_state=42, verbosity=-1, force_col_wise=True))
    ])
    final_lgb.fit(X_trainval, y_trainval)
    joblib.dump(final_lgb, 'prototype_model_lgb.pkl')
    print("Saved final LightGBM pipeline to 'prototype_model_lgb.pkl'.")
else:
    final_lgb = None
    print("LightGBM not available (install lightgbm to enable). Skipping LightGBM final training.")

# Train final Logistic Regression on entire train+val
final_log = Pipeline([
    ('scaler', MinMaxScaler()),
    ('log', LogisticRegression(max_iter=1000, random_state=42))
])
final_log.fit(X_trainval, y_trainval)
joblib.dump(final_log, 'prototype_model_logistic.pkl')
print("Saved final Logistic Regression pipeline to 'prototype_model_logistic.pkl'.")

# Train final Decision Tree on entire train+val
final_dt = Pipeline([
    ('scaler', MinMaxScaler()),
    ('dt', DecisionTreeClassifier(random_state=42))
])
final_dt.fit(X_trainval, y_trainval)
joblib.dump(final_dt, 'prototype_model_dt.pkl')
print("Saved final Decision Tree pipeline to 'prototype_model_dt.pkl'.")

# Train final Support Vector Machine on entire train+val and save
final_svm = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])
final_svm.fit(X_trainval, y_trainval)
joblib.dump(final_svm, 'prototype_model_svm.pkl')
print("Saved final SVM pipeline to 'prototype_model_svm.pkl'.")

# Train final Stacking ensemble on entire train+val and save
try:
    final_estimators = [
        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('log', LogisticRegression(max_iter=1000, random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42))
    ]
    if CatBoostClassifier is not None:
        final_estimators.append(('cat', CatBoostClassifier(iterations=500, learning_rate=0.05, random_state=42, verbose=0)))
    if XGBClassifier is not None:
        final_estimators.append(('xgb', XGBClassifier(eval_metric='logloss', n_estimators=200, random_state=42)))

    final_stacking = Pipeline([
        ('scaler', MinMaxScaler()),
        ('stack', StackingClassifier(estimators=final_estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1))
    ])
    final_stacking.fit(X_trainval, y_trainval)
    joblib.dump(final_stacking, 'prototype_model_stacking.pkl')
    print("Saved final Stacking pipeline to 'prototype_model_stacking.pkl'.")
except Exception as e:
    final_stacking = None
    print('Could not train/save final stacking ensemble:', e)
# Train final Naive Bayes on entire train+val and save
final_nb = Pipeline([
    ('scaler', MinMaxScaler()),
    ('nb', GaussianNB())
])
final_nb.fit(X_trainval, y_trainval)
joblib.dump(final_nb, 'prototype_model_nb.pkl')
print("Saved final Naive Bayes pipeline to 'prototype_model_nb.pkl'.")

# Save final pipelines and scaler
joblib.dump(final_mlp, 'prototype_model_mlp.pkl')
joblib.dump(final_rf, 'prototype_model_rf.pkl')
# also save scaler from MLP pipeline (fitted on full train+val)
joblib.dump(final_mlp.named_steps['scaler'], 'prototype_scaler.pkl')
print("Saved final pipelines to 'prototype_model_mlp.pkl' and 'prototype_model_rf.pkl' and scaler to 'prototype_scaler.pkl'.")

# Evaluate on test set for both models
y_test_pred_mlp = final_mlp.predict(X_test)
y_test_pred_rf = final_rf.predict(X_test)
if final_xgb is not None:
    y_test_pred_xgb = final_xgb.predict(X_test)
else:
    y_test_pred_xgb = None
# Predict test with logistic
y_test_pred_log = final_log.predict(X_test)
# Predict test with Decision Tree
y_test_pred_dt = final_dt.predict(X_test)
# Predict test with Naive Bayes
y_test_pred_nb = final_nb.predict(X_test)
# Predict test with SVM
y_test_pred_svm = final_svm.predict(X_test)
# Predict test with Stacking (if trained)
if final_stacking is not None:
    try:
        y_test_pred_stack = final_stacking.predict(X_test)
    except Exception:
        y_test_pred_stack = None
else:
    y_test_pred_stack = None

test_acc_m = accuracy_score(y_test, y_test_pred_mlp)
test_prec_m = precision_score(y_test, y_test_pred_mlp, zero_division=0)
test_rec_m = recall_score(y_test, y_test_pred_mlp, zero_division=0)
test_f1_m = f1_score(y_test, y_test_pred_mlp, zero_division=0)

test_acc_r = accuracy_score(y_test, y_test_pred_rf)
test_prec_r = precision_score(y_test, y_test_pred_rf, zero_division=0)
test_rec_r = recall_score(y_test, y_test_pred_rf, zero_division=0)
test_f1_r = f1_score(y_test, y_test_pred_rf, zero_division=0)

print("\nMLP Test set classification report:")
print(classification_report(y_test, y_test_pred_mlp, zero_division=0))
print("\nRF Test set classification report:")
print(classification_report(y_test, y_test_pred_rf, zero_division=0))

cm_mlp = confusion_matrix(y_test, y_test_pred_mlp)
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
if final_cat is not None:
    y_test_pred_cat = final_cat.predict(X_test)
    cm_cat = confusion_matrix(y_test, y_test_pred_cat)
else:
    y_test_pred_cat = None
    cm_cat = None
if y_test_pred_xgb is not None:
    cm_xgb = confusion_matrix(y_test, y_test_pred_xgb)
else:
    cm_xgb = None
cm_log = confusion_matrix(y_test, y_test_pred_log)
cm_dt = confusion_matrix(y_test, y_test_pred_dt)
cm_nb = confusion_matrix(y_test, y_test_pred_nb)
if final_lgb is not None:
    y_test_pred_lgb = final_lgb.predict(X_test)
    cm_lgb = confusion_matrix(y_test, y_test_pred_lgb)
else:
    y_test_pred_lgb = None
    cm_lgb = None
cm_svm = confusion_matrix(y_test, y_test_pred_svm)
if y_test_pred_stack is not None:
    cm_stack = confusion_matrix(y_test, y_test_pred_stack)
else:
    cm_stack = None
print("MLP confusion matrix:\n", cm_mlp)
print("RF confusion matrix:\n", cm_rf)

print("\n--- TEST RESULTS (NUMBERS) ---")
print(f"MLP Accuracy: {test_acc_m:.4f}, Precision: {test_prec_m:.4f}, Recall: {test_rec_m:.4f}, F1: {test_f1_m:.4f}")
print(f"RF  Accuracy: {test_acc_r:.4f}, Precision: {test_prec_r:.4f}, Recall: {test_rec_r:.4f}, F1: {test_f1_r:.4f}")
print("-----------------------------------\n")

# Confusion matrix plots and save images
os.makedirs('confusion_matrices', exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))
disp_m = ConfusionMatrixDisplay(confusion_matrix=cm_mlp, display_labels=["Normal (0)", "Failure (1)"])
disp_m.plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Confusion Matrix - MLP Test Set')
plt.savefig(os.path.join('confusion_matrices', 'confusion_mlp.png'), bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
disp_r = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Normal (0)", "Failure (1)"])
disp_r.plot(cmap='Greens', values_format='d', ax=ax)
plt.title('Confusion Matrix - RF Test Set')
plt.savefig(os.path.join('confusion_matrices', 'confusion_rf.png'), bbox_inches='tight')
plt.show()

if cm_cat is not None:
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_c = ConfusionMatrixDisplay(confusion_matrix=cm_cat, display_labels=["Normal (0)", "Failure (1)"])
    disp_c.plot(cmap='Purples', values_format='d', ax=ax)
    plt.title('Confusion Matrix - CatBoost Test Set')
    plt.savefig(os.path.join('confusion_matrices', 'confusion_catboost.png'), bbox_inches='tight')
    plt.show()

if cm_xgb is not None:
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_x = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=["Normal (0)", "Failure (1)"])
    disp_x.plot(cmap='Oranges', values_format='d', ax=ax)
    plt.title('Confusion Matrix - XGBoost Test Set')
    plt.savefig(os.path.join('confusion_matrices', 'confusion_xgb.png'), bbox_inches='tight')
    plt.show()

# LightGBM confusion matrix
if cm_lgb is not None:
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_lg = ConfusionMatrixDisplay(confusion_matrix=cm_lgb, display_labels=["Normal (0)", "Failure (1)"])
    disp_lg.plot(cmap='YlOrBr', values_format='d', ax=ax)
    plt.title('Confusion Matrix - LightGBM Test Set')
    plt.savefig(os.path.join('confusion_matrices', 'confusion_lgb.png'), bbox_inches='tight')
    plt.show()

# SVM confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=["Normal (0)", "Failure (1)"])
disp_svm.plot(cmap='PuBu', values_format='d', ax=ax)
plt.title('Confusion Matrix - SVM Test Set')
plt.savefig(os.path.join('confusion_matrices', 'confusion_svm.png'), bbox_inches='tight')
plt.show()

# Logistic confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp_l = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=["Normal (0)", "Failure (1)"])
disp_l.plot(cmap='gray', values_format='d', ax=ax)
plt.title('Confusion Matrix - Logistic Regression Test Set')
plt.savefig(os.path.join('confusion_matrices', 'confusion_logistic.png'), bbox_inches='tight')
plt.show()

# Decision Tree confusion matrix and drawing the tree
fig, ax = plt.subplots(figsize=(8, 6))
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=["Normal (0)", "Failure (1)"])
disp_dt.plot(cmap='cividis', values_format='d', ax=ax)
plt.title('Confusion Matrix - Decision Tree Test Set')
plt.savefig(os.path.join('confusion_matrices', 'confusion_dt.png'), bbox_inches='tight')
plt.show()

# Draw and save the decision tree figure (use the trained tree inside pipeline)
try:
    dt_estimator = final_dt.named_steps['dt']

    # Helper: prune decision tree via cost-complexity pruning using CV to pick ccp_alpha
    def prune_decision_tree_pipeline(pipeline, X_full, y_full, cv=skf, scoring='f1'):
        try:
            scaler = pipeline.named_steps.get('scaler', None)
            est = pipeline.named_steps.get('dt', None)
            if est is None:
                return pipeline, None, None

            # Transform features the same way the tree saw them during training
            if scaler is not None:
                X_for_tree = scaler.transform(X_full)
            else:
                X_for_tree = X_full.values

            path = est.cost_complexity_pruning_path(X_for_tree, y_full)
            ccp_alphas = path.ccp_alphas
            if len(ccp_alphas) <= 1:
                return pipeline, None, None

            # drop the largest alpha which produces a single-node tree
            ccp_alphas = np.unique(ccp_alphas)[:-1]

            best_alpha = None
            best_score = -np.inf
            for a in ccp_alphas:
                try:
                    candidate = DecisionTreeClassifier(random_state=getattr(est, 'random_state', 42), ccp_alpha=a)
                    scores = cross_val_score(candidate, X_for_tree, y_full, cv=cv, scoring=scoring, n_jobs=-1)
                    mean_score = scores.mean()
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = a
                except Exception:
                    continue

            if best_alpha is None:
                return pipeline, None, None

            pruned = DecisionTreeClassifier(random_state=getattr(est, 'random_state', 42), ccp_alpha=best_alpha)
            pruned.fit(X_for_tree, y_full)
            if scaler is not None:
                new_pipeline = Pipeline([('scaler', scaler), ('dt', pruned)])
            else:
                new_pipeline = Pipeline([('dt', pruned)])
            return new_pipeline, best_alpha, best_score
        except Exception as e:
            print('Pruning failed:', e)
            return pipeline, None, None

    pruned_dt_pipeline, chosen_alpha, alpha_score = prune_decision_tree_pipeline(final_dt, X_trainval, y_trainval, cv=skf)
    if pruned_dt_pipeline is not None and pruned_dt_pipeline is not final_dt:
        dt_estimator = pruned_dt_pipeline.named_steps['dt']
        try:
            plt.figure(figsize=(20, 12))
            plot_tree(dt_estimator, feature_names=selected_features, class_names=['Normal', 'Failure'], filled=True, fontsize=10)
            plt.title(f'Decision Tree (pruned, ccp_alpha={chosen_alpha})')
            plt.savefig('decision_tree_pruned.png', bbox_inches='tight')
            plt.show()
            joblib.dump(pruned_dt_pipeline, 'prototype_model_dt_pruned.pkl')
            print(f"Saved pruned decision tree pipeline to 'prototype_model_dt_pruned.pkl' (ccp_alpha={chosen_alpha}, cv_score={alpha_score:.4f})")
        except Exception as e:
            print('Could not draw pruned decision tree:', e)
    else:
        # fallback to original unpruned tree
        plt.figure(figsize=(20, 12))
        plot_tree(dt_estimator, feature_names=selected_features, class_names=['Normal', 'Failure'], filled=True, fontsize=10)
        plt.title('Decision Tree')
        plt.savefig('decision_tree.png', bbox_inches='tight')
        plt.show()
except Exception as e:
    print('Could not draw decision tree:', e)

# Naive Bayes confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=["Normal (0)", "Failure (1)"])
disp_nb.plot(cmap='spring', values_format='d', ax=ax)
plt.title('Confusion Matrix - Naive Bayes Test Set')
plt.savefig(os.path.join('confusion_matrices', 'confusion_nb.png'), bbox_inches='tight')
plt.show()

# Stacking confusion matrix (if available)
if cm_stack is not None:
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_stack = ConfusionMatrixDisplay(confusion_matrix=cm_stack, display_labels=["Normal (0)", "Failure (1)"])
    disp_stack.plot(cmap='magma', values_format='d', ax=ax)
    plt.title('Confusion Matrix - Stacking Ensemble Test Set')
    plt.savefig(os.path.join('confusion_matrices', 'confusion_stacking.png'), bbox_inches='tight')
    plt.show()

# --------------------------
# ROC / AUC curves for all trained models
# --------------------------
model_scores = {}
def get_score_for_model(model, X):
    # return a score array for the positive class if available
    try:
        probs = model.predict_proba(X)
        return probs[:, 1]
    except Exception:
        try:
            # some models provide decision_function
            scores = model.decision_function(X)
            # if decision_function returns shape (n_samples, ), use it directly
            if scores.ndim == 1:
                return scores
            # else assume second column is positive class
            return scores[:, 1]
        except Exception:
            return None

# Prepare a mapping of model label -> pipeline object (if trained)
models_to_plot = {
    'MLP': final_mlp,
    'RandomForest': final_rf,
    'Logistic': final_log,
    'DecisionTree': final_dt,
    'NaiveBayes': final_nb,
    'SVM': final_svm,
    'KNN': final_knn,
    'Stacking': final_stacking
}
if final_cat is not None:
    models_to_plot['CatBoost'] = final_cat
if final_xgb is not None:
    models_to_plot['XGBoost'] = final_xgb
if final_lgb is not None:
    models_to_plot['LightGBM'] = final_lgb

plt.figure(figsize=(10, 8))
for name, mdl in models_to_plot.items():
    if mdl is None:
        continue
    scores = get_score_for_model(mdl, X_test)
    if scores is None:
        print(f"Skipping ROC for {name}: no probability/score method available.")
        continue
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    # save individual ROC plot
    plt.figure(figsize=(6,5))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name).plot()
    plt.title(f'ROC Curve - {name} (AUC={roc_auc:.3f})')
    plt.savefig(f'roc_{name.lower()}.png', bbox_inches='tight')
    plt.close()

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_auc_all_models.png', bbox_inches='tight')
plt.show()

# --------------------------
# SHAP explanations and summary plots for each model
# --------------------------
os.makedirs('SHAP', exist_ok=True)
try:
    import shap
except Exception:
    shap = None

def _get_pipeline_estimator_and_scaler(pipeline):
    """Return (estimator, scaler) extracted from a sklearn Pipeline-like object."""
    if hasattr(pipeline, 'named_steps'):
        scaler = pipeline.named_steps.get('scaler', None)
        # pick first non-scaler step as estimator
        est = None
        for k, v in pipeline.named_steps.items():
            if k != 'scaler':
                est = v
                break
        return est, scaler
    return pipeline, None



# --------------------------
# Feature importance for best model (if available)
# --------------------------
def compute_and_save_feature_importance(name, mdl, X_df):
    est, scaler = _get_pipeline_estimator_and_scaler(mdl)
    try:
        if scaler is not None:
            X_in = scaler.transform(X_df)
        else:
            X_in = X_df.values

        if hasattr(est, 'feature_importances_'):
            fi = est.feature_importances_
        elif hasattr(est, 'coef_'):
            coef = est.coef_
            if coef.ndim == 1:
                fi = np.abs(coef)
            else:
                fi = np.sum(np.abs(coef), axis=0)
        else:
            print(f'No feature importance available for {name}.')
            return

        fi_series = pd.Series(fi, index=X_df.columns).sort_values(ascending=False)
        plt.figure(figsize=(8, max(4, 0.25*len(fi_series))))
        sns.barplot(x=fi_series.values, y=fi_series.index)
        plt.title(f'Feature importance - {name}')
        plt.xlabel('Importance')
        out_path = os.path.join('SHAP', f'feature_importance_{name.lower()}.png')
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        # also save CSV
        fi_series.to_csv(os.path.join('SHAP', f'feature_importance_{name.lower()}.csv'))
        print(f"Saved feature importance for {name} -> {out_path}")
    except Exception as e:
        print(f'Could not compute feature importance for {name}: {e}')

try:
    if 'df_model_comp' in globals() and not df_model_comp.empty:
        best_name = df_model_comp.loc[0, 'model']
    elif 'best_model_name' in globals():
        best_name = best_model_name
    else:
        best_name = None

    if best_name is not None:
        best_mdl = models_to_plot.get(best_name)
        if best_mdl is not None:
            compute_and_save_feature_importance(best_name, best_mdl, X_test)
        else:
            print(f'Best model {best_name} not available for feature importance.')
except Exception as e:
    print('Error computing feature importance for best model:', e)

# --------------------------
# Compare models: accuracy, f1, roc_auc, errors -> OVERALL PERFORMANCE
# --------------------------
summary_rows = []
epsilon = 1e-6 # To prevent division by zero in performance calculation

for name, mdl in models_to_plot.items():
    if mdl is None:
        continue
    try:
        y_pred = mdl.predict(X_test)
        
        # Hata metrikleri için olasılıkları al (Eğer model desteklemiyorsa varsayılan atarız)
        try:
            y_prob = mdl.predict_proba(X_test)[:, 1]
            loss_val = log_loss(y_test, y_prob)
            brier_val = brier_score_loss(y_test, y_prob)
        except:
            # SVM gibi predict_proba desteklemeyen modeller için hard-coded hata tahmini
            loss_val = log_loss(y_test, y_pred)
            brier_val = brier_score_loss(y_test, y_pred)

    except Exception as e:
        print(f"Could not get predictions for {name}: {e}")
        continue
        
    # Temel Metrikler (Yüksek olması istenenler - Pay)
    acc = accuracy_score(y_test, y_pred)
    f1s = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    # ROC AUC
    scores = get_score_for_model(mdl, X_test)
    auc_val = 0.0 
    if scores is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, scores)
            auc_val = auc(fpr, tpr)
        except:
            pass

    # ==========================================
    # OVERALL PERFORMANCE HESAPLAMASI
    # Formula: 0.25*F1 + 0.20*AUC + 0.15*Acc + 0.15*Prec + 0.15*Rec + 0.10*(1-FPR)/(1+LogLoss+BrierError+FNR)
    # ==========================================

    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel()

    # FPR (False Positive Rate) ve FNR (False Negative Rate) hesaplaması
    fpr_val = fp / (tn + fp) if (tn + fp) > 0 else epsilon
    fnr = fn / (fn + tp) if (fn + tp) > 0 else epsilon

    # Overall Performance 
    overall_performance_raw = (
        0.25 * f1s + 
        0.20 * auc_val + 
        0.15 * acc + 
        0.15 * prec + 
        0.15 * rec + 
        0.10 * (1 - fpr_val) / (1 + loss_val + brier_val + fnr)
    )

    summary_rows.append({
        'model': name, 
        'accuracy': acc, 
        'precision': prec,
        'recall': rec,
        'f1_score': f1s, 
        'roc_auc': auc_val,
        'log_loss': loss_val,
        'brier_error': brier_val,
        'false_positive_rate': fpr_val,
        'false_negative_rate': fnr,
        'overall_performance': overall_performance_raw
    })

df_model_comp = pd.DataFrame(summary_rows)
if df_model_comp.empty:
    print('No model comparison metrics could be computed.')
else:
    # MODELLERİ YENİ METRİĞE GÖRE SIRALA
    df_model_comp = df_model_comp.sort_values(by='overall_performance', ascending=False).reset_index(drop=True)
    print('\nModel comparison (sorted by OVERALL PERFORMANCE):\n', df_model_comp)

    # BAR GRAFİĞİ: Accuracy, F1, ROC-AUC ve Raw Overall Performance
    plot_df = df_model_comp.set_index('model')[['accuracy', 'f1_score', 'roc_auc', 'overall_performance']].copy()
    
    plt.figure(figsize=(14, 7))
    plot_df[['accuracy', 'f1_score', 'roc_auc', 'overall_performance']].plot(kind='bar', figsize=(14,7))
    plt.title('Model comparison: Accuracy, F1, ROC-AUC & Overall Performance (Raw)')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('model_comparison_metrics.png', bbox_inches='tight')
    plt.show()

    # Formatlama: Excel için tabloyu resimdeki gibi hazırla
    df_excel = df_model_comp.copy()
    cols_to_pct = ['accuracy', 'precision', 'recall', 'f1_score', 'log_loss', 'brier_error', 'roc_auc', 'false_positive_rate', 'false_negative_rate']
    for c in cols_to_pct:
        df_excel[c] = (df_excel[c] * 100).map('{:.2f}%'.format)
    df_excel['overall_performance'] = df_excel['overall_performance'].map('{:.2f}'.format)
    
    df_excel = df_excel.rename(columns={
        'model': 'Model',
        'accuracy': 'Acc.',
        'precision': 'Prec.',
        'recall': 'Rec.',
        'f1_score': 'F1-Sc.',
        'log_loss': 'Log Loss',
        'brier_error': 'Bri. Err.',
        'roc_auc': 'ROC-AUC',
        'false_positive_rate': 'FPR',
        'false_negative_rate': 'FNR',
        'overall_performance': 'OP'
    })
    df_excel = df_excel[['Model', 'Acc.', 'Prec.', 'Rec.', 'F1-Sc.', 'Log Loss', 'Bri. Err.', 'ROC-AUC', 'FPR', 'FNR', 'OP']]

    # Global değişkene ata ki dosyanın sonunda da (üzerine yazıldığında) eklenebilsin
    global df_excel_global
    df_excel_global = df_excel

    # Excel'e Kaydetme
    try:
        with pd.ExcelWriter('cv_model_performance_results.xlsx', mode='a', engine='openpyxl') as writer:
            df_excel.to_excel(writer, sheet_name='Overall_Performance', index=False)
            best_model_name = df_model_comp.loc[0, 'model']
            pd.DataFrame([{'best_model': best_model_name}]).to_excel(writer, sheet_name='best_model', index=False)
    except Exception as e:
        print('Could not append to Excel. Saving CSV instead:', e)
        df_excel.to_csv('model_overall_performance.csv', index=False)

    # EN İYİ MODELİ (Overall Performance'ı en yüksek olanı) DİSKE KAYDET
    best_model_name = df_model_comp.loc[0, 'model']
    best_model_obj = models_to_plot.get(best_model_name)
    if best_model_obj is not None:
        joblib.dump(best_model_obj, 'prototype_model_best.pkl')
        print(f"Saved best model '{best_model_name}' (Highest Overall Performance) to 'prototype_model_best.pkl'.")
    else:
        print('Best model object not found; nothing saved.')

# --------------------------
# SHAP, LIME and feature-importance for the best model
# These are run after model comparison so `df_model_comp` and `best_model_name` are available.
# --------------------------
os.makedirs('SHAP', exist_ok=True)
try:
    import shap
except Exception:
    shap = None

def _get_pipeline_estimator_and_scaler(pipeline):
    if hasattr(pipeline, 'named_steps'):
        scaler = pipeline.named_steps.get('scaler', None)
        est = None
        for k, v in pipeline.named_steps.items():
            if k != 'scaler':
                est = v
                break
        return est, scaler
    return pipeline, None

def compute_and_save_shap(name, mdl, X_df):
    if shap is None:
        print(f"SHAP not installed — skipping SHAP for {name}.")
        return
    try:
        est, scaler = _get_pipeline_estimator_and_scaler(mdl)
        if scaler is not None:
            X_for_model = scaler.transform(X_df)
        else:
            X_for_model = X_df.values
        try:
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X_for_model)
            if isinstance(shap_values, list) or (hasattr(shap_values, 'shape') and getattr(shap_values, 'ndim', 0) == 3):
                try:
                    sv = shap_values[1]
                except Exception:
                    sv = shap_values[0]
            else:
                sv = shap_values
        except Exception:
            try:
                bg_size = min(100, X_for_model.shape[0])
                rng_idx = np.random.choice(X_for_model.shape[0], size=bg_size, replace=False)
                background = X_for_model[rng_idx]
                def model_fn(x):
                    if scaler is not None:
                        return est.predict_proba(scaler.transform(x))
                    return est.predict_proba(x)
                explainer = shap.KernelExplainer(model_fn, background)
                shap_values = explainer.shap_values(X_for_model, nsamples=100)
                if isinstance(shap_values, list):
                    sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    sv = shap_values
            except Exception as e:
                print(f"SHAP KernelExplainer failed for {name}: {e}")
                return
        plt.figure()
        try:
            display_names = _map_display_names(list(X_df.columns))
            shap.summary_plot(sv, features=X_df, feature_names=display_names, show=False)
            out_path = os.path.join('SHAP', f'shap_{name.lower()}.png')
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f"Saved SHAP summary for {name} -> {out_path}")
        except Exception as e:
            print(f"Could not create/save SHAP plot for {name}: {e}")
            plt.close()
    except Exception as e:
        print(f"Unexpected error computing SHAP for {name}: {e}")

os.makedirs('LIME', exist_ok=True)
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

def compute_and_save_lime(name, mdl, X_train_df, X_test_df, n_samples=3):
    if LimeTabularExplainer is None:
        print(f"LIME not installed — skipping LIME for {name}.")
        return
    try:
        est, scaler = _get_pipeline_estimator_and_scaler(mdl)
        def model_predict_proba(x_array):
            try:
                if scaler is not None:
                    x_trans = scaler.transform(x_array)
                else:
                    x_trans = x_array
                return est.predict_proba(x_trans)
            except Exception:
                try:
                    if scaler is not None:
                        x_trans = scaler.transform(x_array)
                    else:
                        x_trans = x_array
                    scores = est.decision_function(x_trans)
                    exp = np.exp(scores)
                    if exp.ndim == 1:
                        probs_pos = exp / (1 + exp)
                        probs = np.vstack([1-probs_pos, probs_pos]).T
                    else:
                        probs = exp / np.sum(exp, axis=1, keepdims=True)
                    return probs
                except Exception:
                    raise
        explainer = LimeTabularExplainer(X_train_df.values,
                                         feature_names=X_train_df.columns.tolist(),
                                         class_names=['Normal','Failure'],
                                         discretize_continuous=True)
        n_to_run = min(n_samples, X_test_df.shape[0])
        for i in range(n_to_run):
            instance = X_test_df.iloc[i].values
            try:
                exp = explainer.explain_instance(instance, model_predict_proba, num_features=min(len(X_test_df.columns), 7))
                fig = exp.as_pyplot_figure()
                out_path = os.path.join('LIME', f'lime_{name.lower()}_sample{i+1}.png')
                fig.savefig(out_path, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved LIME for {name} sample {i+1} -> {out_path}")
            except Exception as e:
                print(f"Could not compute/save LIME for {name} sample {i+1}: {e}")
    except Exception as e:
        print(f"Unexpected error computing LIME for {name}: {e}")

def compute_and_save_feature_importance(name, mdl, X_df):
    est, scaler = _get_pipeline_estimator_and_scaler(mdl)
    try:
        if scaler is not None:
            X_in = scaler.transform(X_df)
        else:
            X_in = X_df.values
        if hasattr(est, 'feature_importances_'):
            fi = est.feature_importances_
        elif hasattr(est, 'coef_'):
            coef = est.coef_
            if coef.ndim == 1:
                fi = np.abs(coef)
            else:
                fi = np.sum(np.abs(coef), axis=0)
        else:
            print(f'No feature importance available for {name}.')
            return
        mapped_idx = _map_display_names(list(X_df.columns))
        fi_series = pd.Series(fi, index=mapped_idx).sort_values(ascending=False)
        plt.figure(figsize=(8, max(4, 0.25*len(fi_series))))
        sns.barplot(x=fi_series.values, y=fi_series.index)
        plt.title(f'Feature importance - {name}')
        plt.xlabel('Importance')
        out_path = os.path.join('SHAP', f'feature_importance_{name.lower()}.png')
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        fi_series.to_csv(os.path.join('SHAP', f'feature_importance_{name.lower()}.csv'))
        print(f"Saved feature importance for {name} -> {out_path}")
    except Exception as e:
        print(f'Could not compute feature importance for {name}: {e}')

# Run SHAP, LIME and feature importance for the best model (if available)
try:
    best_name = None
    if 'df_model_comp' in globals() and not df_model_comp.empty:
        best_name = df_model_comp.loc[0, 'model']
    elif 'best_model_name' in globals():
        best_name = best_model_name

    if best_name is None:
        print('No best model found for SHAP/LIME/feature importance; skipping.')
    else:
        best_mdl = models_to_plot.get(best_name)
        if best_mdl is None:
            print(f'Best model {best_name} not available for explanations.')
        else:
            # SHAP
            if shap is not None:
                compute_and_save_shap(best_name, best_mdl, X_test)
            else:
                print('SHAP not installed; skipping SHAP generation.')
            # LIME
            if LimeTabularExplainer is not None:
                compute_and_save_lime(best_name, best_mdl, X_trainval, X_test)
            else:
                print('LIME package not available; install lime to generate LIME plots.')
            # Feature importance
            compute_and_save_feature_importance(best_name, best_mdl, X_test)
except Exception as e:
    print('Error during SHAP/LIME/feature importance stage:', e)

# Save CV fold metrics and test metrics to Excel (both models)
df_mlp_test = pd.DataFrame([{ 'accuracy': test_acc_m, 'precision': test_prec_m, 'recall': test_rec_m, 'f1_score': test_f1_m }])
df_rf_test = pd.DataFrame([{ 'accuracy': test_acc_r, 'precision': test_prec_r, 'recall': test_rec_r, 'f1_score': test_f1_r }])
df_nb_test = pd.DataFrame([{ 'accuracy': accuracy_score(y_test, y_test_pred_nb), 'precision': precision_score(y_test, y_test_pred_nb, zero_division=0), 'recall': recall_score(y_test, y_test_pred_nb, zero_division=0), 'f1_score': f1_score(y_test, y_test_pred_nb, zero_division=0) }])
if y_test_pred_cat is not None:
    df_cat_test = pd.DataFrame([{ 'accuracy': accuracy_score(y_test, y_test_pred_cat), 'precision': precision_score(y_test, y_test_pred_cat, zero_division=0), 'recall': recall_score(y_test, y_test_pred_cat, zero_division=0), 'f1_score': f1_score(y_test, y_test_pred_cat, zero_division=0) }])
else:
    df_cat_test = pd.DataFrame()

if y_test_pred_lgb is not None:
    df_lgb_test = pd.DataFrame([{ 'accuracy': accuracy_score(y_test, y_test_pred_lgb), 'precision': precision_score(y_test, y_test_pred_lgb, zero_division=0), 'recall': recall_score(y_test, y_test_pred_lgb, zero_division=0), 'f1_score': f1_score(y_test, y_test_pred_lgb, zero_division=0) }])
else:
    df_lgb_test = pd.DataFrame()
# SVM test results
df_svm_test = pd.DataFrame([{ 'accuracy': accuracy_score(y_test, y_test_pred_svm), 'precision': precision_score(y_test, y_test_pred_svm, zero_division=0), 'recall': recall_score(y_test, y_test_pred_svm, zero_division=0), 'f1_score': f1_score(y_test, y_test_pred_svm, zero_division=0) }])

if y_test_pred_xgb is not None:
    df_xgb_test = pd.DataFrame([{ 'accuracy': accuracy_score(y_test, y_test_pred_xgb), 'precision': precision_score(y_test, y_test_pred_xgb, zero_division=0), 'recall': recall_score(y_test, y_test_pred_xgb, zero_division=0), 'f1_score': f1_score(y_test, y_test_pred_xgb, zero_division=0) }])
else:
    df_xgb_test = pd.DataFrame()
# logistic test results
df_logistic_test = pd.DataFrame([{ 'accuracy': accuracy_score(y_test, y_test_pred_log), 'precision': precision_score(y_test, y_test_pred_log, zero_division=0), 'recall': recall_score(y_test, y_test_pred_log, zero_division=0), 'f1_score': f1_score(y_test, y_test_pred_log, zero_division=0) }])
# decision tree test results
df_dt_test = pd.DataFrame([{ 'accuracy': accuracy_score(y_test, y_test_pred_dt), 'precision': precision_score(y_test, y_test_pred_dt, zero_division=0), 'recall': recall_score(y_test, y_test_pred_dt, zero_division=0), 'f1_score': f1_score(y_test, y_test_pred_dt, zero_division=0) }])

# Stacking test results dataframe (if available)
if y_test_pred_stack is not None:
    df_stacking_test = pd.DataFrame([{ 'accuracy': accuracy_score(y_test, y_test_pred_stack), 'precision': precision_score(y_test, y_test_pred_stack, zero_division=0), 'recall': recall_score(y_test, y_test_pred_stack, zero_division=0), 'f1_score': f1_score(y_test, y_test_pred_stack, zero_division=0) }])
else:
    df_stacking_test = pd.DataFrame()

with pd.ExcelWriter('cv_model_performance_results.xlsx') as writer:
    if 'df_excel_global' in globals() and not df_excel_global.empty:
        df_excel_global.to_excel(writer, sheet_name='Overall_Performance', index=False)
    if 'df_model_comp' in globals() and not df_model_comp.empty:
        best_model_name = df_model_comp.loc[0, 'model']
        pd.DataFrame([{'best_model': best_model_name}]).to_excel(writer, sheet_name='best_model', index=False)
    
    df_mlp_folds.to_excel(writer, sheet_name='mlp_cv_folds', index=False)
    df_rf_folds.to_excel(writer, sheet_name='rf_cv_folds', index=False)
    if not df_cat_folds.empty:
        df_cat_folds.to_excel(writer, sheet_name='cat_cv_folds', index=False)
    if not df_xgb_folds.empty:
        df_xgb_folds.to_excel(writer, sheet_name='xgb_cv_folds', index=False)
    df_mlp_test.to_excel(writer, sheet_name='mlp_test_result', index=False)
    df_rf_test.to_excel(writer, sheet_name='rf_test_result', index=False)
    if not df_cat_test.empty:
        df_cat_test.to_excel(writer, sheet_name='cat_test_result', index=False)
    if not df_xgb_test.empty:
        df_xgb_test.to_excel(writer, sheet_name='xgb_test_result', index=False)
    df_logistic_test.to_excel(writer, sheet_name='logistic_test_result', index=False)
    df_dt_test.to_excel(writer, sheet_name='dt_test_result', index=False)
    # Naive Bayes sheets
    if not df_nb_folds.empty:
        df_nb_folds.to_excel(writer, sheet_name='nb_cv_folds', index=False)
    df_nb_test.to_excel(writer, sheet_name='nb_test_result', index=False)
    # LightGBM sheets
    if not df_lgb_folds.empty:
        df_lgb_folds.to_excel(writer, sheet_name='lgb_cv_folds', index=False)
    if not df_lgb_test.empty:
        df_lgb_test.to_excel(writer, sheet_name='lgb_test_result', index=False)
    # SVM sheets
    if not df_svm_folds.empty:
        df_svm_folds.to_excel(writer, sheet_name='svm_cv_folds', index=False)
    df_svm_test.to_excel(writer, sheet_name='svm_test_result', index=False)
    # Stacking sheets (if present)
    if not df_stacking_folds.empty:
        df_stacking_folds.to_excel(writer, sheet_name='stacking_cv_folds', index=False)
    if not df_stacking_test.empty:
        df_stacking_test.to_excel(writer, sheet_name='stacking_test_result', index=False)
    if 'stacking_mean_metrics' in globals() and stacking_mean_metrics is not None:
        try:
            stacking_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='stacking_cv_mean')
        except Exception:
            pass
    mlp_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='mlp_cv_mean')
    rf_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='rf_cv_mean')
    if cat_mean_metrics is not None:
        cat_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='cat_cv_mean')
    if xgb_mean_metrics is not None:
        xgb_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='xgb_cv_mean')
    logistic_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='logistic_cv_mean')
    dt_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='dt_cv_mean')
    if nb_mean_metrics is not None:
        nb_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='nb_cv_mean')
    if lgb_mean_metrics is not None:
        lgb_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='lgb_cv_mean')
    if svm_mean_metrics is not None:
        svm_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='svm_cv_mean')
    # Ensure stacking mean metrics are written if available
    try:
        if 'stacking_mean_metrics' in globals() and stacking_mean_metrics is not None:
            stacking_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name='stacking_cv_mean')
    except Exception:
        pass

def plot_feature_importance_for_best(mdl_pipeline, feature_names, out_path='feature_importance_best_model.png'):
    est, scaler = _get_pipeline_estimator_and_scaler(mdl_pipeline)
    # Prepare feature data for importance ordering (not used for importance values)
    if hasattr(est, 'feature_importances_'):
        importances = est.feature_importances_
    elif hasattr(est, 'coef_'):
        coef = est.coef_
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
    else:
        print('Best estimator does not expose feature importances or coefficients. Skipping plot.')
        return

    # Create bar chart with distinct colors per bar
    idx = np.argsort(importances)[::-1]
    sorted_feats = [feature_names[i] for i in idx]
    sorted_imps = importances[idx]
    plt.figure(figsize=(10,6))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(sorted_feats))]
    bars = plt.bar(range(len(sorted_feats)), sorted_imps, color=colors)
    plt.xticks(range(len(sorted_feats)), sorted_feats, rotation=45, ha='right')
    plt.ylabel('Importance (abs or value)')
    plt.title('Feature importances - Best model')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    # KNN mean metrics for the selected best_k (if available)
    try:
        if 'knn_mean_by_k' in globals() and 'best_k' in globals() and best_k in knn_mean_by_k.index:
            knn_mean_metrics = knn_mean_by_k.loc[best_k]
            knn_mean_metrics.to_frame(name='mean').to_excel(writer, sheet_name=f'knn_cv_mean_k{best_k}')
        else:
            knn_mean_metrics = None
    except Exception:
        knn_mean_metrics = None

    # Combined CV means for all models (one table)
    try:
        combined_means = {}
        def safe_series(s):
            if s is None:
                return pd.Series({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})
            # if it's a DataFrame row or Series with named metrics
            if isinstance(s, pd.Series):
                return s.reindex(['accuracy','precision','recall','f1_score'])
            return pd.Series({'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None})

        combined_means['MLP'] = safe_series(mlp_mean_metrics)
        combined_means['RandomForest'] = safe_series(rf_mean_metrics)
        combined_means['Logistic'] = safe_series(logistic_mean_metrics)
        combined_means['DecisionTree'] = safe_series(dt_mean_metrics)
        combined_means['NaiveBayes'] = safe_series(nb_mean_metrics)
        combined_means['SVM'] = safe_series(svm_mean_metrics)
        combined_means['KNN'] = safe_series(knn_mean_metrics)
        combined_means['Stacking'] = safe_series(stacking_mean_metrics if 'stacking_mean_metrics' in globals() else None)
        combined_means['CatBoost'] = safe_series(cat_mean_metrics if 'cat_mean_metrics' in globals() else None)
        combined_means['XGBoost'] = safe_series(xgb_mean_metrics if 'xgb_mean_metrics' in globals() else None)
        combined_means['LightGBM'] = safe_series(lgb_mean_metrics if 'lgb_mean_metrics' in globals() else None)

        df_combined_means = pd.DataFrame(combined_means).transpose()
        df_combined_means.to_excel(writer, sheet_name='cv_means_all_models')
    except Exception as e:
        print('Could not write combined CV means sheet:', e)
    # Write hyperparameter tuning results (one sheet per model + best_params)
    try:
        if 'tuning_results' in globals():
            for name, res in tuning_results.items():
                try:
                    if res is None:
                        continue
                    df_res = res.get('cv_results') if isinstance(res, dict) else None
                    if df_res is None:
                        continue
                    cols = [c for c in ['params', 'mean_test_score', 'std_test_score', 'rank_test_score'] if c in df_res.columns]
                    df_to_write = df_res[cols] if len(cols) > 0 else df_res
                    sheet_name = name + '_hyper'
                    sheet_name = sheet_name[:31]
                    df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                    # best params
                    best_params = res.get('best_params') if isinstance(res, dict) else None
                    if best_params is not None:
                        bp_sheet = (name + '_best_params')[:31]
                        pd.DataFrame([best_params]).to_excel(writer, sheet_name=bp_sheet, index=False)
                except Exception:
                    continue
    except Exception as e:
        print('Could not write tuning results to Excel:', e)
    except Exception as e:
        print('Could not write combined CV means sheet:', e)
    # confusion matrices numbers
    pd.DataFrame(cm_mlp, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='mlp_confusion')
    pd.DataFrame(cm_rf, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='rf_confusion')
    if cm_cat is not None:
        pd.DataFrame(cm_cat, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='cat_confusion')
    if cm_xgb is not None:
        pd.DataFrame(cm_xgb, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='xgb_confusion')
    pd.DataFrame(cm_log, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='logistic_confusion')
    pd.DataFrame(cm_dt, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='dt_confusion')
    pd.DataFrame(cm_nb, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='nb_confusion')
    if cm_lgb is not None:
        pd.DataFrame(cm_lgb, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='lgb_confusion')
    pd.DataFrame(cm_svm, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='svm_confusion')
    # Stacking confusion numbers
    if cm_stack is not None:
        pd.DataFrame(cm_stack, index=['true_0','true_1'], columns=['pred_0','pred_1']).to_excel(writer, sheet_name='stacking_confusion')

print("Saved CV fold metrics, test results and confusion matrices to 'cv_model_performance_results.xlsx' and confusion images.")

# --------------------------
# Classification reports per model -> save to Excel (one sheet per model)
# --------------------------
models_pred_map = {
    'MLP': y_test_pred_mlp,
    'RandomForest': y_test_pred_rf,
    'Logistic': y_test_pred_log,
    'DecisionTree': y_test_pred_dt,
    'NaiveBayes': y_test_pred_nb,
    'SVM': y_test_pred_svm,
    'KNN': y_test_pred_knn,
    'Stacking': y_test_pred_stack
}
if 'y_test_pred_cat' in globals() and y_test_pred_cat is not None:
    models_pred_map['CatBoost'] = y_test_pred_cat
if 'y_test_pred_xgb' in globals() and y_test_pred_xgb is not None:
    models_pred_map['XGBoost'] = y_test_pred_xgb
if 'y_test_pred_lgb' in globals() and y_test_pred_lgb is not None:
    models_pred_map['LightGBM'] = y_test_pred_lgb

try:
    with pd.ExcelWriter('model_classification_report.xlsx') as cr_writer:
        for name, preds in models_pred_map.items():
            sheet_name = name[:31]
            if preds is None:
                pd.DataFrame([{'note': 'No predictions available for this model.'}]).to_excel(cr_writer, sheet_name=sheet_name, index=False)
                continue
            try:
                cr_dict = classification_report(y_test, preds, output_dict=True)
                df_cr = pd.DataFrame(cr_dict).transpose()
                df_cr.to_excel(cr_writer, sheet_name=sheet_name)
            except Exception as e:
                pd.DataFrame([{'error': str(e)}]).to_excel(cr_writer, sheet_name=sheet_name, index=False)
    print("Saved per-model classification reports to 'model_classification_report.xlsx'.")
except Exception as e:
    print('Could not save model classification reports to Excel:', e)
