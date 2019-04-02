import pandas as pd
import os
from sklearn.externals import joblib
import scipy.stats as ss
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/train.csv'))
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/test.csv'))

X = train.drop(["target", "ID_code"], axis=1)
y = train["target"]
X_test = test.drop(["ID_code"], axis=1)
ID_code = test["ID_code"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)


def xgb_model(X_train, y_train, X_val, y_val, save_file, folds, param_comb, n_jobs, scoring):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=123)
    xgb = XGBClassifier(n_jobs=n_jobs, random_state=123)
    params = {"learning_rate": ss.uniform(0.001, 0.5),
              "n_estimators": [1000],
              "max_depth": ss.randint(3, 10),
              "min_child_weight": ss.randint(1, 15),
              "gamma": ss.randint(0, 10),
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              "reg_lambda": ss.uniform(0.05, 5),
              "reg_alpha": ss.uniform(0.05, 5)}
    model_xgb = RandomizedSearchCV(estimator=xgb,
                                   param_distributions=params,
                                   n_iter=param_comb,
                                   scoring=scoring,
                                   n_jobs=n_jobs,
                                   cv=skf.split(X_train, y_train),
                                   verbose=0,
                                   random_state=123)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', early_stopping_rounds=50)
    joblib.dump(model_xgb.best_estimator_, os.path.join(os.path.dirname(__file__), f'best_{save_file}'), compress=1)
    return model_xgb


def rf_model(X_train, y_train, save_file, folds, param_comb, n_jobs, scoring):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=123)
    rf = RandomForestClassifier(n_jobs=n_jobs, random_state=123)
    params = {"bootstrap": [True, False],
              "max_depth": ss.randint(3, 10),
              "max_features": ['auto', 'sqrt'],
              "min_samples_leaf": ss.randint(1, 10),
              "min_samples_split": ss.randint(2, 10),
              "n_estimators": [1000],
              "criterion": ["gini", "entropy"],
              "class_weight": ["balanced", "balanced_subsample", None]
              }
    model_rf = RandomizedSearchCV(estimator=rf,
                                  param_distributions=params,
                                  n_iter=param_comb,
                                  scoring=scoring,
                                  n_jobs=n_jobs,
                                  cv=skf.split(X_train, y_train),
                                  verbose=0,
                                  random_state=123)
    model_rf.fit(X_train, y_train)
    joblib.dump(model_rf.best_estimator_, os.path.join(os.path.dirname(__file__), f'best_{save_file}'), compress=1)
    return model_rf


def lgb_model(X_train, y_train, X_val, y_val, save_file, folds, param_comb, n_jobs, scoring):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=123)
    lgb = LGBMClassifier(n_jobs=n_jobs, random_state=123)
    params = {"num_leaves": ss.randint(2, 50),
              "max_depth": ss.randint(3, 10),
              "learning_rate": ss.uniform(0.001, 0.5),
              "n_estimators": [1000],
              "objective": ["binary"],
              "class_weight": ["balanced", None],
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              "reg_lambda": ss.uniform(0.05, 5),
              "reg_alpha": ss.uniform(0.05, 5),
              "min_child_weight": ss.randint(1, 15)}
    model_lgb = RandomizedSearchCV(estimator=lgb,
                                   param_distributions=params,
                                   n_iter=param_comb,
                                   scoring=scoring,
                                   n_jobs=n_jobs,
                                   cv=skf.split(X_train, y_train),
                                   verbose=0,
                                   random_state=123)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', early_stopping_rounds=50)
    joblib.dump(model_lgb.best_estimator_, os.path.join(os.path.dirname(__file__), f'best_{save_file}'), compress=1)
    return model_lgb


folds = 10
param_comb = 100
n_jobs = -1
scoring = "roc_auc"

model_xgb = xgb_model(X_train, y_train, X_val, y_val, 'xgb.pkl', folds, param_comb, n_jobs, scoring)
model_rf = rf_model(X_train, y_train, 'rf.pkl', folds, param_comb, n_jobs, scoring)
model_lgb = lgb_model(X_train, y_train, X_val, y_val, 'lgb.pkl', folds, param_comb, n_jobs, scoring)

# ### Optional
# model_xgb = joblib.load(os.path.join(os.path.dirname(__file__), 'xgb.sav'))
# model_rf = joblib.load(os.path.join(os.path.dirname(__file__), 'rf.sav'))
# model_lgb = joblib.load(os.path.join(os.path.dirname(__file__), 'lgb.sav'))

XGB_predictions = model_xgb.predict_proba(X_test)[:, 1]
RF_predictions = model_rf.predict_proba(X_test)[:, 1]
LGB_predictions = model_lgb.predict_proba(X_test)[:, 1]
ensemble_prediction = (XGB_predictions + RF_predictions + LGB_predictions) / 3

predictions = {"xgb": XGB_predictions,
               "rf": RF_predictions,
               "lgb": LGB_predictions,
               "ensemble": ensemble_prediction}

for pred in predictions:
    df = pd.DataFrame({"ID_code": ID_code,
                       "target": predictions[pred]})
    df.to_csv(os.path.join(os.path.dirname(__file__), f'{pred}_submit.csv'), index=None)
