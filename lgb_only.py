import pandas as pd
import os
from sklearn.externals import joblib
import scipy.stats as ss
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from lightgbm.sklearn import LGBMClassifier

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/train.csv'), nrows=1000)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data/test.csv'), nrows=1000)

X = train.drop(["target", "ID_code"], axis=1)
y = train["target"]
X_test = test.drop(["ID_code"], axis=1)
ID_code = test["ID_code"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)


def lgb_model(X_train, y_train, X_val, y_val, save_file, folds, param_comb, n_jobs, scoring):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=123)
    lgb = LGBMClassifier(n_jobs=n_jobs, random_state=123)
    params = {"num_leaves": [3],
              "max_depth": ss.randint(2, 25),
              "learning_rate": [0.2575640770995011],
              "n_estimators": [5000],
              "objective": ["binary"],
              "class_weight": ["balanced", None],
              "subsample": [0.7],
              "colsample_bytree": [0.6],
              "reg_lambda": [1.6599030323415402],
              "reg_alpha": [0.7044747533204038],
              "min_child_weight": [7]}
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

model_lgb = lgb_model(X_train, y_train, X_val, y_val, 'lgb_n5000.pkl', folds, param_comb, n_jobs, scoring)
LGB_predictions = model_lgb.predict_proba(X_test)[:, 1]
predictions = {"lgb": LGB_predictions,}

for pred in predictions:
    df = pd.DataFrame({"ID_code": ID_code,
                       "target": predictions[pred]})
    df.to_csv(os.path.join(os.path.dirname(__file__), f'{pred}_submit.csv'), index=None)
