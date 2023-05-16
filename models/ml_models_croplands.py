# %%
import os
import pickle
import sys

sys.path.append(os.path.join(".."))


import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from src.model_utils import roc_auc_score_multiclass
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


# %% [markdown]
# ##  Data

# %%
# Features
with open(
    os.path.join("..", "data", "processed_files", "pkls", "X_down.pkl"), "rb"
) as fp:
    X = pickle.load(fp)

# Target Variable
with open(
    os.path.join("..", "data", "processed_files", "pkls", "y_down.pkl"), "rb"
) as fp:
    y = pickle.load(fp)


X_train = X["Train"]
X_test = X["Test"]
y_train = np.argmax(y["Train"], 1)
y_test = np.argmax(y["Test"], 1)

# %%
# Using LabelEncoder for target variable
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)

# Save target names in a list
target_names = list(np.unique(le.inverse_transform(y_train)))

# %% [markdown]
# ## Helper function


# %%
def run_classifier(clf, param_grid, target_names, title, le=None):
    # -----------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    # Randomized grid search
    n_iter_search = 25
    gs = RandomizedSearchCV(
        CalibratedClassifierCV(clf, method="isotonic"),
        param_distributions=param_grid,
        n_iter=n_iter_search,
        cv=cv,
        verbose=3,
        return_train_score=True,
        scoring="f1_macro",
        n_jobs=-1,
    )
    # -----------------------------------------------------
    # Train model
    gs.fit(X_train, y_train)
    # Saving the model:
    calibrated_clf = gs.best_estimator_
    # Saving the model:
    with open(
        os.path.join("..", "results", "pickle_models", title + "_crops_final.pkl"),
        "wb",
    ) as f:
        pickle.dump(calibrated_clf, f, protocol=4)
    # -----------------------------------------------------
    # Predict on test set
    y_pred = calibrated_clf.predict(X_test)  # type: ignore
    y_prob = calibrated_clf.predict_proba(X_test)  # type: ignore
    # Create file to print at
    f = open(
        os.path.join("..", "results", "pickle_models", title + "_crops_final.txt"),
        "w",
    )
    # Printing best parameters
    print("The best parameters are %s" % (gs.best_params_))
    print("The best parameters are %s" % (gs.best_params_), file=f)
    # -----------------------------------------------------
    # Printing metrics
    print(classification_report(y_test, y_pred))
    print(classification_report(y_test, y_pred), file=f)
    print("Roc_auc scores:")
    [
        print(str(key) + ": %.2f%%" % (value * 100))
        for key, value in roc_auc_score_multiclass(y_test, y_prob, le).items()
    ]
    print("Roc_auc scores:", file=f)
    [
        print(str(key) + ": %.2f%%" % (value * 100), file=f)
        for key, value in roc_auc_score_multiclass(y_test, y_prob, le).items()
    ]
    f.close()
    # -----------------------------------------------------
    #  # Plot confusion matrix
    plt.figure(figsize=(10, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False,
        fmt="d",
        linewidths=1,
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.show()
    # -----------------------------------------------------


# %% [markdown]
# ## XGBoost

calib_prefix = "base_estimator__"

# %%
xgbt = xgb.XGBClassifier(
    tree_method="hist", nthread=1, n_jobs=-1
)  # tree_method='gpu_hist', gpu_id=2

# Create parameters to search
param_grid = {
    calib_prefix + "objective": ["multi:softproba"],
    calib_prefix + "gamma": [1, 3, 5, 7, 9],
    calib_prefix + "max_depth": [3, 5, 8, 10, 12],
    calib_prefix + "reg_alpha": [0.1, 0.5, 2, 10, 50],
    calib_prefix + "reg_lambda": [0.1, 0.5, 2, 10, 50],
    calib_prefix + "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4],
    calib_prefix + "subsample": np.arange(0.5, 1.0, 0.1),
    calib_prefix + "colsample_bytree": np.arange(0.5, 1.0, 0.1),
    calib_prefix + "colsample_bylevel": np.arange(0.5, 1.0, 0.1),
    calib_prefix + "n_estimators": [50, 100, 250, 500, 750, 1000],
    calib_prefix + "min_child_weight": [1, 3, 5, 7, 9],
    calib_prefix + "num_class": [np.unique(y_train)],
}

run_classifier(xgbt, param_grid, target_names, "XGBoost", le)


# %% [markdown]
# ## LightGBM

# %%
lgbm = lgb.LGBMClassifier(boosting_type="gbdt", seed=500, n_jobs=-1)

# Create parameters to search
param_grid = {
    calib_prefix + "objective": ["softmax", "multiclass_ova", "num_class"],
    calib_prefix + "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4],
    calib_prefix + "n_estimators": [8, 24],
    calib_prefix + "num_leaves": [6, 8, 12, 16, 64],
    calib_prefix + "colsample_bytree": [0.65, 0.75, 0.8],
    calib_prefix + "subsample": [0.7, 0.75],
    calib_prefix + "reg_alpha": [1, 2, 6],
    calib_prefix + "reg_lambda": [1, 2, 6],
}

run_classifier(lgbm, param_grid, target_names, "LightGBM", le)


# %% [markdown]
# ## Logistic Regression

# %%
lr = LogisticRegression(solver="saga", max_iter=500, n_jobs=-1)

# Create parameters to search
param_grid = {
    calib_prefix + "penalty": ["l2", "l1"],
    calib_prefix + "C": [0.01, 0.1, 3, 5, 10],
}
run_classifier(lr, param_grid, target_names, "Logistic_Regression", le)
