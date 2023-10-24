import os
import pickle
import sys

sys.path.append(os.path.join(".."))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from src.model_utils import roc_auc_score_multiclass

# Constants
DATA_PATH = os.path.join("..", "data", "processed_files", "pkls")
RESULTS_PATH = os.path.join("..", "results", "pickle_models")


def load_data():
    """Load features and targets from pickle files."""
    with open(os.path.join(DATA_PATH, "X_down.pkl"), "rb") as fp:
        X = pickle.load(fp)
    with open(os.path.join(DATA_PATH, "y_down.pkl"), "rb") as fp:
        y = pickle.load(fp)

    X_train, X_test = X["Train"], X["Test"]
    y_train, y_test = np.argmax(y["Train"], 1), np.argmax(y["Test"], 1)

    le = preprocessing.LabelEncoder().fit(y_train)
    y_train = le.transform(y_train)
    target_names = list(np.unique(le.inverse_transform(y_train)))

    return X_train, X_test, y_train, y_test, target_names, le


def run_classifier(
    clf, param_grid, X_train, y_train, X_test, y_test, target_names, title, le
):
    """Train and evaluate a classifier."""
    # -----------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    # Randomized grid search
    n_iter_search = 40
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
    with open(
        os.path.join(RESULTS_PATH, title + ".pkl"),
        "wb",
    ) as f:
        pickle.dump(calibrated_clf, f, protocol=4)
    # -----------------------------------------------------
    # Predict on test set
    y_pred = calibrated_clf.predict(X_test)
    y_prob = calibrated_clf.predict_proba(X_test)
    # Create file to print at
    f = open(
        os.path.join(RESULTS_PATH, title + ".txt"),
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


def main():
    X_train, X_test, y_train, y_test, target_names, le = load_data()
    calib_prefix = "base_estimator__"
    # XGBoost
    xgbt = xgb.XGBClassifier(tree_method="hist", nthread=1, n_jobs=-1)
    param_grid_xgb = {
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
    run_classifier(
        xgbt,
        param_grid_xgb,
        X_train,
        y_train,
        X_test,
        y_test,
        target_names,
        "XGBoost",
        le,
    )

    # LightGBM
    lgbm = lgb.LGBMClassifier(boosting_type="gbdt", seed=500, n_jobs=-1)
    param_grid_lgbm = {
        calib_prefix + "objective": ["softmax", "multiclass_ova"],
        calib_prefix + "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4],
        calib_prefix + "n_estimators": [8, 24],
        calib_prefix + "num_leaves": [6, 8, 12, 16, 64],
        calib_prefix + "colsample_bytree": [0.65, 0.75, 0.8],
        calib_prefix + "subsample": [0.7, 0.75],
        calib_prefix + "reg_alpha": [1, 2, 6],
        calib_prefix + "reg_lambda": [1, 2, 6],
        calib_prefix + "num_class": len(np.unique(y_train)),
    }
    run_classifier(
        lgbm,
        param_grid_lgbm,
        X_train,
        y_train,
        X_test,
        y_test,
        target_names,
        "LightGBM",
        le,
    )

    # Logistic Regression
    lr = LogisticRegression(solver="saga", max_iter=500, n_jobs=-1)
    param_grid_lr = {
        calib_prefix + "penalty": ["l2", "l1"],
        calib_prefix + "C": [0.01, 0.1, 3, 5, 10],
    }
    run_classifier(
        lr,
        param_grid_lr,
        X_train,
        y_train,
        X_test,
        y_test,
        target_names,
        "Logistic_Regression",
        le,
    )


if __name__ == "__main__":
    main()
