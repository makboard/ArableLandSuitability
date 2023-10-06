# %%
import os
import pickle

import numpy as np 

import optuna
from optuna.integration import CatBoostPruningCallback
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score


# %%
# Read dictionary pkl file
with open(os.path.join("..", "data", "processed_files", "pkls", "X_FR.pkl"), "rb") as fp:
    X = pickle.load(fp)

with open(os.path.join("..", "data", "processed_files", "pkls", "y_FR.pkl"), "rb") as fp:
    y = pickle.load(fp)


# %%
# Assuming you have X_train, y_train, X_val, and y_val defined
X_train, X_val, X_test = (X[key] for key in X.keys())
y_train, y_val, y_test = (y[key].argmax(axis=1) for key in y.keys())


# %%
def objective(trial):
    param = {
        "objective": "MultiClass",
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "eval_metric": "TotalF1",
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10),
        "border_count": trial.suggest_int("border_count", 32, 255, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "auto_class_weights": trial.suggest_categorical("auto_class_weights", [None, "Balanced", "SqrtBalanced"])
    }
    
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    model = CatBoostClassifier(**param, verbose=True)
    
    pruning_callback = CatBoostPruningCallback(trial, "TotalF1")
    
    model.fit(
        X_train,
        y_train,    
        eval_set=[(X_val, y_val)],
        verbose=0,
        early_stopping_rounds=30,
        callbacks=[pruning_callback],
    )
    pruning_callback.check_pruned()
    y_pred = model.predict(X_val)
    pred_labels = np.rint(y_pred)
    f1 = f1_score(y_val, pred_labels, average="macro")

    return f1


# %%
study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_f1 = study.best_value

print("Best Parameters:", best_params)
print("Best Weighted F1-score:", best_f1)


# %%
best_model = CatBoostClassifier(
    **best_params, verbose=True
)
best_model.fit(X_train, y_train)
test_pred = best_model.predict(X_test)
test_f1 = f1_score(y_test, test_pred, average="macro")
print("Test macro F1-score:", test_f1)

# %%
# Save the trained model to a file
model_filename = os.path.join("..", "results", "pickle_models", "catboost_optimized_FR.pkl")
best_model.save_model(model_filename)
print("Model saved as:", model_filename)
