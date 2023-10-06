# %%
import os
import pickle

import numpy as np

import optuna
from sklearn.linear_model import LogisticRegression
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
        "penalty" : trial.suggest_categorical("penalty",
                                            ["l1", "l2"]),
        "tol" : trial.suggest_float("tol",
                                    1e-6 , 1e-3, log=False),
        "C" : trial.suggest_float("C",
                                    1e-2, 1, log=True),
        "fit_intercept" : trial.suggest_categorical("fit_intercept" ,
                                                    [True, False]),
        "solver": trial.suggest_categorical("solver",
                                            ["liblinear", "saga"])
        # "multi_class": trial.suggest_categorical("multi_class",
        #                                         ["auto", "ovr", "multinomial"])
    }

    if (param["solver"] == "liblinear") & (param["fit_intercept"]):
            param["intercept_scaling"] = trial.suggest_float("intercept_scaling",
                                                            1e-2, 1000, log=True)

    model = LogisticRegression(**param, verbose=True)

    model.fit(
        X_train,
        y_train
    )

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
best_model = LogisticRegression(
    **best_params, verbose=True
)
best_model.fit(X_train, y_train)
test_pred = best_model.predict(X_test)
test_f1 = f1_score(y_test, test_pred, average="macro")
print("Test macro F1-score:", test_f1)

# %%
# Save the trained model to a file
model_filename = os.path.join("..", "results", "pickle_models", "logreg_optimized_FR.pkl")
pickle.dump(best_model, open(model_filename, 'wb'))
print("Model saved as:", model_filename)