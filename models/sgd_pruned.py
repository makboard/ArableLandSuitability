# %%
import os
import pickle

import numpy as np

import optuna
from sklearn.linear_model import SGDClassifier
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
N_TRAIN_ITER = 50
classes = np.unique(y_train)

# define study with hyperband pruner.
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=N_TRAIN_ITER, reduction_factor=3
    ),
)

for _ in range(50):
    trial = study.ask()

    param = {
        "loss" : trial.suggest_categorical("loss" ,
                                        ["log_loss", "modified_huber", "perceptron"]),
        "penalty" : trial.suggest_categorical("penalty",
                                            ["l1", "l2"]),
        "alpha" : trial.suggest_float("alpha",
                                    1e-5, 1e-1, log=True),
        "tol" : trial.suggest_float("tol",
                                    1e-6 , 1e-3, log=False),
        "fit_intercept" : trial.suggest_categorical("fit_intercept" ,
                                                    [True, False])
    }

    model = SGDClassifier(**param)
    pruned_trial = False
    
    for step in range(N_TRAIN_ITER):
        model.partial_fit(X_train, y_train, classes = classes)

        intermediate_value = model.score(X_val, y_val)
        trial.report(intermediate_value, step)

        if trial.should_prune():
            pruned_trial = True
            break

    if pruned_trial:
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)  # tell the pruned state
    else:
        y_pred = model.predict(X_val)
        pred_labels = np.rint(y_pred)
        f1 = f1_score(y_val, pred_labels, average="macro")
        study.tell(trial, f1)  # tell objective value


# %%

best_params = study.best_params
best_f1 = study.best_value

print("Best Parameters:", best_params)
print("Best Weighted F1-score:", best_f1)


# %%
best_model = SGDClassifier(
    **best_params, verbose=True
)
best_model.fit(X_train, y_train)
test_pred = best_model.predict(X_test)
test_f1 = f1_score(y_test, test_pred, average="macro")
print("Test macro F1-score:", test_f1)

# %%
# Save the trained model to a file
model_filename = os.path.join("..", "results", "pickle_models", "sgd_optimized_FR.pkl")
pickle.dump(best_model, open(model_filename, 'wb'))
# best_model.save_model(model_filename)
print("Model saved as:", model_filename)
