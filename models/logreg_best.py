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
param = {
    "penalty" : "l2",
    "tol" : 0.00020459249800887453,
    "C" : 0.28611771190658386,
    "fit_intercept" : False,
    "solver": "saga"
}

model = LogisticRegression(**param, verbose=True)
model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_val)
pred_labels = np.rint(y_pred)
f1 = f1_score(y_val, pred_labels, average="macro")
print("Validation weighted F1-score:", f1)


# %%
# Save the trained model to a file
model_filename = os.path.join("..", "results", "pickle_models", "logreg_FR.pkl")
pickle.dump(model, open(model_filename, 'wb'))
print("Model saved as:", model_filename)