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
param = {
    "colsample_bylevel": 0.09313485735132296,
    "depth": 11,
    "boosting_type": "Plain",
    "bootstrap_type": "MVS",
    "eval_metric": "TotalF1",
    "iterations": 949,
    "learning_rate": 0.2087284056487625,
    "l2_leaf_reg": 5.388505443919626,
    "border_count": 72,
    "random_strength": 0.0014595230005226797,
    "min_data_in_leaf": 39,
    "auto_class_weights": None
    }

model = CatBoostClassifier(**param, verbose=True)

model.fit(
    X_train,
    y_train,    
    eval_set=[(X_val, y_val)],
    verbose=0,
    early_stopping_rounds=30
)

y_pred = model.predict(X_val)
pred_labels = np.rint(y_pred)
f1 = f1_score(y_val, pred_labels, average="macro")

print("Weighted F1-score on validation:", f1)


# %%
# Save the trained model to a file
model_filename = os.path.join("..", "results", "pickle_models", "catboost_FR.pkl")
model.save_model(model_filename)
model.save_model('catboost')  
print("Model saved as:", model_filename)