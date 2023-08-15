import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
import pickle

import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def roc_auc_score_multiclass(actual_class, prob, le=None):
    """Calculate OvR roc_auc score for multiclass case

    Args:
        actual_class (ArrayLike): array of actual classes
        prob (ArrayLike): class probabilities for each sample
        le (LabelEncoder, optional): LabelEncoder for classes if needed. Defaults to None.

    Returns:
        dict: key -- class label; value -- OvR roc_auc score
    """
    # creating a set of all the unique classes using the actual class list
    unique_class = list(set(actual_class))
    roc_auc_dict = {}
    for i in range(len(unique_class)):
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != unique_class[i]]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_prob = prob[:, i]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_prob)

        if le is None:
            roc_auc_dict[[unique_class[i]][0]] = roc_auc
        else:
            roc_auc_dict[le.inverse_transform([unique_class[i]])[0]] = roc_auc

    return roc_auc_dict


def get_unique_values(data):
    """Generate sorted unique values of the dataset

    Args:
        data (ArrayLike): initial data

    Returns:
        list: list of tuples with counts for each unique value
    """
    # get unique values and counts of each value
    unique, counts = np.unique(data, return_counts=True)
    unique = list((x for x in unique))
    # display unique values and counts side by side
    values = []
    for i in range(len(unique)):
        values.append(tuple([unique[i], counts[i]]))
    values.sort(key=lambda tup: tup[1])
    return values


def downsample(X, y, oversampling=False):
    """Downsample the most frequent class to the second frequent one
    and optionally oversample the rest classes
    Downsampling is performed by RandomUnderSampler
    Oversampling is performed by SMOTE
    Args:
        X (DataFrame or ArrayLike): Initial features
        y (DataFrame or ArrayLike): Initial target variable
        oversampling (bool, optional): If True, oversampling is preformed
        for the rest classes. Defaults to False.

    Returns:
        X, y: downsampled dataset
    """

    # Define classes distribution
    counter = Counter(y)
    print("Initial data:")
    print(counter.most_common())

    # Define over- and undersampling strategies
    over = SMOTE(random_state=42, n_jobs=-1)
    under = RandomUnderSampler(
        sampling_strategy={counter.most_common()[0][0]: counter.most_common()[1][1]},  # type: ignore
        random_state=42,
    )

    # Define pipeline steps
    if oversampling:
        steps = [("u", under), ("o", over)]
    else:
        steps = [("u", under)]
    pipeline = Pipeline(steps=steps)

    # transform the dataset
    X, y = pipeline.fit_resample(X, y)

    # Print final distribution
    counter = Counter(y)
    print("Resampled data:")
    print(counter.most_common())

    return X, y


def drop_classes(X, y, classes_to_drop: list):
    """Drop specified classes from a dataset

    Args:
        X (DataFrame or ArrayLike): Initial features
        y (DataFrame or ArrayLike): Initial target variable
        classes_to_drop (list): List of classes to drop.

    Returns:
        X, y: modified dataset
    """
    X["target"] = y
    # Drop some classes from the data
    for i in range(len(classes_to_drop)):
        if np.isnan(classes_to_drop[i]):
            X = X.drop(X.index[np.isnan(X["target"])])
        else:
            X = X.drop(X.index[X["target"] == classes_to_drop[i]])
    # Updating target variable
    y = X["target"]
    # Drop target data from Features dataset
    X.drop("target", axis=1, inplace=True)

    return X, y


def reshape_data(X):
    """Prepare dataset in shape (N, num_of_monthly + num_of_static, 12)
    N - number of samples
    num_of_monthly - number of monthly features
    num_of_static - number of static features
    12 - number of months

    Args:
        X (DataFrame): Initial features

    Returns:
        ndarray: Modified dataset
    """
    non_sueqence_features = [
        "altitude",
        "_3",
        "_11",
        "_47",
        "std_precip_index",
        "latitude",
        "longitude",
    ]
    # Create list of monthly features
    list_of_monthly_features = [
        x for x in X.keys() if not any(elem in x for elem in non_sueqence_features)
    ]
    # Create list of static features
    list_of_static_features = [
        x for x in X.keys() if any(elem in x for elem in non_sueqence_features)
    ]
    # Create separate DataFrames for monthly and static features
    X_monthly = X.drop(columns=list_of_static_features)
    X_static = X.drop(columns=list_of_monthly_features)
    # Reshape monthly features
    X_tmp = X_monthly.to_numpy().reshape(
        X_monthly.shape[0], len(list_of_monthly_features) // 12, 12
    )
    # Create an empty output ndarray
    X_new = np.empty(
        (
            X_monthly.shape[0],
            12,
            len(list_of_monthly_features) // 12 + len(list_of_static_features),
        )
    )
    # Fill the output ndarray.
    for i in range(12):
        X_new[:, i, :] = np.concatenate([X_tmp[:, :, i], X_static.to_numpy()], axis=1)

    return X_new


def calculate_tpr_fpr(y_real, y_pred):
    """
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) 
            based on real and predicted observations

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    """

    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculates tpr and fpr
    tpr = TP / (TP + FN)  # sensitivity - true positive rate
    fpr = 1 - TN / (TN + FP)  # 1-specificity - false positive rate

    return tpr, fpr


def get_all_roc_coordinates(y_real, y_proba):
    """
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point 
        as a threshold for the predicion of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, 
            obtained by using the `.predict_proba()` method.

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    """
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def custom_multiclass_report(y_test, y_pred, y_prob):
    """Custom multiclass report which consist of:
    - classification report (Sklearn)
    - confusion matrix
    - precision recall curve
    - ROC curve
    - OvR roc_auc scores and plots
    - OvO roc_auc scores and plots
    Args:
        y_test (ArrayLike): actual values
        y_pred (ArrayLike): predicted values
        y_prob (ArrayLike): class probabilities for each sample
    """

    print(classification_report(y_test, y_pred))

    # -----------------------------------------------------
    # Plot confusion matrix
    plt.figure(figsize=(10, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=list(np.unique(y_test)),
        yticklabels=list(np.unique(y_test)),
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

    # precision recall curve
    precision = dict()
    recall = dict()
    classes = np.unique(y_test)
    for i in range(len(set(y_test))):
        precision[i], recall[i], _ = precision_recall_curve(
            label_binarize(y_test, classes=classes)[:, i], y_prob[:, i]
        )
        ap = average_precision_score(
            label_binarize(y_test, classes=classes)[:, i], y_prob[:, i]
        )
        plt.plot(
            recall[i],
            precision[i],
            lw=2,
            label="class {}".format(i) + " AP = {}".format(round(ap, 2)),
        )

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()
    # -----------------------------------------------------

    # roc curve
    fpr = dict()
    tpr = dict()
    for i in range(len(set(y_test))):
        fpr[i], tpr[i], _ = roc_curve(
            label_binarize(y_test, classes=classes)[:, i], y_prob[:, i]
        )
        plt.plot(fpr[i], tpr[i], lw=2, label="class {}".format(i))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.show()
    # -----------------------------------------------------

    # Plots the Probability Distributions and the ROC Curves One vs Rest
    plt.figure(figsize=(20, 4))
    bins = [i / 30 for i in range(30)] + [1]
    roc_auc_ovr = {}

    for i in range(len(classes)):
        # Gets the class
        c = classes[i]

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame()
        df_aux["class"] = [1 if y == c else 0 for y in y_test]
        df_aux["prob"] = y_prob[:, i]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(1, len(classes), i + 1)
        sns.histplot(x="prob", data=df_aux, hue="class", color="b", ax=ax, bins=bins)
        ax.set_title(c)
        ax.legend([f"Class: {c}", "Rest"])
        ax.set_xlabel(f"P(x = {c})")
        # Calculates the ROC AUC OvR
        roc_auc_ovr[c] = roc_auc_score(df_aux["class"], df_aux["prob"])

    # Displays the ROC AUC for each class
    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovr:
        avg_roc_auc += roc_auc_ovr[k]
        i += 1
        print(f"{k} ROC AUC OvR: {roc_auc_ovr[k]:.4f}")
    print(f"average ROC AUC OvR: {avg_roc_auc/i:.4f}")

    plt.tight_layout()
    plt.show()
    # -----------------------------------------------------

    # Plots the Probability Distributions and the ROC Curves One vs ONe
    plt.figure(figsize=(20, 7))
    bins = [i / 20 for i in range(20)] + [1]
    roc_auc_ovo = {}
    classes_combinations = []
    class_list = list(classes)
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            classes_combinations.append([class_list[i], class_list[j]])
            classes_combinations.append([class_list[j], class_list[i]])

    for i in range(len(classes_combinations)):
        # Gets the class
        comb = classes_combinations[i]
        c1 = comb[0]
        c2 = comb[1]
        c1_index = class_list.index(c1)
        title = str(c1) + " vs " + str(c2)

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame()
        df_aux["class"] = y_test
        df_aux["prob"] = y_prob[:, c1_index]

        # Slices only the subset with both classes
        df_aux = df_aux[(df_aux["class"] == c1) | (df_aux["class"] == c2)]
        df_aux["class"] = [1 if y == c1 else 0 for y in df_aux["class"]]
        df_aux = df_aux.reset_index(drop=True)

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 6, i + 1)
        sns.histplot(x="prob", data=df_aux, hue="class", color="b", ax=ax, bins=bins)
        ax.set_title(title)
        ax.legend([f"Class: {c1}", f"Class: {c2}"])
        ax.set_xlabel(f"P(x = {c1})")
        # Calculates the ROC AUC OvO
        roc_auc_ovo[title] = roc_auc_score(df_aux["class"], df_aux["prob"])

    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovo:
        avg_roc_auc += roc_auc_ovo[k]
        i += 1
        print(f"{k} ROC AUC OvO: {roc_auc_ovo[k]:.4f}")
    print(f"average ROC AUC OvO: {avg_roc_auc/i:.4f}")

    plt.tight_layout()
    plt.show()
    # -----------------------------------------------------


class CroplandDataModule_LSTM(pl.LightningDataModule):
    """
    This module defines a LightningDataModule class for loading and preparing data for a Cropland classification model using LSTM architecture.

    Args:
    X (dict): A dictionary containing the input data for Train, Validation, and Test sets.
    y (dict): A dictionary containing the corresponding target values for Train, Validation, and Test sets.
    batch_size (int): The batch size to be used for training and evaluation. Default is 128.
    """

    def __init__(self, X: dict, y: dict, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self.X_train, self.X_val, self.X_test = (
            torch.FloatTensor(X["Train"]),
            torch.FloatTensor(X["Val"]),
            torch.FloatTensor(X["Test"]),
        )
        self.y_train, self.y_val, self.y_test = (
            torch.LongTensor(y["Train"]),
            torch.LongTensor(y["Val"]),
            torch.LongTensor(y["Test"]),
        )

        self.dl_dict = {"batch_size": self.batch_size}

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = TensorDataset(self.X_train, self.y_train)
            self.dataset_val = TensorDataset(self.X_val, self.y_val)

        if stage == "test" or stage is None:
            self.dataset_test = TensorDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, shuffle=True, **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, **self.dl_dict)


class Crop_Transformer(nn.Module):
    """
    A PyTorch module implementing a Crop Transformer classifier.

    The Crop_Transformer module takes as input a sequence of feature vectors and applies a Transformer encoder (https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
    followed by two linear layers with ReLU activation to predict the output.

    Args:
    input_size (int): The number of expected features in the input (default: 52).
    hidden_size (int): The number of features in the hidden state (default: 104).
    num_layers (int): Number of recurrent layers (default: 4).
    output_size (int): The number of output logits (default: 4).

    Inputs:
    X (torch.Tensor): A tensor of shape (batch_size, sequence_length, input_size) containing the input sequence.

    Outputs:
    out (torch.Tensor): A tensor of shape (batch_size, output_size) containing the output logits.

    """

    def __init__(
        self,
        d_model=52,
        nhead=13,
        dim_feedforward=108,
        dropout=0.1,
        activation="relu",
        output_size=4,
    ) -> None:
        super(Crop_Transformer, self).__init__()
        self.transformer_enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.linearLayer1 = nn.Linear(d_model, d_model)
        self.linearLayer2 = nn.Linear(12 * d_model, output_size)
        self.act = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, X):
        out = self.act(self.transformer_enc(X) + X)
        out = self.act(self.linearLayer1(out) + out)
        out = self.flatten(out)
        out = self.linearLayer2(out)
        return out


class Crop_LSTM(nn.Module):
    """
    A PyTorch module implementing a Crop LSTM network.

    The Crop_LSTM module takes as input a sequence of feature vectors and applies a multi-layer LSTM network
    followed by two linear layers with ReLU activation to predict the output.

    Args:
    input_size (int): The number of expected features in the input (default: 52).
    hidden_size (int): The number of features in the hidden state (default: 104).
    num_layers (int): Number of recurrent layers (default: 4).
    output_size (int): The number of output logits (default: 4).

    Inputs:
    X (torch.Tensor): A tensor of shape (batch_size, sequence_length, input_size) containing the input sequence.

    Outputs:
    out (torch.Tensor): A tensor of shape (batch_size, output_size) containing the output logits.

    """

    def __init__(
        self,
        input_size=52,
        hidden_size=104,
        num_layers=4,
        output_size=4,
    ) -> None:
        super(Crop_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linearLayer1 = nn.Linear(hidden_size, hidden_size)
        self.linearLayer2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, X):
        out, _ = self.lstm(X)
        out = out[:, -1, :]
        out = self.act(self.linearLayer1(out))
        out = self.linearLayer2(out)
        return F.log_softmax(out, dim=1)


class CroplandDataModule_MLP(pl.LightningDataModule):
    """
    This module defines a LightningDataModule class for loading and preparing data for a Cropland classification model using MLP architecture.

    Args:
    X (dict): A dictionary containing the input data for Train, Validation, and Test sets.
    y (dict): A dictionary containing the corresponding target values for Train, Validation, and Test sets.
    batch_size (int): The batch size to be used for training and evaluation. Default is 128.
    """

    def __init__(self, X: dict, y: dict, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self.X_train, self.X_val, self.X_test = (
            torch.FloatTensor(X["Train"]),
            torch.FloatTensor(X["Val"]),
            torch.FloatTensor(X["Test"]),
        )
        self.y_train, self.y_val, self.y_test = (
            torch.LongTensor(y["Train"]),
            torch.LongTensor(y["Val"]),
            torch.LongTensor(y["Test"]),
        )

        self.dl_dict = {"batch_size": self.batch_size}

    def prepare_data(self):
        # Calculate class weights for imbalanced dataset
        _, counts = torch.unique(self.y_train.argmax(dim=1), return_counts=True)
        class_weights = 1.0 / torch.sqrt(counts.float())
        loss_weights = class_weights / class_weights.sum()
        ds = self.y_train.argmax(dim=1)
        weights = [loss_weights[i] for i in ds]
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = TensorDataset(self.X_train, self.y_train)
            self.dataset_val = TensorDataset(self.X_val, self.y_val)

        if stage == "test" or stage is None:
            self.dataset_test = TensorDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            sampler=self.sampler,
            **self.dl_dict,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, **self.dl_dict)


class Crop_MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) used for crop classification.

    Args:
        input_size (int): The number of input features (default: 162).
        output_size (int): The number of output logits (default: 4).

    Inputs:
        X (torch.Tensor): A tensor of shape (batch_size, input_size) containing input data.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, output_size) containing the output logits.
    """

    def __init__(self, input_size=162, output_size=4) -> None:
        super(Crop_MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 16 * input_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16 * input_size),
            nn.Linear(16 * input_size, 8 * input_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8 * input_size),
            nn.Linear(8 * input_size, 4 * input_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4 * input_size),
            nn.Linear(4 * input_size, 2 * input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * input_size, input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, input_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size // 2, input_size // 8),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size // 8, output_size),
        )

    def forward(self, X) -> torch.Tensor:
        output = F.log_softmax(self.net(X), dim=1)
        return output


class Crop_PL(pl.LightningModule):
    """
    PyTorch Lightning module for training a crop classification neural network.

    Args:
    net (torch.nn.Module): the neural network module to be trained.
    num_classes

    Attributes:
    softmax (nn.Softmax): softmax activation function.
    criterion (nn.CrossEntropyLoss): cross entropy loss function.
    optimizer (torch.optim.Adam)
    scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau)
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes=4,
        lr=1e-3,
        weight_decay=0.03,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.softmax = nn.Softmax()
        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        self.val_F1Score_best = torchmetrics.MaxMetric()

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=1
        )

        self.train_avg_precision = torchmetrics.AveragePrecision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.val_avg_precision = torchmetrics.AveragePrecision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.test_avg_precision = torchmetrics.AveragePrecision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )

        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.test_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )

        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )

        self.train_F1Score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.val_F1Score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )
        self.test_F1Score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, top_k=1, average="macro"
        )

    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        self.val_F1Score_best.reset()

    def model_step(self, batch):
        objs, target = batch
        predictions = self(objs)
        loss = self.loss(predictions, target.float())
        return loss, self.softmax(predictions), torch.argmax(target, dim=1)

    def training_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.train_loss(loss)
        self.train_accuracy(predictions, target)
        self.train_recall(predictions, target)
        self.train_precision(predictions, target)
        self.train_F1Score(predictions, target)
        self.train_avg_precision(predictions, target)

        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log(
            "train/F1Score",
            self.train_F1Score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("train/AP", self.train_avg_precision, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": predictions, "target": target}

    def validation_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.val_loss(loss)
        self.val_accuracy(predictions, target)
        self.val_recall(predictions, target)
        self.val_precision(predictions, target)
        self.val_F1Score(predictions, target)
        self.val_avg_precision(predictions, target)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log(
            "val/F1Score", self.val_F1Score, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/AP", self.val_avg_precision, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": predictions, "target": target}

    def validation_epoch_end(self, outputs):
        F1Score = self.val_F1Score.compute()
        self.val_F1Score_best(F1Score)
        self.log("val/F1Score_best", self.val_F1Score_best.compute(), prog_bar=False)

    def test_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.test_loss(loss)
        self.test_accuracy(predictions, target)
        self.test_recall(predictions, target)
        self.test_precision(predictions, target)
        self.test_F1Score(predictions, target)
        self.test_avg_precision(predictions, target)

        self.log("test/loss", self.test_loss, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log(
            "test/F1Score",
            self.test_F1Score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test/AP", self.test_avg_precision, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": predictions, "target": target}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        if scheduler is not None:
            scheduler = scheduler(
                optimizer=optimizer,
                patience=10,
                mode="min",
                factor=0.5,
                verbose=True,
                min_lr=1e-8,
                threshold=5e-4,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
