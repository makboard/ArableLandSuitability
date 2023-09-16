from unicodedata import bidirectional
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import math
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
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
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader, TensorDataset, Dataset
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


def downsample(X, y, oversampling=False, oversampling_strategy='ROS'):
    """Downsample the most frequent class to the second frequent one
    and optionally oversample the rest classes
    Downsampling is performed by RandomUnderSampler
    Oversampling is performed by SMOTE or RandomOverSampler
    Args:
        X (DataFrame or ArrayLike): Initial features
        y (DataFrame or ArrayLike): Initial target variable
        oversampling (bool, optional): If True, oversampling is performed
        for the rest classes. Defaults to False.

    Returns:
        X, y: downsampled dataset
    """

    # Define classes distribution
    counter = Counter(y)
    print("Initial data:")
    print(counter.most_common())

    # Define over- and undersampling strategies
    overSMOTE = SMOTE(random_state=42, n_jobs=-1)
    overROS =  RandomOverSampler(random_state=42)
    under = RandomUnderSampler(
        sampling_strategy={counter.most_common()[0][0]: counter.most_common()[1][1]},  
        random_state=42,
    )

    # Define pipeline steps
    if oversampling:
        if oversampling_strategy == 'ROS':
            steps = [("u", under), ("o", overROS)]
        elif oversampling_strategy == 'SMOTE':
            steps = [("u", under), ("o", overSMOTE)]
        else:
            raise ValueError("Not implemented oversampling strategy!")
    else:
        steps = [("u", under)]
    pipeline = Pipeline(steps=steps)

    # Transform the dataset
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
    """Prepare dataset in shape (N, 12, num_of_monthly), (N, num_of_static)
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
    X_monthly = X[list_of_monthly_features]
    X_static = X[list_of_static_features].to_numpy()
    # Reshape monthly features
    X_monthly = X_monthly.to_numpy().reshape(
        X_monthly.shape[0], len(list_of_monthly_features) // 12, 12)
    X_monthly = np.transpose(X_monthly, (0, 2, 1)) # samples, sequence, features

    return X_monthly, X_static, list_of_monthly_features, list_of_static_features


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
    # -----------------------------------------------------

class CroplandDataset(Dataset):
    def __init__(self, X, y):
        self.X_monthly = X[0]  
        self.X_static = X[1] 
        self.y = y 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_monthly = self.X_monthly[idx]
        x_static = self.X_static[idx]
        target = self.y[idx]
        
        return (x_monthly, x_static), target

class CroplandDataModuleLSTM(pl.LightningDataModule):
    """
    This module defines a LightningDataModule class for loading and preparing data for a Cropland classification model using LSTM architecture.

    Args:
    X (dict): A dictionary containing the input data for Train, Validation, and Test sets.
    y (dict): A dictionary containing the corresponding target values for Train, Validation, and Test sets.
    batch_size (int): The batch size to be used for training and evaluation. Default is 128.
    """

    def __init__(self, X: dict, y: dict, batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_monthly_train, self.X_monthly_val, self.X_monthly_test = (
            torch.FloatTensor(X["Train"][0]),
            torch.FloatTensor(X["Val"][0]),
            torch.FloatTensor(X["Test"][0]),
        )
        self.X_static_train, self.X_static_val, self.X_static_test = (
            torch.FloatTensor(X["Train"][1]),
            torch.FloatTensor(X["Val"][1]),
            torch.FloatTensor(X["Test"][1]),
        )
        self.y_train, self.y_val, self.y_test = (
            torch.LongTensor(y["Train"]),
            torch.LongTensor(y["Val"]),
            torch.LongTensor(y["Test"]),
        )

        self.dl_dict = {"batch_size": self.batch_size, "num_workers": self.num_workers}

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = CroplandDataset((self.X_monthly_train, self.X_static_train), self.y_train)
            self.dataset_val = CroplandDataset((self.X_monthly_val, self.X_static_val), self.y_val)

        if stage == "test" or stage is None:
            self.dataset_test = CroplandDataset((self.X_monthly_test, self.X_static_test), self.y_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, shuffle=True, pin_memory=True, **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, pin_memory=True, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, pin_memory=True, **self.dl_dict)
    
class CropTransformer(nn.Module):
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
        input_size=10,
        d_model=256,
        nhead=16,
        dim_feedforward=256,
        hidden_size=300,
        num_layers=2,
        dropout=0.2,
        activation="relu",
        output_size=4,
    ) -> None:
        super(CropTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        
        self.transformer_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        ), num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, X):
        montly_input = X[0]
        embedded = self.embedding(montly_input)
        encoded = self.transformer_enc(embedded)
        output = encoded[:, -1, :]
        output = self.classifier(torch.cat((output, X[1]), dim=1))
        return output


class CropLSTM(nn.Module):
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
        X (tuple): X[0] is a tensor of shape (batch_size, sequence_length, input_size) containing the monthly input sequence.
                    X[1] is a tensor of shape (batch_size, input_size) containing the static input sequence.

    Outputs:
    out (torch.Tensor): A tensor of shape (batch_size, output_size) containing the output logits.

    """

    def __init__(
        self,
        input_size=10,
        hidden_size_lstm=128,
        hidden_size_mlp=300,
        num_layers=1,
        output_size=4,
        dropout=0.2,
    ) -> None:
        super(CropLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size_lstm,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True
                            )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_mlp, hidden_size_mlp),
            nn.BatchNorm1d(hidden_size_mlp),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size_mlp, output_size),
        )
    def forward(self, X):
        output, _ = self.lstm(X[0])
        # extract only the last hidden
        output = output[:, -1, :]
        output = self.classifier(torch.cat((output, X[1]), dim=1))
        return output
    
class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, weights, num_samples, replacement=True):
        super().__init__(weights, num_samples, replacement=replacement)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                    size=self.num_samples,
                                    p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                    replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


class ConvLSTMCell(nn.Module):
    """
    Initialize ConvLSTM cell.

    Args:
    input_dim (int): Number of channels of input tensor.
    hidden_dim (int): Number of channels of hidden state.
    kernel_size (int): Size of the convolutional kernel.
    bias (bool): Whether to add the bias.
    """
    
    def __init__(self,
        input_dim,
        hidden_dim,
        kernel_size,
        bias):

        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2
        self.bias = bias

        self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                            out_channels=4 * self.hidden_dim,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            bias=self.bias)

    def forward(self, input, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, length):
        return (torch.zeros(batch_size, self.hidden_dim, length, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, length, device=self.conv.weight.device))


class CropConvLSTM(nn.Module):

    """
    A PyTorch module implementing a Crop Conv LSTM network.
    
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        n_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers

    Input:
        A tensor of size B, T, C
    Output:
        A tuple of two lists of length n_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: tuple,
        n_layers: int,
        n_classes: int,
        input_len_monthly: int,
        seq_len: int,
        input_len_static: int,
        bias: bool=True,
        return_all_layers: bool=False
        ) -> None:
        super(CropConvLSTM, self).__init__()

        
        
        assert (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))),'`kernel_size` must be tuple or list of tuples' 
        # self._check_kernel_size_consistency(kernel_size)
        
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == n_layers
        kernel_size = self._extend_for_multilayer(kernel_size, n_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, n_layers)
        assert len(kernel_size) == len(hidden_dim) == n_layers, 'Inconsistent list length.'

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.n_classes = n_classes
        self.input_len_monthly = input_len_monthly
        self.seq_len = seq_len
        self.input_len_static = input_len_static
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                        hidden_dim=self.hidden_dim[i],
                                        kernel_size=self.kernel_size[i],
                                        bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
        
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim[0]*self.seq_len*self.input_len_monthly+self.input_len_static, self.hidden_dim[0]*self.seq_len),
            nn.BatchNorm1d(self.hidden_dim[0]*self.seq_len),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim[0]*self.seq_len, self.seq_len),
            nn.BatchNorm1d(self.seq_len),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.seq_len, self.n_classes)
        )
        
    def forward(self, input, hidden_state=None):
        """
        Args:
        input (tuple):  input[0] is a tensor of shape (batch_size, sequence_length, input_size) containing the monthly input sequence.
                        input[1] is a tensor of shape (batch_size, input_size) containing the static input sequence.

        Returns
        -------
        last_state_list, layer_output
        """
        input_monthly = input[0][:, None, :, :]  #fictional dimension added. will be used as channnels
        input_static = input[1]
        b = input_monthly.size()[0]

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, length=self.input_len_monthly)

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_monthly

        for layer_idx in range(self.n_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(self.seq_len):
                h, c = self.cell_list[layer_idx](input = cur_layer_input[:, :, t, :],
                                                cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        output_monthly = self.flatten(layer_output_list[0])        
        output = self.net(torch.cat((output_monthly, input_static), dim=1))

        return output

    def _init_hidden(self, batch_size, length):
        init_states = []
        for i in range(self.n_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, length))
        return init_states

    # @staticmethod
    # def _check_kernel_size_consistency(kernel_size):
    #     if not (isinstance(kernel_size, tuple) or
    #             (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
    #         raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, n_layers):
        if not isinstance(param, list):
            param = [param] * n_layers
        return param


class CroplandDataModuleMLP(pl.LightningDataModule):
    """
    This module defines a LightningDataModule class for loading and 
        preparing data for a Cropland classification model using MLP architecture.
    Args:
    X (dict): A dictionary containing the input data for Train, Validation, and Test sets.
    y (dict): A dictionary containing the corresponding target values for Train, Validation, and Test sets.
    batch_size (int): The batch size to be used for training and evaluation. Default is 128.
    """

    def __init__(self, X: dict, y: dict, batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
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

        self.dl_dict = {"batch_size": self.batch_size, "num_workers": self.num_workers}

    def prepare_data(self):
        # Calculate class weights for imbalanced dataset
        _, counts = torch.unique(self.y_train.argmax(dim=1), return_counts=True)
        class_weights = 1.0 / torch.sqrt(counts.float())
        loss_weights = class_weights / class_weights.sum()
        ds = self.y_train.argmax(dim=1)
        weights = [loss_weights[i] for i in ds]
        self.sampler = CustomWeightedRandomSampler(
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
            # shuffle=True,
            sampler=self.sampler,
            **self.dl_dict,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, **self.dl_dict)


class CropMLP(nn.Module):
    """
    A multi-layer perceptron (MLP) used for crop classification.
    Args:
        input_size (int): The number of input features (default: 164).
        output_size (int): The number of output logits (default: 4).
    Inputs:
        X (torch.Tensor): A tensor of shape (batch_size, input_size) containing input data.
    Returns:
        torch.Tensor: A tensor of shape (batch_size, output_size) containing the output logits.
    """

    def __init__(self, input_size=164, output_size=4) -> None:
        super(CropMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 8 * input_size),
            nn.BatchNorm1d(8 * input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(8 * input_size, 4 * input_size),
            nn.BatchNorm1d(4 * input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(4 * input_size, 2 * input_size),
            nn.BatchNorm1d(2 * input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(2 * input_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_size, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_size // 2, output_size),
        )

    def initialize_bias_weights(self, y_train):
        """
        Initialize the bias weights of the final linear layer based on the class distribution.

        Args:
            y_train (torch.Tensor): A tensor of shape (num_samples,) containing the target labels.
        """
        _, class_counts = torch.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        class_distribution = class_counts.float() / total_samples

        # Initialize bias weights for the final linear layer
        bias_weights = -torch.log(class_distribution)
        self.net[-1].bias.data = bias_weights

    def forward(self, X) -> torch.Tensor:
        output = self.net(X)
        return output


class CropPL(pl.LightningModule):
    """
    PyTorch Lightning module for training a crop classification neural network.

    Args:
        net (torch.nn.Module): the neural network module to be trained.
        num_classes (int): the number of classes in the dataset.
        lr (float): the learning rate for optimization.
        weight_decay (float): the weight decay for optimization.
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
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        
        # for tracking best so far validation f1score
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
            task="multiclass", num_classes=num_classes, average="macro")
        self.val_avg_precision = torchmetrics.AveragePrecision(
            task="multiclass", num_classes=num_classes,  average="macro")
        self.test_avg_precision = torchmetrics.AveragePrecision(
            task="multiclass", num_classes=num_classes, average="macro")

        self.train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro")

        self.train_F1Score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro")
        self.val_F1Score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro")
        self.test_F1Score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.net(x)
    
    def loss(self, y_hat, y):
        return self.criterion(y_hat, y)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        self.val_loss.reset()
        self.val_accuracy.reset()
        self.val_avg_precision.reset()
        self.val_precision.reset() 
        self.val_recall.reset()
        self.val_F1Score.reset()
        self.val_F1Score_best.reset()
        

    def model_step(self, batch):
        features, ohe_targets = batch
        logits = self.forward(features)
        loss = self.loss(logits, ohe_targets.float())
        preds = self.softmax(logits)
        return loss, preds, ohe_targets.argmax(dim=1)

    def training_step(self, batch, batch_idx):
        loss, preds, target = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        # To account for Dropout behavior during evaluation
        self.net.eval()
        with torch.no_grad():
            _, preds, target = self.model_step(batch)
        self.train_accuracy(preds, target)
        self.train_recall(preds, target)
        self.train_precision(preds, target)
        self.train_F1Score(preds, target)
        self.train_avg_precision(preds, target)

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
        self.net.train()
        return {"loss": loss, "preds": preds, "target": target}
    
    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, target = self.model_step(batch)

        self.val_loss(loss)
        self.val_accuracy(preds, target)
        self.val_recall(preds, target)    
        self.val_precision(preds, target)
        self.val_F1Score(preds, target)
        self.val_avg_precision(preds, target)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log(
            "val/F1Score", self.val_F1Score, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/AP", self.val_avg_precision, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "target": target}
    
    def on_validation_epoch_end(self): 
        f1sc = self.val_F1Score.compute() # get current val f1score
        self.val_F1Score_best(f1sc) # update best so far val f1score
        self.log("val/F1Score_best", self.val_F1Score_best.compute(), prog_bar=False) # sync_dist=True

    def test_step(self, batch, batch_idx):
        loss, preds, target = self.model_step(batch)

        self.test_loss(loss)
        self.test_accuracy(preds, target)
        self.test_recall(preds, target)
        self.test_precision(preds, target)
        self.test_F1Score(preds, target)
        self.test_avg_precision(preds, target)

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

        return {"loss": loss, "preds": preds, "target": target}
    
    def on_test_epoch_end(self) -> None:
        pass
    
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
                    "interval": "epoch",
                },
            }
        return {"optimizer": optimizer}
