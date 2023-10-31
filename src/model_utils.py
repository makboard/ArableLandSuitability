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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import List, Dict, Optional, Union, Any, Tuple


def roc_auc_score_multiclass(
    actual_class: List[int], prob: List[List[float]], le: Optional[LabelEncoder] = None
) -> Dict[Union[int, str], float]:
    """Calculate OvR roc_auc score for multiclass case

    Args:
        actual_class (List[int]): array of actual classes
        prob (List[List[float]]): class probabilities for each sample
        le (LabelEncoder, optional): LabelEncoder for classes if needed. Defaults to None.

    Returns:
        Dict[Union[int, str], float]: key -- class label; value -- OvR roc_auc score
    """
    # creating a set of all the unique classes using the actual class list
    unique_class = list(set(actual_class))
    roc_auc_dict: Dict[Union[int, str], float] = {}

    for per_class in unique_class:
        # marking the current class as 1 and all other classes as 0
        new_actual_class = [1 if x == per_class else 0 for x in actual_class]
        new_prob = prob[:, unique_class.index(per_class)]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_prob)

        if le is None:
            roc_auc_dict[per_class] = roc_auc
        else:
            roc_auc_dict[le.inverse_transform([per_class])[0]] = roc_auc

    return roc_auc_dict


def downsample(
    X: Union[List[List[Union[int, float]]], np.ndarray],
    y: Union[List[Union[int, float]], np.ndarray],
    oversampling: bool = False,
) -> Tuple[
    Union[List[List[Union[int, float]]], np.ndarray],
    Union[List[Union[int, float]], np.ndarray],
]:
    """Downsample the most frequent class to the second frequent one
    and optionally oversample the rest classes.
    Downsampling is performed by RandomUnderSampler.
    Oversampling is performed by SMOTE.

    Args:
        X (Union[List[List[Union[int, float]]], np.ndarray]): Initial features.
        y (Union[List[Union[int, float]], np.ndarray]): Initial target variable.
        oversampling (bool, optional): If True, oversampling is performed
        for the rest classes. Defaults to False.

    Returns:
        Tuple[Union[List[List[Union[int, float]]], np.ndarray], Union[List[Union[int, float]], np.ndarray]]:
        Downsampled dataset.
    """

    # Define classes distribution
    counter = Counter(y)
    print("Initial data:")
    print(counter.most_common())

    # Define over- and undersampling strategies
    over = SMOTE(random_state=42, n_jobs=-1)
    most_common_class, most_common_count = counter.most_common()[0]
    second_most_common_count = counter.most_common()[1][1]

    under = RandomUnderSampler(
        sampling_strategy={most_common_class: second_most_common_count}, random_state=42
    )

    # Define pipeline steps
    steps = [("u", under)]
    if oversampling:
        steps.append(("o", over))
    pipeline = Pipeline(steps=steps)

    # transform the dataset
    X, y = pipeline.fit_resample(X, y)

    # Print final distribution
    counter = Counter(y)
    print("Resampled data:")
    print(counter.most_common())

    return X, y


def get_feature_lists(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Get lists of monthly and static features.

    Args:
        X (pd.DataFrame): Initial features

    Returns:
        Tuple[List[str], List[str]]: Lists of monthly and static features
    """
    static_keywords = ["DEM", "morf", "12m_SPI"]

    list_of_monthly_features = [
        x for x in X.columns if not any(keyword in x for keyword in static_keywords)
    ]
    list_of_static_features = [
        x for x in X.columns if any(keyword in x for keyword in static_keywords)
    ]

    return list_of_monthly_features, list_of_static_features


def get_feature_lists_alt(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Get lists of monthly and static features via "M" at the end of feature name.

    Args:
        X (pd.DataFrame): Initial features

    Returns:
        Tuple[List[str], List[str]]: Lists of monthly and static features
    """
    # static_keywords = ["DEM", "morf", "12m_SPI"]

    list_of_monthly_features = [v for v in X.columns if ("M" in v.split("_")[-1])]
    list_of_static_features = [v for v in X.columns if not ("M" in v.split("_")[-1])]

    return list_of_monthly_features, list_of_static_features


def reshape_data(X: pd.DataFrame) -> np.ndarray:
    """Prepare dataset in shape (N, num_of_monthly + num_of_static, 12)
    N - number of samples
    num_of_monthly - number of monthly features
    num_of_static - number of static features
    12 - number of months

    Args:
        X (pd.DataFrame): Initial features

    Returns:
        np.ndarray: Modified dataset
    """
    list_of_monthly_features, list_of_static_features = get_feature_lists_alt(X)

    X_monthly = X[list_of_monthly_features]
    X_static = X[list_of_static_features].to_numpy()

    num_samples = X_monthly.shape[0]
    num_monthly = len(list_of_monthly_features) // 12
    num_static = len(list_of_static_features)

    X_monthly_reshaped = X_monthly.to_numpy().reshape(num_samples, 12, num_monthly)

    # Create an empty output ndarray and fill it
    X_new = np.empty((num_samples, 12, num_monthly + num_static))
    X_new[:, :, :num_monthly] = X_monthly_reshaped
    X_new[:, :, num_monthly:] = X_static[:, np.newaxis, :]

    return X_new


def calculate_tpr_fpr(
    y_real: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]
) -> Tuple[float, float]:
    """
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations.

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    """

    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    if cm.shape != (2, 2):  # Ensure binary classification
        raise ValueError("This function is designed for binary classification only.")

    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Avoiding zero division
    denominator_tpr = TP + FN
    denominator_fpr = TN + FP

    tpr = TP / denominator_tpr if denominator_tpr != 0 else 0.0
    fpr = 1 - (TN / denominator_fpr) if denominator_fpr != 0 else 0.0

    return tpr, fpr


def get_all_roc_coordinates(
    y_real: Union[List[int], np.ndarray], y_proba: Union[List[float], np.ndarray]
) -> Tuple[List[float], List[float]]:
    """
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the prediction of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for the positive class, usually obtained using the `.predict_proba()` method.

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    """

    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        # Assuming binary classification and probabilities are in the second column
        y_proba = y_proba[:, 1]

    fpr_list, tpr_list, _ = roc_curve(y_real, y_proba)

    return list(tpr_list), list(fpr_list)


def plot_precision_recall_curve(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    """Plot the precision-recall curve for each class and return average AP over all classes."""
    avg_precisions = []

    for i, class_ in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_proba[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_proba[:, i])
        avg_precisions.append(avg_precision)

        plt.plot(
            recall, precision, lw=2, label=f"class {class_} AP={avg_precision:.2f}"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision vs. Recall curve")
    plt.show()

    mean_avg_precision = np.mean(avg_precisions)
    print(f"Average AP over all classes: {mean_avg_precision:.4f}")

    return mean_avg_precision


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> None:
    """Plot the ROC curve for each class."""
    for i, class_ in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_proba[:, i])

        plt.plot(fpr, tpr, lw=2, label=f"class {class_}")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.title("ROC Curve")
    plt.show()


def plot_prob_distribution(
    y_test: np.ndarray, y_proba: np.ndarray, classes: np.ndarray, type_: str = "ovr"
) -> Tuple[float, dict]:
    """Plot probability distributions and return AUC scores."""
    roc_auc = {}
    total_auc = 0
    plt.figure(figsize=(20, 7))

    if type_ == "ovr":
        bins = np.linspace(0, 1, 31)
        for i, class_ in enumerate(classes):
            df_aux = pd.DataFrame(
                {
                    "class": [1 if y == class_ else 0 for y in y_test],
                    "prob": y_proba[:, i],
                }
            )
            plt.subplot(1, len(classes), i + 1)
            sns.histplot(data=df_aux, x="prob", hue="class", bins=bins)
            plt.title(class_)
            roc_auc[class_] = roc_auc_score(df_aux["class"], df_aux["prob"])
    elif type_ == "ovo":
        bins = np.linspace(0, 1, 21)
        combinations = [
            (i, j) for i in range(len(classes)) for j in range(i + 1, len(classes))
        ]

        for index, (i, j) in enumerate(combinations):
            df_aux = pd.DataFrame()
            df_aux["class"] = y_test
            df_aux["prob"] = y_proba[:, i]

            # Filter the dataframe to only contain instances of the two classes being considered
            df_aux = df_aux[
                (df_aux["class"] == classes[i]) | (df_aux["class"] == classes[j])
            ]

            # Convert the target to binary: 1 for the current class (classes[i]) and 0 for the other class (classes[j])
            df_aux["class"] = [1 if y == classes[i] else 0 for y in df_aux["class"]]

            # Check if there are more than 1 unique class after binarization
            if len(df_aux["class"].unique()) <= 1:
                continue
            plt.subplot(2, len(combinations) // 2, index + 1)
            sns.histplot(data=df_aux, x="prob", hue="class", bins=bins, legend=False)
            plt.title(f"{classes[i]} vs {classes[j]}")
            roc_auc[f"{classes[i]} vs {classes[j]}"] = roc_auc_score(
                df_aux["class"], df_aux["prob"]
            )

    for key, value in roc_auc.items():
        print(f"{key} ROC AUC {type_.upper()}: {value:.4f}")
        total_auc += value

    avg_auc = total_auc / len(roc_auc)
    print(f"Average ROC AUC {type_.upper()}: {avg_auc:.4f}")

    plt.tight_layout()
    plt.show()

    return avg_auc, roc_auc


def custom_multiclass_report(
    y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> None:
    """Display various multiclass classification metrics and plots."""
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    print(classification_report(y_test, y_pred))

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        fmt=".2f",
        linewidths=1,
        cmap="Blues",
        annot_kws={"size": 12, "weight": "bold"},
    )

    # Adjusting color of annotations based on background
    for t in ax.texts:
        t.set_text(t.get_text() + " %")
        if float(t.get_text().strip(" %")) > 30:
            t.set_color("white")
        else:
            t.set_color("black")

    plt.title("Normalized Confusion Matrix", size=15)
    plt.xlabel("Predicted class", size=13)
    plt.ylabel("Actual class", size=13)
    plt.show()

    plot_precision_recall_curve(y_test_bin, y_proba, classes)
    plot_roc_curve(y_test_bin, y_proba, classes)

    plot_prob_distribution(y_test, y_proba, classes, "ovr")
    plot_prob_distribution(y_test, y_proba, classes, "ovo")


class ConvLSTMCell(nn.Module):
    """
    A ConvLSTM cell which uses convolutional gates for input transformations, suitable for
    sequences of image-like data with spatial correlations.

    Args:
    - input_dim (int): Number of channels of input tensor.
    - hidden_dim (int): Number of channels of hidden state.
    - kernel_size (int or tuple): Size of the convolutional kernel.
    - bias (bool): Whether to add the bias.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int]],
        bias: bool = True,
    ) -> None:
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Ensure kernel_size is a tuple
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )

        self.padding = self.kernel_size[0] // 2
        self.bias = bias

        self.conv = nn.Conv1d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(
        self, input: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConvLSTM cell.

        Args:
        - input (torch.Tensor): The input tensor for the current timestep.
        - cur_state (Tuple[torch.Tensor, torch.Tensor]): Tuple containing the current hidden state
          and cell state.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: The next hidden and cell states.
        """
        h_cur, c_cur = cur_state
        combined = torch.cat([input, h_cur], dim=1)  # concatenate along channel axis

        # Apply convolution
        combined_conv = self.conv(combined)

        # Split the convolutional output into four parts for gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Calculate gate values
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Compute next cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self, batch_size: int, length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the hidden and cell states to zeros.

        Args:
        - batch_size (int): The batch size.
        - length (int): The sequence length.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Initialized hidden and cell states.
        """
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, length, device=device),
            torch.zeros(batch_size, self.hidden_dim, length, device=device),
        )


class CropConvLSTM(nn.Module):

    """
    A PyTorch module implementing a Crop Conv LSTM network.

    Attributes:
    -----------
    input_dim : int
        Number of channels in the input.
    hidden_dim : Union[int, List[int]]
        Number of hidden channels.
    kernel_size : Union[Tuple[int], List[Tuple[int]]]
        Size of the convolutional kernel.
    n_layers : int
        Number of LSTM layers.
    n_classes : int
        Number of output classes.
    input_len_monthly : int
        Length of the monthly input sequence.
    seq_len : int
        Sequence length.
    bias : bool
        If True, adds a bias term; Otherwise no bias term.
    return_all_layers : bool
        If True, returns all LSTM layers; Otherwise only the last layer.
    cell_list : nn.ModuleList
        List containing ConvLSTMCell modules.
    flatten : nn.Flatten
        Flatten layer.
    net : nn.Sequential
        Sequential network for processing LSTM outputs.

    Input:
    --------
        A tensor of size B, T, C
    Output:
    --------
        A tuple of two lists of length n_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: Union[int, List[int]] = 16,
        kernel_size: Union[Tuple[int], List[Tuple[int]]] = (3,),
        n_layers: int = 1,
        n_classes: int = 4,
        input_len_monthly: int = 52,
        seq_len: int = 12,
        bias: bool = False,
        return_all_layers: bool = False,
    ) -> None:
        """
        Initializes the CropConvLSTM module.
        """
        super(CropConvLSTM, self).__init__()
        self._validate_kernel_size(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, n_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, n_layers)
        assert (
            len(kernel_size) == len(hidden_dim) == n_layers
        ), "Inconsistent list length."

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.n_classes = n_classes
        self.input_len_monthly = input_len_monthly
        self.seq_len = seq_len

        self.cell_list = nn.ModuleList(
            [
                ConvLSTMCell(
                    input_dim=self.input_dim if i == 0 else self.hidden_dim[i - 1],
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
                for i in range(self.n_layers)
            ]
        )

        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(
                self.hidden_dim[0] * self.seq_len * self.input_len_monthly,
                self.hidden_dim[0] * self.seq_len,
            ),
            nn.BatchNorm1d(self.hidden_dim[0] * self.seq_len),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim[0] * self.seq_len, self.seq_len),
            nn.BatchNorm1d(self.seq_len),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.seq_len, self.n_classes),
        )

    def forward(
        self, input_monthly: torch.Tensor, hidden_state: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the CropConvLSTM module.

        Parameters:
        -----------
        input_monthly : torch.Tensor
            Tensor of shape (batch_size, sequence_length, input_size) containing the monthly input sequence.
        hidden_state : torch.Tensor, optional
            Initial hidden state for the LSTM cells.

        Returns:
        --------
        torch.Tensor
            Model's output tensor.
        """
        input_monthly = input_monthly[:, None, :, :]
        b = input_monthly.size()[0]

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(
                batch_size=b, length=self.input_len_monthly
            )

        layer_output_list, last_state_list, cur_layer_input = [], [], input_monthly

        for layer_idx in range(self.n_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(self.seq_len):
                h, c = self.cell_list[layer_idx](
                    input=cur_layer_input[:, :, t, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        output_monthly = self.flatten(layer_output_list[0])
        output = self.net(output_monthly)

        return output

    def _init_hidden(
        self, batch_size: int, length: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [cell.init_hidden(batch_size, length) for cell in self.cell_list]

    @staticmethod
    def _validate_kernel_size(kernel_size: Union[Tuple[int], List[Tuple[int]]]) -> None:
        if not (
            isinstance(kernel_size, tuple)
            or all(isinstance(elem, tuple) for elem in kernel_size)
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(
        param: Union[int, List[int]], n_layers: int
    ) -> List[int]:
        return [param] * n_layers if not isinstance(param, list) else param


class CropTransformer(nn.Module):
    """
    Crop Transformer classifier.

    This module applies the Transformer encoder on input sequences and then
    uses a feed-forward neural network for classification.

    Args:
        d_model (int): Number of expected features in the input. Default: 52.
        nhead (int): Number of heads in the multi-head attention mechanism. Default: 4.
        dim_feedforward (int): Dimension of the feedforward network. Default: 256.
        hidden_size (int): Number of features in the hidden state. Default: 128.
        num_layers (int): Number of Transformer encoder layers. Default: 2.
        dropout (float): Dropout rate. Default: 0.2.
        activation (str): Activation function for feed-forward networks ("relu" or "gelu"). Default: "relu".
        output_size (int): Number of output logits. Default: 4.

    Inputs:
        X (torch.Tensor): Tensor of shape (batch_size, sequence_length, d_model)
                          representing the input sequence.

    Outputs:
        out (torch.Tensor): Tensor of shape (batch_size, output_size)
                            representing the output logits.
    """

    def __init__(
        self,
        d_model: int = 52,
        nhead: int = 4,
        dim_feedforward: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: str = "relu",
        output_size: int = 4,
    ) -> None:
        super(CropTransformer, self).__init__()

        # Define the Transformer encoder
        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, X):
        """Forward pass for Crop Transformer."""
        # Apply Transformer encoder
        encoded = self.transformer_enc(X)

        # Use encoding of the last time step for classification
        output = encoded[:, -1, :]
        output = self.classifier(output)
        return output


class CropLSTM(nn.Module):
    """
    Crop LSTM network.

    This module applies a multi-layer LSTM on input sequences and then
    uses a feed-forward neural network for classification.

    Args:
        input_size (int): Number of expected features in the input. Default: 52.
        hidden_size (int): Number of features in the hidden state. Default: 104.
        num_layers (int): Number of recurrent LSTM layers. Default: 4.
        output_size (int): Number of output logits. Default: 4.

    Inputs:
        X (torch.Tensor): Tensor of shape (batch_size, sequence_length, input_size)
                          representing the input sequence.

    Outputs:
        out (torch.Tensor): Tensor of shape (batch_size, output_size)
                            representing the output logits.
    """

    def __init__(
        self,
        input_size: int = 52,
        hidden_size: int = 104,
        num_layers: int = 4,
        output_size: int = 4,
    ) -> None:
        super(CropLSTM, self).__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # LSTM weights initialization
        self._initialize_lstm_weights()

        # Define the classifier layers
        self.linearLayer1 = nn.Linear(hidden_size, hidden_size)
        self.linearLayer2 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()

    def forward(self, X):
        out, _ = self.lstm(X)
        out = out[:, -1, :]
        out = self.act(self.linearLayer1(out))
        out = self.linearLayer2(out)
        return out

    def _initialize_lstm_weights(self):
        """Initialize LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)


class CroplandDataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading and preparing data for a Cropland classification models.

    Args:
        X (dict): A dictionary containing the input data for Train, Validation, and Test sets.
        y (dict): A dictionary containing the corresponding target values for Train, Validation, and Test sets.
        batch_size (int): The batch size to be used for training and evaluation. Default is 128.
    """

    def __init__(self, X: dict, y: dict, batch_size: int = 128) -> None:
        super().__init__()
        self.batch_size = batch_size

        # Convert input data to tensors
        self.X_train = torch.FloatTensor(X["Train"])
        self.X_val = torch.FloatTensor(X["Val"])
        self.X_test = torch.FloatTensor(X["Test"])

        # Convert target labels to tensors
        self.y_train = torch.LongTensor(y["Train"])
        self.y_val = torch.LongTensor(y["Val"])
        self.y_test = torch.LongTensor(y["Test"])

        self.dl_dict = {"batch_size": self.batch_size}

    def prepare_data(self) -> None:
        """
        Calculate class weights for imbalanced dataset and initialize sampler.
        """
        _, counts = torch.unique(self.y_train.argmax(dim=1), return_counts=True)
        class_weights = 1.0 / torch.sqrt(counts.float())
        loss_weights = class_weights / class_weights.sum()
        ds = self.y_train.argmax(dim=1)
        weights = [loss_weights[i] for i in ds]
        self.sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

    def setup(self, stage: str = None) -> None:
        """
        Initialize datasets for train, validation, and test phases.
        """
        if stage == "fit" or stage is None:
            self.dataset_train = TensorDataset(self.X_train, self.y_train)
            self.dataset_val = TensorDataset(self.X_val, self.y_val)

        if stage == "test" or stage is None:
            self.dataset_test = TensorDataset(self.X_test, self.y_test)

    def train_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: DataLoader object for training data.
        """
        return DataLoader(self.dataset_train, sampler=self.sampler, **self.dl_dict)

    def val_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: DataLoader object for validation data.
        """
        return DataLoader(self.dataset_val, **self.dl_dict)

    def test_dataloader(self) -> DataLoader:
        """
        Returns:
            DataLoader: DataLoader object for testing data.
        """
        return DataLoader(self.dataset_test, **self.dl_dict)


class CropMLP(nn.Module):
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

    def __init__(self, input_size: int = 162, output_size: int = 4) -> None:
        super(CropMLP, self).__init__()

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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


class CropPL(pl.LightningModule):
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
        num_classes: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        self.val_F1Score_best = torchmetrics.MaxMetric()

        metric_args = {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": "macro",
        }

        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self.train_precision = torchmetrics.Precision(**metric_args)
        self.val_precision = torchmetrics.Precision(**metric_args)
        self.test_precision = torchmetrics.Precision(**metric_args)

        self.train_recall = torchmetrics.Recall(**metric_args)
        self.val_recall = torchmetrics.Recall(**metric_args)
        self.test_recall = torchmetrics.Recall(**metric_args)

        self.train_F1Score = torchmetrics.F1Score(**metric_args)
        self.val_F1Score = torchmetrics.F1Score(**metric_args)
        self.test_F1Score = torchmetrics.F1Score(**metric_args)

        self.train_avg_precision = torchmetrics.AveragePrecision(**metric_args)
        self.val_avg_precision = torchmetrics.AveragePrecision(**metric_args)
        self.test_avg_precision = torchmetrics.AveragePrecision(**metric_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(y_hat, y)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        self.val_F1Score_best.reset()

    def _model_step(self, batch: tuple) -> tuple:
        x, y = batch
        y_hat = self(x)
        loss = self._compute_loss(y_hat, y.float())
        return loss, self.softmax(y_hat), torch.argmax(y, dim=1)

    def _log_metrics(
        self, phase: str, loss: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ) -> None:
        getattr(self, f"{phase}_loss")(loss)
        getattr(self, f"{phase}_accuracy")(y_hat, y)
        getattr(self, f"{phase}_precision")(y_hat, y)
        getattr(self, f"{phase}_recall")(y_hat, y)
        getattr(self, f"{phase}_F1Score")(y_hat, y)
        getattr(self, f"{phase}_avg_precision")(y_hat, y)

        self.log(f"{phase}/loss", getattr(self, f"{phase}_loss"))
        self.log(f"{phase}/accuracy", getattr(self, f"{phase}_accuracy"))
        self.log(f"{phase}/precision", getattr(self, f"{phase}_precision"))
        self.log(f"{phase}/recall", getattr(self, f"{phase}_recall"))
        self.log(f"{phase}/F1Score", getattr(self, f"{phase}_F1Score"))
        self.log(f"{phase}/AP", getattr(self, f"{phase}_avg_precision"))

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        loss, y_hat, y = self._model_step(batch)
        self._log_metrics("train", loss, y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        loss, y_hat, y = self._model_step(batch)
        self._log_metrics("val", loss, y_hat, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        F1Score = self.val_F1Score.compute()
        self.val_F1Score_best(F1Score)
        self.log("val/F1Score_best", self.val_F1Score_best.compute(), prog_bar=True)

    def test_step(self, batch: tuple, batch_idx: int) -> dict:
        loss, y_hat, y = self._model_step(batch)
        self._log_metrics("test", loss, y_hat, y)
        return {"test_loss": loss}

    def configure_optimizers(self) -> dict[str, Any] | dict[str, torch.optim.Adam]:
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=3e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        if scheduler is not None:
            scheduler = scheduler(
                optimizer=optimizer,
                patience=25,
                mode="min",
                factor=0.1,
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

class CroplandDataModuleMLP(pl.LightningDataModule):
    """
    This module defines a LightningDataModule class for loading and
        preparing data for a Cropland classification model using MLP architecture.
    Args:
    X (dict): A dictionary containing the input data for Train, Validation, and Test sets.
    y (dict): A dictionary containing the corresponding target values for Train, Validation, and Test sets.
    batch_size (int): The batch size to be used for training and evaluation. Default is 128.
    """

    def __init__(
        self,
        X: dict,
        y: dict,
        batch_size: int = 128,
        num_workers: int = 4,
        num_classes: int = 4,
    ):
        super().__init__()
        assert (num_classes == 4) or (
            num_classes == 2
        ), "Only 4 or 2 classes are supported"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train, self.X_val, self.X_test = (
            torch.FloatTensor(X["Train"]),
            torch.FloatTensor(X["Val"]),
            torch.FloatTensor(X["Test"]),
        )
        if num_classes == 4:
            self.y_train, self.y_val, self.y_test = (
                torch.LongTensor(y["Train"]),
                torch.LongTensor(y["Val"]),
                torch.LongTensor(y["Test"]),
            )
        elif num_classes == 2:
            self.y_train, self.y_val, self.y_test = (
                torch.ShortTensor(
                    np.stack(
                        (y["Train"][:, 0], y["Train"][:, 1:].sum(axis=-1)), axis=-1
                    )
                ),
                torch.ShortTensor(
                    np.stack((y["Val"][:, 0], y["Val"][:, 1:].sum(axis=-1)), axis=-1)
                ),
                torch.ShortTensor(
                    np.stack((y["Test"][:, 0], y["Test"][:, 1:].sum(axis=-1)), axis=-1)
                ),
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
            self.dataset_train = CroplandDataset(
                (self.X_monthly_train, self.X_static_train), self.y_train
            )
            self.dataset_val = CroplandDataset(
                (self.X_monthly_val, self.X_static_val), self.y_val
            )

        if stage == "test" or stage is None:
            self.dataset_test = CroplandDataset(
                (self.X_monthly_test, self.X_static_test), self.y_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            #   shuffle=True,
            pin_memory=True,
            sampler=self.sampler,
            **self.dl_dict,
        )

    def val_dataloader(self):
        return DataLoader(self.dataset_val, pin_memory=True, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, pin_memory=True, **self.dl_dict)


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, weights, num_samples, replacement=True):
        super().__init__(weights, num_samples, replacement=replacement)

    def __iter__(self):
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=self.weights.numpy() / torch.sum(self.weights).numpy(),
            replace=self.replacement,
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())