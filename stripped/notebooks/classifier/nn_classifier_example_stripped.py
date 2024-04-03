%load_ext autoreload
%autoreload 2

import os

while "notebooks" in os.getcwd():
    os.chdir("..")

from datasets import load_dataset
from numpy import array_equal
import torch
import torch.nn as nn
from torchmetrics import Precision
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mim_nlp.classifier.nn import NNClassifier
# ---
torch.__version__
# ---
torch.cuda.is_available()
# ---
"""
# Loading an open-source dataset
"""
dataset = load_dataset("imdb")
# ---
dataset
# ---
x_train, x_val, y_train, y_val = train_test_split(
    dataset["train"]["text"], dataset["train"]["label"], train_size=20000, random_state=0
)
# ---
"""
# Training the model on the dataset
"""
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x
# ---
def accuracy(y_pred, y_target):
    y_pred = y_pred > 0.5
    return torch.sum(y_target == y_pred) / len(y_target)
# ---
tokenizer = TfidfVectorizer(sublinear_tf=True, min_df=0.01, max_df=0.5, ngram_range=(1, 2))
tokenizer = tokenizer.fit(dataset["train"]["text"])
input_size = len(tokenizer.vocabulary_)
print(input_size)
# ---
MODEL_PARAMS = {
    "batch_size": 256,
    "epochs": 2,
    "optimizer_params": {"lr": 1e-4},
    "device": "cpu",
    "many_gpus": False,
}
BINARY_METRICS = {
    "train_metrics_dict": {
        "accuracy": accuracy,
    },
    "eval_metrics_dict": {
        "accuracy": accuracy,
    },
}
# ---
model = NNClassifier(**MODEL_PARAMS, input_size=input_size, neural_network=MLP(input_size), tokenizer=tokenizer)
model.fit(x_train, y_train)
# ---
"""
# Get predictions on the test set
"""
predictions = model.predict(dataset["test"]["text"])
# ---
"""
# Calculate the accuracy score
"""
accuracy_score(dataset["test"]["label"], predictions)
# ---
"""
# Saving the model
"""
model.save_without_stop_words("models/nn_classifier")
# ---
"""
# Loading the model
"""
model_loaded = NNClassifier.load("models/nn_classifier", device="cpu")
# ---
predictions_from_loaded = model.predict(dataset["test"]["text"])
# ---
assert array_equal(predictions, predictions_from_loaded)
# ---
model_loaded.test(
    dataset["test"]["text"],
    dataset["test"]["label"],
    test_metrics_dict={"acc": accuracy, "precision": Precision(task="binary")},
)
# ---
