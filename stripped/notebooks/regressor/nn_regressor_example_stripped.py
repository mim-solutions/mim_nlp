%load_ext autoreload
%autoreload 2

import os

while "notebooks" in os.getcwd():
    os.chdir("..")

import torch
import torch.nn as nn
from datasets import load_dataset
from numpy import array_equal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error

from mim_nlp.regressor import NNRegressor
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
"""
# Training the model on the dataset
"""
tokenizer = TfidfVectorizer(sublinear_tf=True, min_df=0.01, max_df=0.5, ngram_range=(1, 3))
tokenizer = tokenizer.fit(dataset["train"]["text"])
# ---
input_size = len(tokenizer.vocabulary_)
print(input_size)
# ---
class MLP(nn.Module):
    def __init__(self):
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
model = NNRegressor(
    batch_size=4,
    learning_rate=1e-3,
    epochs=8,
    input_size=input_size,
    tokenizer=tokenizer,
    neural_network=MLP(),
    device="cuda:0",
    many_gpus=False,
)
# ---
model.fit(dataset["train"]["text"], dataset["train"]["label"])
# ---
"""
# Get predictions on the test set
"""
predictions = model.predict(dataset["test"]["text"])
# ---
"""
# Calculate the accuracy score
"""
mean_squared_error(dataset["test"]["label"], predictions)
# ---
"""
# Saving the model
"""
model.save_without_stop_words("models/nn_regressor")
# ---
"""
# Loading the model
"""
model_loaded = NNRegressor.load("models/nn_regressor")
# ---
predictions_from_loaded = model.predict(dataset["test"]["text"])
# ---
assert array_equal(predictions, predictions_from_loaded)
# ---
