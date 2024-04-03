%load_ext autoreload
%autoreload 2

import os

while "notebooks" in os.getcwd():
    os.chdir("..")

from datasets import load_dataset
from numpy import array_equal
from sklearn.metrics import mean_squared_error

from mim_nlp.regressor import AutoRegressor
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
model = AutoRegressor(
    batch_size=4,
    learning_rate=1e-3,
    epochs=4,
    pretrained_model_name_or_path="bert-base-uncased",
    input_size=512,
    device="cuda",
    many_gpus=True,
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
model.save("models/auto_regressor")
# ---
"""
# Loading the model
"""
model_loaded = AutoRegressor.load("models/auto_regressor")
# ---
predictions_from_loaded = model.predict(dataset["test"]["text"])
# ---
assert array_equal(predictions, predictions_from_loaded)
# ---
