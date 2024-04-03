import os

from datasets import load_dataset

while "notebooks" in os.getcwd():
    os.chdir("..")

from numpy import array_equal
from sklearn.metrics import accuracy_score

from mim_nlp.classifier.svm import SVMClassifier
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
model = SVMClassifier(
    tfidf_transformer_parameters={
        "sublinear_tf": True,
        "min_df": 5,
        "max_df": 0.5,
        "norm": "l2",
        "encoding": "latin-1",
        "ngram_range": (1, 2),
    },
    linear_svc_parameters={"C": 5, "fit_intercept": True},
)
# ---
model.fit(dataset["train"]["text"], dataset["train"]["label"])
# ---
"""
# Get predictions on the test set
"""
predictions = model.predict(dataset["test"]["text"])
# ---
predictions
# ---
"""
# Calculate the accuracy score
"""
accuracy_score(dataset["test"]["label"], predictions)
# ---
"""
# Explaining the result
"""
example_text = dataset["test"]["text"][0]
example_text
# ---
model.explain_text(example_text, 10)
# ---
model.get_top_words_from_text(example_text, 10)
# ---
"""
# Saving the model
"""
model.save_without_stop_words("models/svm")
# ---
"""
# Loading the model
"""
model_loaded = SVMClassifier.load("models/svm")
# ---
predictions_from_loaded = model.predict(dataset["test"]["text"])
# ---
assert array_equal(predictions, predictions_from_loaded)
# ---
