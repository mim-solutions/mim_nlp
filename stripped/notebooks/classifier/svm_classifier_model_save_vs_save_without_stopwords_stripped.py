import os

from datasets import load_dataset

while "notebooks" in os.getcwd():
    os.chdir("..")

from numpy import array_equal

from mim_nlp.classifier.svm import SVMClassifier
from mim_nlp.general_utils import get_size_in_megabytes
# ---
"""
# Training the model
"""
dataset = load_dataset("imdb")
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
model.fit(dataset["train"]["text"], dataset["train"]["label"])
# ---
"""
# Compare vocabulary size
"""
len(model.pipeline[0].vocabulary_)
# ---
len(model.pipeline[0].stop_words_)
# ---
"""
# Save the full model
"""
%%time
model.save("models/svm")
# ---
get_size_in_megabytes("models/svm")
# ---
%%time
model.save_without_stop_words("models/svm_small")
# ---
get_size_in_megabytes("models/svm_small")
# ---
"""
# Prediction check
"""
model = SVMClassifier.load("models/svm")
predictions_full = model.predict_scores(dataset["test"]["text"])
# ---
model = SVMClassifier.load("models/svm_small")
predictions_small = model.predict_scores(dataset["test"]["text"])
# ---
assert array_equal(predictions_full, predictions_small)
# ---
