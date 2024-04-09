import os

import pandas as pd
from datasets import load_dataset

while "notebooks" in os.getcwd():
    os.chdir("..")

from mim_nlp.preprocessing import Deduplicator
# ---
"""
# Loading an open-source dataset
"""
dataset = load_dataset("tweets_hate_speech_detection")
# ---
dataset
# ---
dataset["train"]["tweet"][0]
# ---
texts = dataset["train"]["tweet"]
# ---
"""
# Load Deduplicator
"""
deduplicator = Deduplicator()
# ---
"""
# Clusterize partial duplicates
"""
N = 1000
texts_sample = texts[:N]
# ---
%%time
clusters = deduplicator.clusterize_partial_duplicates(texts_sample)
# ---
df = pd.DataFrame({"text": texts_sample, "cluster": clusters})
# ---
df.value_counts(["cluster"])
# ---
df[df["cluster"] == 35].head()
# ---
"""
# Eliminate partial duplicates
"""
deduplicator.eliminate_partial_duplicates(df, "text")
# ---
