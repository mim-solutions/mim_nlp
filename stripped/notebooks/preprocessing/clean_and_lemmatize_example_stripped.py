import os

from datasets import load_dataset
from functools import partial

while "notebooks" in os.getcwd():
    os.chdir("..")

import gensim.parsing.preprocessing as gsp

from mim_nlp.preprocessing import (
    lemmatize,
    process_emojis,
    remove_urls,
    strip_multiple_emojis,
    strip_short_words,
    TextCleaner,
    token_usernames,
)
# ---
"""
# Loading an open-source dataset
"""
dataset = load_dataset("allegro/summarization-polish-summaries-corpus")
# ---
texts = dataset["train"]["target"]
# ---
len(texts)
# ---
"""
# Define the preprocessing pipeline
"""
def lowercase(x: str) -> str:
    return x.lower()
# ---
pipeline = [
    lowercase,
    token_usernames,
    gsp.strip_tags,
    remove_urls,
    process_emojis,
    gsp.strip_punctuation,
    gsp.strip_numeric,
    gsp.strip_multiple_whitespaces,
    partial(strip_short_words, minsize=3),
    strip_multiple_emojis,
    lemmatize,
]
# ---
text_cleaner = TextCleaner(pipeline=pipeline)
# ---
"""
# Set the sample size
"""
N = 10000
texts_sample = texts[:N]
# ---
"""
# Run without multiprocessing
"""
%%time
clean_texts = text_cleaner.clean_texts(texts_sample, multiprocessing=False)
# ---
"""
# Run with multiprocessing
"""
%%time
clean_texts = text_cleaner.clean_texts(texts_sample, multiprocessing=True)
# ---
clean_texts[:5]
# ---
