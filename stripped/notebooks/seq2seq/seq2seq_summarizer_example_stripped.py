import os

import torch
from datasets import load_dataset
from transformers import TrainingArguments

while "notebooks" in os.getcwd():
    os.chdir("..")

from mim_nlp.seq2seq import AutoSummarizer
# ---
"""
# Choose GPU
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.is_available()
# ---
"""
# Loading an open-source summary dataset
"""
dataset = load_dataset("allegro/summarization-polish-summaries-corpus")
# ---
dataset
# ---
dataset["train"]["target"][0]
# ---
len(dataset["train"]["source"])
# ---
"""
# Training a model on the dataset sample
"""
training_arguments = TrainingArguments(
    output_dir="tmp", num_train_epochs=1, logging_steps=100, per_device_train_batch_size=2, optim="adamw_torch"
)
# ---
# to use this model, install protobuf==3.20 and sentencepiece
model = AutoSummarizer(
    pretrained_model_name_or_path="allegro/plt5-base", max_length=1024, training_args=training_arguments
)
# ---
N_SAMPLES = 1000
model.fit(dataset["train"]["source"][:N_SAMPLES], dataset["train"]["target"][:N_SAMPLES])
# ---
"""
# Predicting
"""
source = dataset["validation"]["source"][0]
target = dataset["validation"]["target"][0]
# ---
print(target)
# ---
model.predict([source], max_new_tokens=20)
# ---
"""
# Saving the model
"""
model.save("models/my_plt5_model")
# ---
"""
# Loading the model
"""
model = AutoSummarizer.load("models/my_plt5_model/")
# ---
model
# ---
model.predict([source], max_new_tokens=20)
# ---
