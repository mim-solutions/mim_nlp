from functools import partial

import numpy as np
import torch.nn as nn
from numpy.testing import assert_almost_equal
from pytorch_lightning.callbacks import ModelSummary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from torch.optim import SGD, Adam
from torchmetrics import Accuracy, Recall
from transformers import PreTrainedTokenizerFast

from mim_nlp.classifier.nn import NNClassifier


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


class MLPMultiClass(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(64, 3)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


def metric_func_1(y_pred, y_target):
    return 1


def metric_func_2(y_pred, y_target, param):
    return 2


MODEL_PARAMS = {
    "batch_size": 2,
    "epochs": 4,
    "optimizer_params": {"lr": 1e-1},
    "device": "cpu",
    "many_gpus": False,
}
BINARY_METRICS = {
    "train_metrics_dict": {
        "accuracy": Accuracy(task="binary"),
        "recall": Recall(task="binary"),
        "metric_1": metric_func_1,
        "metric_2": partial(metric_func_2, param=2),
    },
    "eval_metrics_dict": {
        "metric_1": metric_func_1,
    },
}


def test_pre_trained_tokenizer(seed):
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]
    x_test = ["tomato", "chair"]

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.train_from_iterator(
        x_train,
        WordLevelTrainer(
            vocab_size=10000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], show_progress=True
        ),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    input_size = 512
    model = NNClassifier(**MODEL_PARAMS, input_size=input_size, neural_network=MLP(input_size), tokenizer=tokenizer)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    assert len(predictions) == len(x_test)


def test_tf_idf_tokenizer(seed):
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    x_test = ["tomato", "chair"]
    expected_classes = [0, 1]

    tf_idf = TfidfVectorizer()

    input_size = len(x_train)
    model = NNClassifier(**MODEL_PARAMS, input_size=input_size, neural_network=MLP(input_size), tokenizer=tf_idf)
    model.fit_tokenizer(x_train)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = model.predict_scores(x_test)

    assert len(predictions) == len(x_test)
    assert np.array_equal(predictions, expected_classes)
    assert scores[0] < 0.3
    assert scores[1] > 0.7


def test_pipeline_tokenizer(seed):
    x_train = [
        "this movie is very interesting",
        "this movie is great",
        "I really want to see this movie again",
        "this movie is not funny",
        "this movie is bad",
        "I never want to see this movie again",
    ]
    y_train = [1, 1, 1, 0, 0, 0]

    x_test = ["this movie is great", "this movie is very absorbing", "this movie is not good"]
    expected_classes = [1, 1, 0]
    expected_features = {"very", "interesting", "great", "really", "not", "funny", "bad", "never"}

    tf_idf = TfidfVectorizer()
    n_features = len(expected_features)
    selector = SelectKBest(f_classif, k=n_features)
    pipeline = Pipeline([("tf_idf", tf_idf), ("selector", selector)])

    model = NNClassifier(**MODEL_PARAMS, input_size=n_features, neural_network=MLP(n_features), tokenizer=pipeline)
    model.fit_tokenizer(x_train, y_train)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = model.predict_scores(x_test)

    assert set(pipeline.get_feature_names_out().tolist()) == expected_features
    assert len(predictions) == len(x_test)
    assert np.array_equal(predictions, expected_classes)
    assert scores[0] > 0.5
    assert scores[1] > 0.5
    assert scores[2] < 0.5


def test_prediction_order(seed):
    """Check if the order of predictions is preserved and scores are reproducible."""
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    x_test = ["carrot"] * 99 + ["chair"] * 1

    tf_idf = TfidfVectorizer()

    input_size = len(x_train)
    model = NNClassifier(
        **MODEL_PARAMS, **BINARY_METRICS, input_size=input_size, neural_network=MLP(input_size), tokenizer=tf_idf
    )
    model.fit_tokenizer(x_train, y_train)
    model.fit(x_train, y_train)

    expected_score_carrot = model.predict_scores(["carrot"])[0]
    expected_score_chair = model.predict_scores(["chair"])[0]

    # Test if only the last prediction is different.
    # This will fail if the predictions are shuffled.
    predicted_scores = model.predict_scores(x_test)
    for i in range(len(x_test) - 1):
        assert_almost_equal(predicted_scores[i], expected_score_carrot)
    assert_almost_equal(predicted_scores[-1], expected_score_chair)


def test_multiclass_classification(seed):
    x_train = (
        ["carrot", "cucumber", "tomato", "potato"]
        + ["table", "chair", "door", "floor"]
        + ["football", "basketball", "volleyball", "baseball"]
    )
    y_train = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    x_test = ["tomato", "chair", "volleyball", "potato"]
    expected_classes = [0, 1, 2, 0]

    tf_idf = TfidfVectorizer()

    input_size = len(x_train)
    model = NNClassifier(
        **MODEL_PARAMS,
        input_size=input_size,
        neural_network=MLPMultiClass(input_size),
        tokenizer=tf_idf,
        loss_function=nn.CrossEntropyLoss(),
    )
    model.fit(x_train, y_train, fit_tokenizer=True)

    predictions = model.predict(x_test)
    scores = model.predict_scores(x_test)

    assert len(predictions) == len(x_test)
    assert np.array_equal(predictions, expected_classes)
    assert np.allclose(np.sum(scores, axis=1), np.ones(len(x_test)))
    assert scores.shape == (4, 3)
    assert scores[0, 0] > 0.5
    assert scores[1, 1] > 0.5
    assert scores[2, 2] > 0.5
    assert scores[3, 0] > 0.5


def test_save_and_load_pre_trained_tokenizer(tmp_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    input_size = 512
    optimizer_kwargs = {"optimizer": SGD, "optimizer_params": {"lr": 1e-2, "momentum": 0.2}}
    model = NNClassifier(
        **dict(MODEL_PARAMS, **optimizer_kwargs),
        **BINARY_METRICS,
        input_size=input_size,
        neural_network=MLP(input_size),
        tokenizer=tokenizer,
        callbacks=ModelSummary(),
    )

    model.save(str(tmp_path))
    model_loaded = NNClassifier.load(str(tmp_path), device=MODEL_PARAMS["device"], many_gpus=MODEL_PARAMS["many_gpus"])

    assert model._get_model_params() == model_loaded._get_model_params()
    assert model.is_fitted == model_loaded.is_fitted
    assert model.nn_module.is_classification == model.nn_module.is_classification
    assert model.nn_module.is_multiclass == model.nn_module.is_multiclass
    assert model.nn_module.optimizer_class == optimizer_kwargs["optimizer"]

    # assert types to be more specific than 'isinstance()'
    assert type(model) == type(model_loaded)  # noqa: E721
    assert type(model.tokenizer) == type(model_loaded.tokenizer)  # noqa: E721
    assert type(model.callbacks) == type(model_loaded.callbacks)  # noqa: E721
    assert type(model.nn_module.neural_network) == type(model_loaded.nn_module.neural_network)  # noqa: E721
    assert type(model.nn_module.loss_fun) == type(model_loaded.nn_module.loss_fun)  # noqa: E721

    assert type(model.nn_module.train_metrics_module_dict) == type(  # noqa: E721
        model_loaded.nn_module.train_metrics_module_dict
    )
    assert type(model.nn_module.eval_metrics_module_dict) == type(  # noqa: E721
        model_loaded.nn_module.eval_metrics_module_dict
    )
    for (k, v), (k_loaded, v_loaded) in zip(
        model.nn_module.train_metrics_module_dict.items(), model_loaded.nn_module.train_metrics_module_dict.items()
    ):
        assert k == k_loaded
        assert v == v_loaded
    for (k, v), (k_loaded, v_loaded) in zip(
        model.nn_module.eval_metrics_module_dict.items(), model_loaded.nn_module.eval_metrics_module_dict.items()
    ):
        assert k == k_loaded
        assert v == v_loaded
    for (k, v), (k_loaded, v_loaded) in zip(
        model.nn_module.train_metrics_dict.items(), model_loaded.nn_module.train_metrics_dict.items()
    ):
        assert k == k_loaded
        assert type(v) == type(v_loaded)  # noqa: E721
        if isinstance(v, partial):
            assert v.func.__code__.co_code == v_loaded.func.__code__.co_code
            assert v.args == v_loaded.args
            assert v.keywords == v_loaded.keywords
        else:
            assert v.__code__.co_code == v_loaded.__code__.co_code
    for (k, v), (k_loaded, v_loaded) in zip(
        model.nn_module.eval_metrics_dict.items(), model_loaded.nn_module.eval_metrics_dict.items()
    ):
        assert k == k_loaded
        assert type(v) == type(v_loaded)  # noqa: E721
        if isinstance(v, partial):
            assert v.func.__code__.co_code == v_loaded.func.__code__.co_code
            assert v.args == v_loaded.args
            assert v.keywords == v_loaded.keywords
        else:
            assert v.__code__.co_code == v_loaded.__code__.co_code


def test_save_and_load_pipeline(tmp_path):
    tf_idf = TfidfVectorizer()
    selector = SelectKBest(f_classif)
    pipeline = Pipeline([("tf_idf", tf_idf), ("selector", selector)])

    input_size = 512
    model = NNClassifier(
        **MODEL_PARAMS,
        input_size=input_size,
        neural_network=MLP(input_size),
        tokenizer=pipeline,
        callbacks=ModelSummary(),
    )

    model.save(str(tmp_path))
    model_loaded = NNClassifier.load(str(tmp_path), device=MODEL_PARAMS["device"], many_gpus=MODEL_PARAMS["many_gpus"])

    assert model._get_model_params() == model_loaded._get_model_params()
    assert model.is_fitted == model_loaded.is_fitted
    assert model.nn_module.is_classification == model.nn_module.is_classification
    assert model.nn_module.is_multiclass == model.nn_module.is_multiclass
    assert model.nn_module.optimizer_class == Adam

    # assert types to be more specific than 'isinstance()'
    assert type(model) == type(model_loaded)  # noqa: E721
    assert type(model.tokenizer) == type(model_loaded.tokenizer)  # noqa: E721
    assert type(model.callbacks) == type(model_loaded.callbacks)  # noqa: E721
    assert type(model.nn_module.neural_network) == type(model_loaded.nn_module.neural_network)  # noqa: E721
    assert type(model.nn_module.loss_fun) == type(model_loaded.nn_module.loss_fun)  # noqa: E721


def test_save_slim(seed, tmp_path):
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    x_test = ["tomato", "chair"]

    tf_idf = TfidfVectorizer()
    input_size = len(x_train)

    model = NNClassifier(**MODEL_PARAMS, input_size=input_size, neural_network=MLP(input_size), tokenizer=tf_idf)
    model.fit_tokenizer(x_train)
    model.fit(x_train, y_train)

    model.save_slim(str(tmp_path))
    model_loaded = NNClassifier.load(str(tmp_path), device=MODEL_PARAMS["device"], many_gpus=MODEL_PARAMS["many_gpus"])

    scores = model.predict_scores(x_test)
    scores_loaded = model_loaded.predict_scores(x_test)

    assert np.array_equal(scores, scores_loaded)
    assert scores[0] < 0.3
    assert scores[1] > 0.7


def test_continue_training_after_loading(seed, tmp_path):
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    x_test = ["tomato", "chair"]
    expected_classes = [0, 1]

    tf_idf = TfidfVectorizer()
    input_size = len(x_train)
    model_params = {
        "batch_size": 2,
        "epochs": 2,
        "input_size": input_size,
        "optimizer_params": {"lr": 1e-1},
        "device": "cpu",
        "many_gpus": False,
    }

    model = NNClassifier(**model_params, neural_network=MLP(input_size), tokenizer=tf_idf)
    model.fit_tokenizer(x_train)
    model.fit(x_train, y_train)

    model.save(str(tmp_path))
    model_loaded = NNClassifier.load(str(tmp_path), device=model_params["device"], many_gpus=model_params["many_gpus"])

    optimizer_state = model.nn_module.optimizers().optimizer.state_dict()
    assert model_loaded.nn_module.optimizer_state.keys() == optimizer_state.keys()
    assert model_loaded.nn_module.optimizer_state["param_groups"] == optimizer_state["param_groups"]

    model_loaded.epochs = 4
    model_loaded.fit(x_train, y_train)

    predictions = model_loaded.predict(x_test)
    scores = model_loaded.predict_scores(x_test)

    assert len(predictions) == len(x_test)
    assert np.array_equal(predictions, expected_classes)
    assert scores[0] < 0.3
    assert scores[1] > 0.7


def test_continue_training_tmp(seed):
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    x_test = ["tomato", "chair"]
    expected_classes = [0, 1]

    tf_idf = TfidfVectorizer()
    input_size = len(x_train)
    model_params = {
        "batch_size": 2,
        "epochs": 2,
        "input_size": input_size,
        "optimizer_params": {"lr": 1e-1},
        "device": "cpu",
        "many_gpus": False,
    }

    model = NNClassifier(**model_params, neural_network=MLP(input_size), tokenizer=tf_idf)
    model.fit_tokenizer(x_train)
    model.fit(x_train, y_train)
    model.epochs = 4
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = model.predict_scores(x_test)

    assert len(predictions) == len(x_test)
    assert np.array_equal(predictions, expected_classes)
    assert scores[0] < 0.3
    assert scores[1] > 0.7
