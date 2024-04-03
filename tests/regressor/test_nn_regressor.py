import torch.nn as nn
from numpy.testing import assert_almost_equal
from pytorch_lightning.callbacks import ModelSummary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import WordPieceTrainer
from torch.optim import Adam
from transformers import PreTrainedTokenizerFast

from mim_nlp.regressor import NNRegressor


class MLPNN(nn.Module):
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


MODEL_PARAMS = {
    "batch_size": 2,
    "epochs": 4,
    "optimizer_params": {"lr": 1e-1},
    "device": "cpu",
    "many_gpus": False,
}


def test_nn_regressor(seed):
    x_train = ["abc", "def", "abe", "cde"] + ["xyz", "wxy", "wxz", "uyz"]
    y_train = [0.01, 0.05, 0.03, 0.005, 99.99, 99.97, 99.95, 99.995]
    x_test = ["abc", "cde", "rtu"]

    vocab = {chr(ascii_idx): i for i, ascii_idx in enumerate(range(97, 123))}  # a: 0, b: 1, ...

    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(Regex("."), "isolated")
    tokenizer.train_from_iterator(
        ["".join(list(vocab.keys()))],
        WordPieceTrainer(
            vocab_size=100, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], show_progress=True
        ),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "unk_token": "[UNK]"})

    input_size = 3
    model = NNRegressor(**MODEL_PARAMS, input_size=input_size, neural_network=MLPNN(input_size), tokenizer=tokenizer)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    assert len(predictions) == len(x_test)
    assert predictions[0] < 10
    assert predictions[1] < 10
    assert predictions[2] > 10


def test_prediction_order(seed):
    """Check if the order of predictions is preserved and predictions are reproducible."""
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 99.99, 99.99, 99.99, 99.99]

    x_test = ["carrot"] * 99 + ["chair"] * 1

    tf_idf = TfidfVectorizer()

    input_size = len(x_train)
    model = NNRegressor(**MODEL_PARAMS, input_size=input_size, neural_network=MLPNN(input_size), tokenizer=tf_idf)
    model.fit_tokenizer(x_train)
    model.fit(x_train, y_train)

    expected_prediction_carrot = model.predict(["carrot"])[0]
    expected_prediction_chair = model.predict(["chair"])[0]

    # Test if only the last prediction is different.
    # This will fail if the predictions are shuffled.
    predictions = model.predict(x_test)
    for i in range(len(x_test) - 1):
        assert_almost_equal(predictions[i], expected_prediction_carrot, decimal=6)
    assert_almost_equal(predictions[-1], expected_prediction_chair, decimal=6)


def test_save_and_load_pre_trained_tokenizer(tmp_path):
    vocab = {chr(ascii_idx): i for i, ascii_idx in enumerate(range(97, 123))}  # a: 0, b: 1, ...
    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(Regex("."), "isolated")
    tokenizer.train_from_iterator(
        ["".join(list(vocab.keys()))],
        WordPieceTrainer(
            vocab_size=100, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], show_progress=True
        ),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "unk_token": "[UNK]"})

    input_size = 512
    model = NNRegressor(
        **MODEL_PARAMS,
        input_size=input_size,
        neural_network=MLPNN(input_size),
        tokenizer=tokenizer,
        callbacks=ModelSummary()
    )

    model.save(str(tmp_path))
    model_loaded = NNRegressor.load(str(tmp_path), device=MODEL_PARAMS["device"], many_gpus=MODEL_PARAMS["many_gpus"])

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


def test_save_and_load_pipeline(tmp_path):
    tf_idf = TfidfVectorizer()
    selector = SelectKBest(f_regression)
    pipeline = Pipeline([("tf_idf", tf_idf), ("selector", selector)])

    input_size = 512
    model = NNRegressor(
        **MODEL_PARAMS,
        input_size=input_size,
        neural_network=MLPNN(input_size),
        tokenizer=pipeline,
        callbacks=ModelSummary()
    )

    model.save(str(tmp_path))
    model_loaded = NNRegressor.load(str(tmp_path), device=MODEL_PARAMS["device"], many_gpus=MODEL_PARAMS["many_gpus"])

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


def test_save_without_stop_words(tmp_path):
    tf_idf = TfidfVectorizer()
    selector = SelectKBest(f_regression)
    pipeline = Pipeline([("tf_idf", tf_idf), ("selector", selector)])

    input_size = 512
    model = NNRegressor(
        **MODEL_PARAMS,
        input_size=input_size,
        neural_network=MLPNN(input_size),
        tokenizer=pipeline,
        callbacks=ModelSummary()
    )

    model.save_without_stop_words(str(tmp_path))
    model_loaded = NNRegressor.load(str(tmp_path), device=MODEL_PARAMS["device"], many_gpus=MODEL_PARAMS["many_gpus"])

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
