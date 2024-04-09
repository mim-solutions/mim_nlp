from mim_nlp.regressor import AutoRegressor

MODEL_PARAMS = {
    "batch_size": 1,
    "learning_rate": 1e-1,
    "epochs": 1,
    "device": "cpu",
    "many_gpus": False,
    "pretrained_model_name_or_path": "prajjwal1/bert-tiny",
}


def test_auto_regressor(seed):
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 99.99, 99.99, 99.99, 99.99]
    x_test = ["carrot", "chair"]

    model = AutoRegressor(**MODEL_PARAMS)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    # Predictions are equal (to several decimal places), no matter the model parameters.
    assert len(predictions) == 2


def test_prediction_order(seed):
    """Check if the order of predictions is preserved and predictions are reproducible."""
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 99.99, 99.99, 99.99, 99.99]
    x_test = ["carrot"] * 99 + ["chair"] * 1

    model = AutoRegressor(**MODEL_PARAMS)
    model.fit(x_train, y_train)

    # Sometimes, the predictions aren't reproducible between method calls.
    predictions = model.predict(x_test)
    expected_prediction_carrot = predictions[0]

    # Test if only the last prediction is different.
    # This will fail if the predictions are shuffled.
    for i in range(1, len(x_test) - 1):
        assert predictions[i] == expected_prediction_carrot
    assert predictions[-1] != expected_prediction_carrot


def test_save_and_load(tmp_path):
    model = AutoRegressor(**MODEL_PARAMS)

    model.save(str(tmp_path))
    model_loaded = AutoRegressor.load(str(tmp_path), device=MODEL_PARAMS["device"], many_gpus=MODEL_PARAMS["many_gpus"])

    assert model._get_model_params() == model_loaded._get_model_params()
    # assert types to be more specific than 'isinstance()'
    assert type(model) == type(model_loaded)  # noqa: E721
    assert type(model.tokenizer) == type(model_loaded.tokenizer)  # noqa: E721
    assert type(model.model) == type(model_loaded.model)  # noqa: E721
