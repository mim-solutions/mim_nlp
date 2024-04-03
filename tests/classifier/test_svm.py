from pathlib import Path
from shutil import rmtree

from numpy import array_equal

from mim_nlp.classifier.svm import SVMClassifier

MODEL_PARAMS = {
    "tfidf_transformer_parameters": {
        "sublinear_tf": True,
        "min_df": 1,
        "max_df": 5,
        "norm": "l2",
        "encoding": "latin-1",
        "ngram_range": (1, 2),
    },
    "linear_svc_parameters": {"C": 5, "fit_intercept": True},
}


def test_fit_and_predict():
    """The test is quite naive, but it goes through all the methods."""
    params = MODEL_PARAMS
    model = SVMClassifier(**params)

    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    x_test = ["tomato", "chair"]
    expected_classes = [0, 1]

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    assert len(predictions) == 2
    assert array_equal(predictions, expected_classes)


def test_prediction_order():
    """Check if the order of predictions is preserved."""
    params = MODEL_PARAMS
    model = SVMClassifier(**params)
    x_train = ["carrot", "cucumber", "tomato", "potato"] + ["table", "chair", "door", "floor"]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    x_test = ["carrot"] * 99 + ["chair"] * 1

    model.fit(x_train, y_train)
    predicted_scores = model.predict_scores(x_test)

    expected_score_pepper = model.predict_scores(["carrot"])[0]
    expected_score_chair = model.predict_scores(["chair"])[0]

    # Test if only last prediction is different
    # This will fail if the predictions are shuffled
    for i in range(len(x_test) - 1):
        assert predicted_scores[i] == expected_score_pepper
    assert predicted_scores[-1] == expected_score_chair


def test_multiclass_classification(seed):
    params = MODEL_PARAMS
    model = SVMClassifier(**params)
    x_train = (
        ["carrot", "cucumber", "tomato", "potato"]
        + ["table", "chair", "door", "floor"]
        + ["football", "basketball", "volleyball", "baseball"]
    )
    y_train = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    x_test = ["tomato", "chair", "volleyball", "potato"]
    expected_classes = [0, 1, 2, 0]

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    scores = model.predict_scores(x_test)

    assert len(predictions) == len(x_test)
    assert array_equal(predictions, expected_classes)
    assert scores.shape == (4, 3)


def test_save_and_load():
    params = MODEL_PARAMS
    model = SVMClassifier(**params)
    path = Path("tmp_model_test_dir")

    model.save_without_stop_words(str(path))

    try:
        SVMClassifier.load(str(path))
    finally:
        rmtree(path)
