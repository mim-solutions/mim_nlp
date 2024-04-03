from __future__ import annotations

import json
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy._typing import _ArrayLikeInt_co, _ArrayLikeStr_co
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from mim_nlp.classifier.svm.svm_explainer import (
    explain,
    get_top_words,
    plot_explanation,
)
from mim_nlp.classifier.svm.svm_reduction import reduce_tfidf_model
from mim_nlp.models import Classifier

Explanation = namedtuple("Explanation", ["word", "score"])


class SVMClassifier(Classifier):
    def __init__(
        self, tfidf_transformer_parameters: dict, linear_svc_parameters: dict, pipeline: Optional[Pipeline] = None
    ):
        self.params = {
            "tfidf": tfidf_transformer_parameters,
            "svc": linear_svc_parameters,
        }
        if pipeline:
            self.pipeline = pipeline
        else:
            tfidf_transformer = TfidfVectorizer(**tfidf_transformer_parameters)
            svc = LinearSVC(**linear_svc_parameters)
            self.pipeline = Pipeline([("tfidf", tfidf_transformer), ("model", svc)])

    def fit(self, x_train: _ArrayLikeStr_co, y_train: _ArrayLikeInt_co) -> None:
        self.pipeline.fit(x_train, y_train)

    def predict(self, x: _ArrayLikeStr_co) -> NDArray[np.int64]:
        predicted_classes = self.pipeline.predict(x)
        return predicted_classes

    def predict_scores(self, x: _ArrayLikeStr_co) -> NDArray[np.float64]:
        return self.pipeline.decision_function(x)

    def save(self, model_dir: Union[str, Path]) -> None:
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(file=model_dir / "metadata.json", mode="w", encoding="utf-8") as file:
            json.dump({"model_type": "svc"}, file)
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as file:
            json.dump(self.params, file)
        with open(file=model_dir / "model.pickle", mode="wb") as handle:
            pickle.dump(self.pipeline, handle)

    def save_without_stop_words(self, model_dir: Union[str, Path]) -> None:
        self.drop_stop_words_set()
        self.save(model_dir)

    def drop_stop_words_set(self) -> None:
        """The method deletes set of stopwords saved in tf idf part of the model.

        For more info see the documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html.
        """
        self.pipeline[0].stop_words_ = None

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> SVMClassifier:
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        with open(file=model_dir / "params.json", mode="r", encoding="utf-8") as file:
            params = json.load(file)
        tfidf_transformer_parameters = params["tfidf"]
        linear_svc_parameters = params["svc"]
        with open(file=model_dir / "model.pickle", mode="rb") as handle:
            svc_pipeline = pickle.load(handle)
        return cls(tfidf_transformer_parameters, linear_svc_parameters, svc_pipeline)

    def get_weighted_coefs_from_text(self, text: str) -> tuple[list[str], list[float]]:
        vectorizer = self.pipeline[:-1]
        all_words = vectorizer.get_feature_names_out()
        svc = self.pipeline[-1]

        all_coefs = svc.coef_[0]

        vectorized = vectorizer.transform([text])
        all_freqs = vectorized.A[0]
        mask = all_freqs > 0
        words = all_words[mask]
        freqs = all_freqs[mask]
        coefs = all_coefs[mask]

        weighted_coefs = freqs * coefs

        return words, weighted_coefs

    def text_score_and_explanation(self, text: str, top_k: int) -> tuple[float, list[Explanation]]:
        score = float(self.pipeline.decision_function([text])[0])
        explanation = self.get_top_words_from_text(text, top_k)
        return score, explanation

    def explain_text(self, text: str, top_k: int) -> None:
        words, weighted_coefs = self.get_weighted_coefs_from_text(text)

        labels, values = explain(words, weighted_coefs, top_k)

        svc = self.pipeline[-1]

        intercept = None
        if svc.fit_intercept:
            intercept = svc.intercept_[0]

        plot_explanation(labels, values, intercept)

    def get_top_words_from_text(self, text: str, top_k: int) -> list[Explanation]:
        words, weighted_coefs = self.get_weighted_coefs_from_text(text)
        words_selected, coefs_selected = get_top_words(words, weighted_coefs, top_k)
        results_clean = [Explanation(word, score) for word, score in zip(words_selected, coefs_selected)]
        return results_clean


class SVMClassifierWithFeatureSelection(SVMClassifier):
    def __init__(
        self,
        tfidf_transformer_parameters,
        feature_selector_parameters,
        linear_svc_parameters,
        pipeline=None,
    ):
        super().__init__(tfidf_transformer_parameters, linear_svc_parameters, pipeline)
        self.params["feature_selector"] = feature_selector_parameters

        if not pipeline:
            feature_selector = SelectKBest(f_classif, **feature_selector_parameters)
            self.pipeline.steps.insert(1, ("feature_selector", feature_selector))

    def save(self, model_dir: Union[str, Path]) -> None:
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(file=model_dir / "metadata.json", mode="w", encoding="utf-8") as file:
            json.dump({"model_type": "SVMClassifierWithFeatureSelection"}, file)
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as file:
            json.dump(self.params, file)
        with open(file=model_dir / "model.pickle", mode="wb") as handle:
            pickle.dump(self.pipeline, handle)

    def save_with_only_selected_words(self, model_dir: Union[str, Path]) -> None:
        """The method saves SVMClassifierWithFeatureSelection model with only words necessary for prediction."""
        reduced_model = self.reduce_to_only_selected_words()
        reduced_model.save_without_stop_words(model_dir)

    def reduce_to_only_selected_words(self) -> SVMClassifier:
        """The method reduces the model SVMClassifierWithFeatureSelection to only n-grams necessary for prediction.

        The following steps are done:

        - Decompose the model pipeline to tfidf, selector and svc.
        - Get the list of all selected words by the method model.pipeline[:-1].get_feature_names_out().
        - That way the order of the words is the same as the order of the parameters of svc.
        - Reduce tfidf model to only selected words based on this list.
        - Now the feature selection part is redundant and should be deleted from the pipeline.
        - Create a pipeline only from reduced tfidf and svc.
        - Save it as a basic SVMClassifier model.
        """
        tfidf_transformer, _, svc = self.pipeline
        tfidf_transformer_parameters = self.params["tfidf"]
        linear_svc_parameters = self.params["svc"]

        selected_words = self.pipeline[:-1].get_feature_names_out()

        tfidf_reduced = reduce_tfidf_model(tfidf_transformer, tfidf_transformer_parameters, selected_words)
        new_pipeline = Pipeline([("tfidf", tfidf_reduced), ("model", svc)])
        return SVMClassifier(tfidf_transformer_parameters, linear_svc_parameters, new_pipeline)

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> SVMClassifierWithFeatureSelection:
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        with open(file=model_dir / "params.json", mode="r", encoding="utf-8") as file:
            params = json.load(file)
        tfidf_transformer_parameters = params["tfidf"]
        feature_selector_parameters = params["feature_selector"]
        linear_svc_parameters = params["svc"]
        with open(file=model_dir / "model.pickle", mode="rb") as handle:
            svc_pipeline = pickle.load(handle)
        return cls(
            tfidf_transformer_parameters,
            feature_selector_parameters,
            linear_svc_parameters,
            svc_pipeline,
        )
