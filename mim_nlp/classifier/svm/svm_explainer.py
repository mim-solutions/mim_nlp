import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeStr_co


def explain(words: _ArrayLikeStr_co, coefs: _ArrayLikeFloat_co, top_k: int) -> tuple[list[str], list[float]]:
    negative_mask = coefs < 0
    negative_coefs = coefs[negative_mask]
    positive_coefs = coefs[~negative_mask]
    negative_words = words[negative_mask]
    positive_words = words[~negative_mask]

    labels = []
    values = []

    # negative words

    words_selected, values_selected = sort_and_select(negative_words, negative_coefs, top_k, False)
    labels.extend(words_selected)
    values.extend(values_selected)

    labels.append("other_negative")
    values.append(sum(negative_coefs) - sum(values_selected))

    # positive words

    words_selected, values_selected = sort_and_select(positive_words, positive_coefs, top_k, True)
    labels.extend(words_selected)
    values.extend(values_selected)

    labels.append("other_positive")
    values.append(sum(positive_coefs) - sum(values_selected))

    return labels, values


def plot_explanation(labels: _ArrayLikeStr_co, values: _ArrayLikeFloat_co, intercept: bool = None) -> None:
    labels = [x for _, x in sorted(zip(values, labels))]
    values = sorted(values)

    negative_bars = sum(np.array(values) < 0)
    positive_bars = sum(np.array(values) >= 0)
    colors = ["red"] * negative_bars + ["green"] * positive_bars

    if intercept is not None:
        labels.append("intercept")
        values.append(intercept)
        colors.append("blue")

    _, ax = plt.subplots(figsize=(10, 10))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, values, align="center", color=colors)
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Scores")
    ax.set_title("Model decision explained")

    plt.show()


def sort_and_select(
    words: _ArrayLikeStr_co, values: _ArrayLikeFloat_co, top_k: int, reverse: bool
) -> tuple[list[str], list[float]]:
    words_selected = [x for _, x in sorted(zip(values, words), reverse=reverse)][:top_k]
    values_selected = sorted(values, reverse=reverse)[:top_k]
    return words_selected, values_selected


def get_top_words(words: _ArrayLikeStr_co, coefs: _ArrayLikeFloat_co, top_k: int) -> tuple[list[str], list[float]]:
    # get absolute values of coefs
    absolute_coefs = np.abs(coefs)

    words_selected = [word for _, word in sorted(zip(absolute_coefs, words), reverse=True)][:top_k]
    coefs_selected = [coef for _, coef in sorted(zip(absolute_coefs, coefs), reverse=True)][:top_k]

    remainder = sum(coefs) - sum(coefs_selected)

    words_selected.append("other words")
    coefs_selected.append(remainder)

    return words_selected, coefs_selected
