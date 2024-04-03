from sklearn.feature_extraction.text import TfidfVectorizer


def reduce_tfidf_model(
    tfidf_transformer: TfidfVectorizer, tfidf_transformer_parameters: dict, selected_words: list[str]
) -> TfidfVectorizer:
    """The function reduces the tf-idf model to only list of selected words."""
    vocabulary = tfidf_transformer.vocabulary_
    idfs = tfidf_transformer.idf_
    vocabulary_reduced, idfs_reduced = reduce_tfidf_vocabulary(vocabulary, idfs, selected_words)
    reduced_tfidf = TfidfVectorizer(vocabulary=vocabulary_reduced, **tfidf_transformer_parameters)
    reduced_tfidf.idf_ = idfs_reduced

    return reduced_tfidf


def reduce_tfidf_vocabulary(
    vocabulary: dict[str, int], idfs: list[float], selected_words: list[str]
) -> tuple[dict[str, int], list[float]]:
    """The function reduces the tf-idf vocabulary and idfs list to list of selected word.

    The order of words is unchanged.
    See the unit test in tests/test_tfidf_reduction.py for an example.
    """
    vocabulary_reduced = {}
    idfs_reduced = []

    for i, word in enumerate(selected_words):
        vocabulary_reduced[word] = i
        idf = idfs[vocabulary[word]]
        idfs_reduced.append(idf)
    return vocabulary_reduced, idfs_reduced
