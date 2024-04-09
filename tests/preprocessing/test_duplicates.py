from mim_nlp.preprocessing import Deduplicator

TEXTS = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",  # noqa: E501
    "Ala ma kota",
    "Ipsum lorem dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",  # noqa: E501
]


def test_clusterize_partial_duplicates():
    deduplicator = Deduplicator()
    expected_clusters = [0, 1, 0]
    clusters = deduplicator.clusterize_partial_duplicates(TEXTS)
    assert expected_clusters == clusters
