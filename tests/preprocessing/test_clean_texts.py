from functools import partial

import gensim.parsing.preprocessing as gsp
import numpy as np
import pytest

from mim_nlp.preprocessing import (
    FunctionCannotBePickledException,
    TextCleaner,
    lemmatize,
    process_emojis,
    remove_urls,
    strip_multiple_emojis,
    strip_short_words,
    token_usernames,
)

TEXT_0 = """
    Niedzielski o kontrakcie z #Pfizerem 💉: odmawiamy przyjmowania szczepionek i wykonywania płatności
#PAPinformacje
https://t.co/1A3FuXPNWh
"""
TEXT_1 = """
➡️ 18:30 Polska wstrzyma dostawy od Pfizera?! Banderowski marsz w kościele! Przegląd mediów

➡️ 20:00 Czeka nas nowa p@ndemia? Kiedy skończy się wojna? Poświąteczny program z Widzami! Ostrowska i Zdziarski

👉ZAPRASZAMY na platformę video BEZ CENZURY https://t.co/5FckdF4wnH!

2/2
    """

TEXTS = [TEXT_0, TEXT_1]


def lowercase(x: str) -> str:
    return x.lower()


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_clean_text_no_pipeline(multiprocessing: bool):
    pipeline = []
    text_cleaner = TextCleaner(pipeline=pipeline)

    clean_texts = text_cleaner.clean_texts(TEXTS, multiprocessing)

    assert np.array_equal(clean_texts, TEXTS)


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_clean_text_example_pipeline(multiprocessing: bool):
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
    ]
    text_cleaner = TextCleaner(pipeline=pipeline)
    text_0_expected_result = """niedzielski kontrakcie pfizerem 💉 odmawiamy przyjmowania szczepionek wykonywania płatności papinformacje"""  # noqa: E501
    text_1_expected_result = """➡ polska wstrzyma dostawy pfizera banderowski marsz kościele przegląd mediów ➡ czeka nas nowa kiedy skończy się wojna poświąteczny program widzami ostrowska zdziarski 👉 zapraszamy platformę video bez cenzury"""  # noqa: E501
    clean_texts = text_cleaner.clean_texts(TEXTS, multiprocessing)

    assert clean_texts[0] == text_0_expected_result
    assert clean_texts[1] == text_1_expected_result


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_clean_and_lemmatize_text_example_pipeline(multiprocessing: bool):
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
    text_cleaner = TextCleaner(pipeline=pipeline)
    text_0_expected_result = """nieDzielski kontrakt pfizerem 💉 odmawiać przyjmować szczepionka wykonywać płatność papinformacje"""  # noqa: E501
    text_1_expected_result = """➡ polski wstrzymać dostawa pfizera banderowski marsz kościół przegląd media ➡ czekać my nowa kiedy skończyć się wojna poświąteczny program widz ostrowski zdziarski 👉 zapraszać platforma video bez cenzura"""  # noqa: E501
    clean_texts = text_cleaner.clean_texts(TEXTS, multiprocessing)

    assert clean_texts[0] == text_0_expected_result
    assert clean_texts[1] == text_1_expected_result


def test_multiprocessing_lambda_exception():
    pipeline = [
        lambda x: x.lower(),
    ]
    text_cleaner = TextCleaner(pipeline=pipeline)

    with pytest.raises(FunctionCannotBePickledException):
        text_cleaner.clean_texts(TEXTS, multiprocessing=True)
