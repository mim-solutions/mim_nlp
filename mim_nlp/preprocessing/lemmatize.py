import morfeusz2

MORFEUSZ = morfeusz2.Morfeusz()


def lemmatize(txt: str) -> str:
    """This function lemmatizes the text in Polish."""
    lemmatized_words = []
    for word in txt.split(" "):
        analysis = MORFEUSZ.analyse(word)
        if not analysis:
            continue
        word_lemma = analysis[0][2][1]
        if ":" in word_lemma:
            word_lemma = word_lemma.split(":")[0]
        lemmatized_words.append(word_lemma)
    return " ".join(lemmatized_words)
