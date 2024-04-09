import re

import advertools as adv
from gensim.utils import to_unicode

# custom filters
RE_USERNAME = re.compile(r"@([A-Za-z0-9_]+)", re.UNICODE)
RE_HASHTAG = re.compile(
    r"/(#|＃)([a-z0-9_\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0300-\u036f\u1e00-\u1eff\u0400-\u04ff\u0500-\u0527\u2de0-\u2dff\ua640-\ua69f\u0591-\u05bf\u05c1-\u05c2\u05c4-\u05c5\u05d0-\u05ea\u05f0-\u05f4\ufb12-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb40-\ufb41\ufb43-\ufb44\ufb46-\ufb4f\u0610-\u061a\u0620-\u065f\u066e-\u06d3\u06d5-\u06dc\u06de-\u06e8\u06ea-\u06ef\u06fa-\u06fc\u0750-\u077f\u08a2-\u08ac\u08e4-\u08fe\ufb50-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc\u200c-\u200c\u0e01-\u0e3a\u0e40-\u0e4e\u1100-\u11ff\u3130-\u3185\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff\uffa1-\uffdc\u30a1-\u30fa\u30fc-\u30fe\uff66-\uff9f\uff10-\uff19\uff21-\uff3a\uff41-\uff5a\u3041-\u3096\u3099-\u309e\u3400-\u4dbf\u4e00-\u9fff\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2f800-\u2fa1f]*[a-z_\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0300-\u036f\u1e00-\u1eff\u0400-\u04ff\u0500-\u0527\u2de0-\u2dff\ua640-\ua69f\u0591-\u05bf\u05c1-\u05c2\u05c4-\u05c5\u05d0-\u05ea\u05f0-\u05f4\ufb12-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb40-\ufb41\ufb43-\ufb44\ufb46-\ufb4f\u0610-\u061a\u0620-\u065f\u066e-\u06d3\u06d5-\u06dc\u06de-\u06e8\u06ea-\u06ef\u06fa-\u06fc\u0750-\u077f\u08a2-\u08ac\u08e4-\u08fe\ufb50-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc\u200c-\u200c\u0e01-\u0e3a\u0e40-\u0e4e\u1100-\u11ff\u3130-\u3185\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff\uffa1-\uffdc\u30a1-\u30fa\u30fc-\u30fe\uff66-\uff9f\uff10-\uff19\uff21-\uff3a\uff41-\uff5a\u3041-\u3096\u3099-\u309e\u3400-\u4dbf\u4e00-\u9fff\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2f800-\u2fa1f][a-z0-9_\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u024f\u0253-\u0254\u0256-\u0257\u0300-\u036f\u1e00-\u1eff\u0400-\u04ff\u0500-\u0527\u2de0-\u2dff\ua640-\ua69f\u0591-\u05bf\u05c1-\u05c2\u05c4-\u05c5\u05d0-\u05ea\u05f0-\u05f4\ufb12-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb40-\ufb41\ufb43-\ufb44\ufb46-\ufb4f\u0610-\u061a\u0620-\u065f\u066e-\u06d3\u06d5-\u06dc\u06de-\u06e8\u06ea-\u06ef\u06fa-\u06fc\u0750-\u077f\u08a2-\u08ac\u08e4-\u08fe\ufb50-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc\u200c-\u200c\u0e01-\u0e3a\u0e40-\u0e4e\u1100-\u11ff\u3130-\u3185\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff\uffa1-\uffdc\u30a1-\u30fa\u30fc-\u30fe\uff66-\uff9f\uff10-\uff19\uff21-\uff3a\uff41-\uff5a\u3041-\u3096\u3099-\u309e\u3400-\u4dbf\u4e00-\u9fff\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2f800-\u2fa1f]*)/gi",  # noqa: E501
    re.UNICODE,
)
RE_QUOTE = re.compile(r'(["\'])(?:(?=(\\?))\2.)*?\1', re.UNICODE)
RE_URL = re.compile(
    r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",  # noqa: E501
    re.UNICODE,
)
# skin and hair modifiers, variants and zero-width-joiner
RE_EMOJI_MODIFIERS = re.compile(rb"u0001f3f[b-f]|u0001f9b[0-3]|ufe0[e-f]|u200d")


def token_usernames(txt: str, token: str = "") -> str:
    txt = to_unicode(txt)
    return RE_USERNAME.sub(token, txt)


def remove_urls(txt: str) -> str:
    txt = to_unicode(txt)
    return RE_URL.sub(" ", txt)


def process_emojis(txt: str) -> str:
    emoji_list = adv.extract_emoji([txt])["emoji"][0]
    for emoji_detected in emoji_list:
        encoded_slash = "\\".encode("unicode-escape")[:1]
        emoji_encoded = emoji_detected.encode("unicode-escape")
        emoji_splitted = emoji_encoded.split(encoded_slash)

        for i, emoji in enumerate(emoji_splitted):
            if len(emoji) < 1:
                continue

            # first character is not `u`
            if emoji.lower()[0:1] != b"u":
                break

            # emoji with modifiers
            if RE_EMOJI_MODIFIERS.match(emoji.lower()):
                emoji_replaced = encoded_slash.join(emoji_splitted[:i]).decode("unicode-escape")
                txt = txt.replace(emoji_detected, " " + emoji_replaced + " ")
                break

            # emoji without modifiers
            if i == len(emoji_splitted) - 1:
                txt = txt.replace(emoji_detected, " " + emoji_detected + " ")
                break
    return txt


def strip_short_words(txt: str, minsize: int = 3) -> str:
    """Removes words shorter than minsize omitting emojis."""

    def filter_short_words(text):
        return (
            len(text) >= minsize
            or adv.extract_emoji(
                [
                    text,
                ]
            )[
                "emoji_counts"
            ][0]
        )

    txt = to_unicode(txt)
    return " ".join(list(filter(filter_short_words, txt.split(" "))))


def strip_multiple_emojis(txt: str) -> str:
    txt = to_unicode(txt)
    splitted = txt.split(" ")
    if len(splitted) < 2:
        return txt

    processed = [splitted[0]]
    for i in range(1, len(splitted)):
        if (
            adv.extract_emoji(
                [
                    splitted[i],
                ]
            )[
                "emoji_counts"
            ][0]
            and splitted[i] == splitted[i - 1]
        ):
            continue
        processed.append(splitted[i])

    return " ".join(processed)


def remove_quotes(txt: str) -> str:
    txt = to_unicode(txt)
    return RE_QUOTE.sub("", txt)


def remove_hashtags(txt: str) -> str:
    txt = to_unicode(txt)
    return RE_HASHTAG.sub("", txt)
