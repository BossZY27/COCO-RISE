"""Thai-aware tokenizer for the demo.

Thai has no spaces between words, so BM25 needs the text split first. This module does a longest match
against a small domain dictionary, then falls back to character trigrams for anything it does not know.
For real use install PyThaiNLP and swap tokenize() for pythainlp.word_tokenize.
"""
import re

LATIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9,./%-]*")
THAI = re.compile(r"[฀-๿]+")


def trigrams(text: str) -> list[str]:
    if len(text) <= 3:
        return [text] if text else []
    return [text[i:i + 3] for i in range(len(text) - 2)]


def segment_thai(text: str, vocab: set[str]) -> list[str]:
    """Longest dictionary match; unknown stretches become trigrams so they can still be matched."""
    out: list[str] = []
    unknown = ""
    longest = max((len(w) for w in vocab), default=0)
    i = 0
    while i < len(text):
        hit = ""
        for size in range(min(longest, len(text) - i), 1, -1):
            if text[i:i + size] in vocab:
                hit = text[i:i + size]
                break
        if hit:
            if unknown:
                out.extend(trigrams(unknown))
                unknown = ""
            out.append(hit)
            i += len(hit)
        else:
            unknown += text[i]
            i += 1
    if unknown:
        out.extend(trigrams(unknown))
    return out


def tokenize(text: str, vocab: set[str]) -> list[str]:
    tokens = [m.group().lower() for m in LATIN.finditer(text)]
    for m in THAI.finditer(text):
        tokens.extend(segment_thai(m.group(), vocab))
    return tokens
