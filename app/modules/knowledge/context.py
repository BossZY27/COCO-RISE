"""A request gets one evidence snapshot; concurrent requests never share mutable data."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

snapshot: ContextVar[SimpleNamespace | None] = ContextVar("rise_corpus", default=None)


@contextmanager
def use_corpus(corpus):
    token = snapshot.set(corpus)
    try:
        yield corpus
    finally:
        snapshot.reset(token)
