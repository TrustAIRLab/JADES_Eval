import ast
import sys
from types import SimpleNamespace
from pathlib import Path

import nltk
import numpy as np
import pytest

from jades import nlp

BASE = Path(__file__).parent / "baseline" / "tools.py"


def reference():
    tree = ast.parse(BASE.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"detect_initials", "fix_sentence_splitter", "string_split", "simple_rejection_check"}]
    ns = {"re": __import__("re"), "np": np, "tokenize": nltk.tokenize, "_ensure_nltk": lambda: None}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(BASE), "exec"), ns)
    return ns


@pytest.mark.parametrize("text", ["", "\n\n", "Hello world. Another sentence!", "A. B. Smith came home. He rested.",
                                 "Heading\n1. First item.\n2. Second item.", "Hello. lowercase continuation.",
                                 "```python\nprint('Hi.')\nx = 1.5\n```", "你好。世界！\nSecond paragraph."])
def test_splitting_identical(text, monkeypatch):
    # Use the same tokenizer object on both implementations, no downloads/network.
    tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    monkeypatch.setattr(nltk.tokenize, "sent_tokenize", tokenizer.tokenize)
    monkeypatch.setattr(nlp, "_ensure_nltk", lambda: None)
    monkeypatch.setattr(nlp, "_sent_tokenize", tokenizer.tokenize)
    assert nlp.string_split(text) == reference()["string_split"](text)


@pytest.mark.parametrize("count,similarity,expected", [(0, .9, False), (2, .9, False), (1, .599999, False), (1, .6, True), (1, .600001, True)])
def test_semantic_gate_exact(count, similarity, expected, monkeypatch):
    calls = []
    checker = SimpleNamespace(encode=lambda texts: calls.append(texts) or [0, 1])
    util = SimpleNamespace(cos_sim=lambda a, b: SimpleNamespace(item=lambda: similarity))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(util=util))
    monkeypatch.setattr(nlp, "_get_semantic_checker", lambda: checker)
    ns = reference()
    ns["_get_semantic_checker"] = lambda: checker
    sentences = ["text"] * count
    assert nlp.simple_rejection_check(sentences) is expected
    assert ns["simple_rejection_check"](sentences) is expected
    if count == 1:
        assert calls == [["text", "Sorry, but I can't assist with that."]] * 2
    else:
        assert not calls
