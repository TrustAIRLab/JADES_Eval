"""Same NLP assets as light, lazily loaded with explicit provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import threading
import time
from pathlib import Path

SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_lock = threading.RLock()
_models: dict[tuple, object] = {}
_nltk_checks: dict[tuple, tuple] = {}
_tokenizers: dict[tuple, object] = {}
_local = threading.local()


def configure(resource_dir=None, revision="main", recorder=None):
    _local.root = Path(resource_dir or ".jades/resources").resolve()
    _local.revision = revision
    _local.recorder = recorder
    _local.nltk_resource = None


def _event(name, start, **meta):
    recorder = getattr(_local, "recorder", None)
    if recorder:
        recorder.resource(name, time.perf_counter() - start, **meta)


def ensure_nltk(*, force=False):
    start = time.perf_counter()
    import nltk
    root = (getattr(_local, "root", None) or Path(".jades/resources").resolve()) / "nltk"
    downloaded = []
    with _lock:
        # Resolve this evaluator's resources without modifying NLTK's process-global path.
        paths = [str(root), *[p for p in nltk.data.path if p != str(root)]]
        # Resource directories and versions are fixed during a run. An effective path,
        # library version or configured revision change requires a fresh check.
        key = (str(root), tuple(paths), nltk.__version__, getattr(_local, "revision", "main"))
        cached = _nltk_checks.get(key)
        reused = cached is not None and not force
        if not reused:
            _nltk_checks.pop(key, None)
            for resource in ("punkt", "punkt_tab"):
                try:
                    nltk.data.find(f"tokenizers/{resource}", paths=paths)
                except LookupError:
                    root.mkdir(parents=True, exist_ok=True)
                    if not nltk.download(resource, download_dir=str(root), quiet=True, raise_on_error=True):
                        raise RuntimeError(f"NLTK resource unavailable: {resource}; run jades prepare-resources")
                    downloaded.append(resource)
            path = Path(str(nltk.data.find("tokenizers/punkt_tab/english", paths=paths)))
            h = hashlib.sha256()
            for p in sorted(path.glob("*")):
                if p.is_file():
                    h.update(p.name.encode()); h.update(p.read_bytes())
            metadata = {"version": nltk.__version__, "punkt_tab_english_sha256": h.hexdigest()}
            cached = (metadata, path)
            _nltk_checks[key] = cached
        metadata, path = cached
        _local.nltk_resource = (str(path), metadata["punkt_tab_english_sha256"], nltk.__version__)
    _event("nltk", start, **metadata, downloaded=downloaded, reused=reused)


def sent_tokenize(text):
    """Same English Punkt algorithm, bound to the resources checked for this thread."""
    key = getattr(_local, "nltk_resource", None)
    if key is None:
        ensure_nltk()
        key = _local.nltk_resource
    start = time.perf_counter()
    with _lock:
        tokenizer = _tokenizers.get(key)
        if tokenizer is None:
            from nltk.tokenize.punkt import PunktSentenceTokenizer, load_punkt_params
            tokenizer = PunktSentenceTokenizer(load_punkt_params(key[0]))
            _tokenizers[key] = tokenizer
            _event("nltk_tokenizer", start, punkt_tab_english_sha256=key[1], version=key[2])
    return tokenizer.tokenize(text)


def get_semantic_checker():
    start = time.perf_counter()
    root = getattr(_local, "root", None) or Path(".jades/resources").resolve()
    revision = getattr(_local, "revision", "main")
    key = (str(root), revision)
    with _lock:
        loaded = key in _models
        if not loaded:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
            # Prefer the user's existing original/light cache; never change its contents.
            try:
                snapshot = snapshot_download(SEMANTIC_MODEL, revision=revision, local_files_only=True)
            except Exception:
                snapshot = snapshot_download(SEMANTIC_MODEL, revision=revision, cache_dir=str(root / "models"))
            model = SentenceTransformer(snapshot)
            versions = {"sentence_transformers": importlib.metadata.version("sentence-transformers"),
                        "torch": importlib.metadata.version("torch")}
            _models[key] = (model, Path(snapshot).name, versions)
        model, resolved_revision, versions = _models[key]
    _event("semantic_model", start, model=SEMANTIC_MODEL, revision=resolved_revision, reused=loaded,
           **versions)
    return model


def prepare(resource_dir=None, revision="main", recorder=None):
    configure(resource_dir, revision, recorder)
    ensure_nltk(force=True)
    get_semantic_checker()
