# Changelog

## 0.1.3

First public distribution of the JADES_light-compatible evaluator.

- Install `jades-eval` with pip and use the `jades` Python API or CLI.
- Configure each module through OpenAI-compatible endpoints, TOML and dotenv credentials.
- Track request/sample/batch time and server-reported tokens, including retries and incomplete usage.
- Resume completed samples through atomic checkpoints, exclusive output locks and a durable request ledger.
- Preserve the reference NLP, prompts, ordered scoring and weighted formula; optional fact checking is disabled by default.
- Keep textual evaluator-refusal detection warning-only and fingerprint display opt-in.
- Fix evaluator configuration drift, NLTK resource/tokenizer mismatch, search concurrency, malformed-query repair budgets and early validation of metadata and explicit env files.

Evaluator configuration is fixed at construction: create a new instance to change models, endpoints or concurrency. Recovery is per sample, not per partially completed node. See `docs/p2-fixes-0.1.3.md` for details.
