# Changelog

## 0.1.4

- Add an English user guide with complete Python and CLI walkthroughs, result inspection, usage metrics, and batch recovery examples.
- Translate repository documentation and example messages into English.
- Update project links to `TrustAIRLab/JADES_Eval` and make documentation links work on both GitHub and PyPI.
- Represent multilingual detection rules and regression samples with Unicode escapes while preserving their exact runtime values.

Evaluation prompts, scoring logic, API behavior, and checkpoint compatibility are unchanged.

## 0.1.3

First public distribution of the JADES_light-compatible evaluator.

- Install `jades-eval` with pip and use the `jades` Python API or CLI.
- Configure each module through OpenAI-compatible endpoints, TOML and dotenv credentials.
- Track request/sample/batch time and server-reported tokens, including retries and incomplete usage.
- Resume completed samples through atomic checkpoints, exclusive output locks and a durable request ledger.
- Preserve the reference NLP, prompts, ordered scoring and weighted formula; optional fact checking is disabled by default.
- Keep textual evaluator-refusal detection warning-only and fingerprint display opt-in.
- Fix evaluator configuration drift, NLTK resource/tokenizer mismatch, search concurrency, malformed-query repair budgets and early validation of metadata and explicit env files.

Evaluator configuration is fixed at construction: create a new instance to change models, endpoints or concurrency. Recovery is per sample, not per partially completed node. See [scoring behavior and compatibility](docs/compatibility.md) for details.
