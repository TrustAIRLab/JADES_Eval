# Release validation

Before preparing the first public 0.1.3 release, 284 offline regression tests passed locally on Python 3.10. They cover baseline behavior, request replay, configuration, resource isolation, refusal warnings, usage, concurrency, cache and resume.

Wheel and source distributions were installed in separate isolated environments, with installed-package import, mocked evaluation and CLI smoke checks. Historical completed checkpoints were also resumed locally without new model requests. Private datasets, raw research outputs and credentials are not part of this repository or its distributions.

GitHub Actions runs the test matrix and checks the distributions for each main-branch update and release tag. Inspect the corresponding workflow run for public CI results. The unit tests use synthetic responses and do not contact LLM or search providers or download NLP model resources.
