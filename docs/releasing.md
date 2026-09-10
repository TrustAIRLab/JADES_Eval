# Publishing GitHub and PyPI releases

The release workflow is `.github/workflows/release.yml`. Main-branch pushes and pull requests test/build only. A pushed `vVERSION` tag runs the same checks, publishes the checked distributions to PyPI with Trusted Publishing, then attaches those exact files and checksums to a GitHub Release draft before publishing it.

## One-time account setup

On PyPI, add a pending GitHub publisher for project `jades-eval`, owner `TrustAIRLab`, repository `jades-eval`, workflow `release.yml`, environment `pypi`. Create the matching GitHub environment. No PyPI API token, HF token or search credential is needed in Actions.

## Prepare a version

1. Update the package version in pyproject.toml, src/jades/__init__.py, src/jades/cli.py and src/jades/evaluator.py, and update CHANGELOG.md.
2. Run `python -m pytest -q`, `python -m build`, `python -m twine check --strict dist/*` and `python tools/check_release.py` (Python 3.11+ for the release checker).
3. Review the Git diff and push the commit to main. Wait for the main-branch workflow to pass.
4. Tag that commit and push the tag, for example `git tag -a v0.1.3 -m "Release 0.1.3"` then `git push origin v0.1.3`.
5. Confirm the PyPI and GitHub release jobs both succeeded and test `pip install jades-eval==0.1.3` in a separate environment.

The workflow keeps testing/building separate from publishing permissions, uses pinned action commits, and verifies the tag matches the package version. Only the current wheel and sdist are passed to PyPI. Credentials, research data, local caches, tests and internal tools are excluded from the distributions.

GitHub and PyPI are separate services, so the operation is not atomic. If publishing to PyPI succeeds but the GitHub release job fails, rerun only failed jobs to use the existing artifacts. Do not rebuild and overwrite an already published PyPI version. Never move a published version tag. Pending publishers do not reserve package names.
