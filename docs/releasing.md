# Release process

## 1. Pre-release validation

From a clean checkout:

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,examples]"
python -m pytest
python -m ruff check .
```

Confirm that `CHANGELOG.md`, `CITATION.cff`, `src/rangebranch/__init__.py`, and
`pyproject.toml` contain the same release version.

## 2. Build and inspect distributions

```bash
python -m build
python -m twine check dist/*
```

The build must create both a `.whl` file and a `.tar.gz` source distribution.

## 3. Test the wheel in a clean environment

```bash
python -m venv wheel_test
wheel_test/Scripts/activate
python -m pip install dist/rangebranch-0.1.0a1-py3-none-any.whl
python -c "from rangebranch import AdaptiveRangeTreeClassifier"
```

Then run a small fit/predict smoke test outside the source checkout.

## 4. TestPyPI

Configure a TestPyPI trusted publisher for the repository workflow and GitHub
environment named `testpypi`. The included release workflow publishes there
when manually dispatched with `target=testpypi`.

Alternatively, for a one-time manual alpha upload:

```bash
python -m twine upload --repository testpypi dist/*
```

Install it without resolving dependencies from TestPyPI:

```bash
python -m pip install \
    --index-url https://test.pypi.org/simple/ \
    --no-deps rangebranch
```

## 5. Production PyPI

1. Confirm that the distribution name is available immediately before release.
2. Configure a PyPI pending trusted publisher for the GitHub repository,
   `.github/workflows/release.yml`, and environment `pypi`.
3. Require manual approval on the `pypi` GitHub environment.
4. Create a GitHub release whose tag matches `v<version>`.
5. The workflow tests, builds, verifies, and publishes the artifacts with OIDC.

Never commit PyPI API tokens to the repository.

## Version sequence

```text
0.1.0a1  First public alpha
0.1.0b1  API and benchmark candidate
0.1.0    Initial supported public API
0.2.0    Statistical stability estimator
0.3.0    Native categorical splits
1.0.0    Stable API and serialization policy
```
