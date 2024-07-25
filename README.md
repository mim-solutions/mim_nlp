# MIM NLP
With this package you can easily use pre-trained models and fine-tune them,
as well as create and train your own neural networks.

Below, we list NLP tasks and models that are available:
  * Classification
    * Neural Network
    * SVM
  * Regression
    * Neural Network
  * Seq2Seq
    * Summarization (Neural Network)

It comes with utilities for text pre-processing such as:
  * Text cleaning
  * Lemmatization
  * Deduplication

## Installation
We recommend installing with pip.
```sh
pip install mim-nlp
```

The package comes with the following extras (optional dependencies for given modules):
- `svm` - simple svm model for classification
- `classifier` - classification models: svm, neural networks
- `regressor` - regression models
- `preprocessing` - cleaning text, lemmatization and deduplication
- `seq2seq` - `Seq2Seq` and `Summarizer` models

## Usage
Examples can be found in the [notebooks directory](/notebooks).

### Model classes
* `classifier.nn.NNClassifier` - Neural Network Classifier
* `classifier.svm.SVMClassifier` - Support Vector Machine Classifier
* `classifier.svm.SVMClassifierWithFeatureSelection` - `SVMClassifier` with additional feature selection step
* `regressor.AutoRegressor` - regressor based on transformers' Auto Classes
* `regressor.NNRegressor` - Neural Network Regressor
* `seq2seq.AutoSummarizer` - summarizer based on transformers' Auto Classes

### Interface
All the model classes have common interface:
* `fit`
* `predict`
* `save`
* `load`

and specific additional methods.

### Text pre-processing
* `preprocessing.TextCleaner` - define a pipeline for text cleaning, supports concurrent processesing
* `preprocessing.lemmatize` - lemmatize text in Polish with Morfeusz
* `preprocessing.Deduplicator` - find near-duplicate texts (depending on `threshold`) with Jaccard index for n-grams

## Development
Remember to use a separate environment for each project.
Run commands below inside the project's environment.

### Dependencies
We use `poetry` for dependency management.
If you have never used it, consult
[poetry documentation](https://python-poetry.org/docs/)
for installation guidelines and basic usage instructions.

```sh
poetry install --with dev
```

To fix the `Failed to unlock the collection!` error or stuck packages installation, execute the below command:
```sh
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

### Git hooks
We use `pre-commit` for git hook management.
If you have never used it, consult
[pre-commit documentation](https://pre-commit.com/)
for installation guidelines and basic usage instructions.
```sh
pre-commit install
```

There are two hooks available:
* _isort_ – runs `isort` for both `.py` files and notebooks.
Fails if any changes are made, so you have to run `git add` and `git commit` once again.
* _Strip notebooks_ – produces _stripped_ versions of notebooks in `stripped` directory.

### Tests
```sh
pytest
```

### Linting
We use `isort` and `flake8` along with `nbqa` to ensure code quality.
The appropriate options are set in configuration files.
You can run them with:
```sh
isort .
nbqa isort notebooks
```
and
```sh
flake8 .
nbqa flake8 notebooks --nbqa-shell
```

### Code formatting
You can run black to format code (including notebooks):
```sh
black .
```

### New version release
In order to add the next version of the package to PyPI, do the following steps:
- First, increment the package version in `pyproject.toml`.
- Then build the new version: run `poetry build` in the root directory.
- Finally, upload to PyPI: `poetry publish` (two newly created files).
  - If you get `Invalid or non-existent authentication information.` error,
    add PyPI token to poetry: `poetry config pypi-token.pypi <my-token>`.
