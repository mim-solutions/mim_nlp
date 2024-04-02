# MIM NLP

## Project goal

## Installation

## Usage

## Development

Remember to use a separate environment for each project.
Run commands below inside the project's environment.

### Dependencies

We use `poetry` for dependency management.
If you have never used it, consult
[poetry documentation](https://python-poetry.org/docs/)
for installation guidelines and basic usage instructions.

```sh
poetry install
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
