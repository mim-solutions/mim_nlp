#!/usr/bin/env bash
set -ex
pip install poetry
poetry install --only lint
poetry run isort . --check-only
poetry run nbqa isort notebooks --check-only
