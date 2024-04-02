#!/usr/bin/env bash
set -ex
pip install poetry
poetry install --only lint
poetry run flake8 .
poetry run nbqa flake8 notebooks --nbqa-shell
