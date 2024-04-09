#!/usr/bin/env bash
set -ex
pip install poetry
poetry install --all-extras --with test
poetry run pytest
