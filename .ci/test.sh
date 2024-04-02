#!/usr/bin/env bash
set -ex
pip install poetry
poetry install
poetry run pytest
