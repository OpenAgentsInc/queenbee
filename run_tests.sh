#!/bin/bash

poetry run python -u -mcoverage run --source ai_spider -m pytest -v tests/ -k "not fine_tune"
coverage html
coverage report --fail-under 77
