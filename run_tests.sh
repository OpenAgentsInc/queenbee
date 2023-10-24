#!/bin/bash

poetry run python -mcoverage run --source ai_spider -m pytest tests/ -l "not fine_tune"
coverage html
coverage report --fail-under 77
