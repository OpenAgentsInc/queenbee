#!/bin/bash

poetry run python -mcoverage run --source ai_spider -m pytest tests/
coverage html
coverage report --fail-under 77
