#!/bin/bash

python -m coverage run --source ai_spider -m pytest tests/
coverage html
coverage report --fail-under 77
