#!/bin/bash -e

poetry run python -u -mcoverage run --source ai_spider -m pytest --log-cli-level=debug -s -v tests/
poetry run python -u -mcoverage run -a --source ai_spider -m pytest --log-cli-level=debug -s -v tests/ -k test_websocket_xx --manual 
coverage html
coverage report --fail-under 77
