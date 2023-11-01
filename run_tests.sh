#!/bin/bash -e

poetry run python -u -mcoverage run --source ai_spider -m pytest -v tests/

# these tests run ok manually
poetry run python -u -mcoverage run -a --source ai_spider -m pytest -v tests/ -k test_websocket_xx --manual 

# embedding live tests and other tests require cuda 

coverage html
coverage report --fail-under 77
