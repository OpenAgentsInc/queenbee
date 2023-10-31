import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--manual",
        action="store_true",
        default=False,
        help="Run manual tests",
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--manual"):
        skipper = pytest.mark.skip(reason="Only run when --manual is given")
        for item in items:
            if "manual" in item.keywords:
                item.add_marker(skipper)

