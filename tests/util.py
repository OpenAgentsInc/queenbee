import asyncio
import contextlib
import os
import unittest.mock

import pytest

from ai_spider.s3 import get_s3
from ai_spider.util import USER_BUCKET_NAME
from ai_spider.workers import get_reg_mgr


def set_bypass_token():
    if "BYPASS_TOKEN" not in os.environ:
        os.environ.setdefault("BYPASS_TOKEN", "ABCDE12345")


@pytest.fixture
async def s3_server():
    from moto.server import ThreadedMotoServer
    server = ThreadedMotoServer(port=9736)
    server.start()
    os.environ["AWS_ENDPOINT_URL"] = "http://127.0.0.1:9736"

    s3 = await get_s3()
    await s3.create_bucket(Bucket=USER_BUCKET_NAME)
    yield

    del os.environ["AWS_ENDPOINT_URL"]
    server.stop()


class MockQueueSocket:
    def __init__(self, predefined_responses):
        self.queue = asyncio.Queue()
        self.results = asyncio.Queue()
        for resp in predefined_responses:
            self.results.put_nowait(resp)
        self.info = {}  # If ws.info is used anywhere, provide a mock value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@contextlib.contextmanager
def mock_sock(predef: list[dict]=[{}]):
    # You can adjust these predefined responses to simulate different scenarios.
    def mock_get_socket_for_inference(*args, **kwargs):
        return MockQueueSocket(predef)

    mgr = get_reg_mgr()
    with unittest.mock.patch.object(mgr, 'get_socket_for_inference', mock_get_socket_for_inference):
        yield
