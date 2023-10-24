import os

import pytest

from ai_spider.s3 import get_s3
from ai_spider.util import USER_BUCKET_NAME


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
