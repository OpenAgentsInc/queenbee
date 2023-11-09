import os

from fastapi.testclient import TestClient
import logging as log

from util import set_bypass_token

set_bypass_token()  # noqa

from ai_spider.app import app

from tests.util import s3_server, mock_sock  # noqa

client = TestClient(app)
token = os.environ["BYPASS_TOKEN"]
client.headers = {"authorization": "bearer: " + token}


def test_imagegen(tmp_path):
    with mock_sock([
        {"object": "list",
         "created": 234,
         "data": [{"b64_json": "skjhsdfjkhsd"},
                  {"b64_json": "kdjfhdskjfh"},
                  ]
         }
    ]):
        response = client.post(
            "/v1/images/generations",
            json={
                "prompt": "picture of a baby seal",
                "model": "stabilityai/stable-diffusion-xl-base-1.0"
        }
        )
        log.info(response.text)
        assert response.status_code == 200
        data = response.json()
        assert data["created"]
        assert data["data"]
        assert data["data"][0]["b64_json"]

