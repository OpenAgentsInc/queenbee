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


def test_embed(tmp_path):
    with mock_sock([
        {"object": "list",
         "usage": {"prompt_tokens": 10, "total_tokens": 10},
         "model": "name",
         "data": [{"object": "embedding", "index": 0, "embedding": [1, 2, 3]},
                  {"object": "embedding", "index": 1, "embedding": [1, 2, 3]},
                  ],
         }
    ]):
        response = client.post(
            "/v1/embeddings",
            json={
                "input": ["embedding doc 1", "embedding doc 2"],
                "model": "BAAI/bge-base-en-v1.5"
            }
        )
        log.info(response.text)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"

