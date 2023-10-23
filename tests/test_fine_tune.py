import asyncio
import contextlib
import os
import unittest.mock

import pytest
from fastapi.testclient import TestClient
from moto import mock_s3
import boto3
import logging as log

from util import set_bypass_token

set_bypass_token()  # noqa

from ai_spider.util import USER_BUCKET_NAME, BYPASS_USER
from ai_spider.fine_tune import fine_tuning_jobs_db
from ai_spider.workers import get_reg_mgr
from ai_spider.app import app

from tests.util import s3_server # noqa

client = TestClient(app)
token = os.environ["BYPASS_TOKEN"]
client.headers = {"authorization": "bearer: " + token}



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
def mock_sock(predef=[{}]):
    # You can adjust these predefined responses to simulate different scenarios.
    def mock_get_socket_for_inference(*args, **kwargs):
        return MockQueueSocket(predef)

    mgr = get_reg_mgr()
    with unittest.mock.patch.object(mgr, 'get_socket_for_inference', mock_get_socket_for_inference):
        yield


def test_create_fine_tuning_job(tmp_path, s3_server):
    fp = tmp_path / "train"
    with open(fp, "w") as fh:
        fh.write("""
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}
""")

    with mock_sock([{"status": "done"}, {}]):
        response = client.post(
            "/v1/fine_tuning/jobs",
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.1",
                "training_file": str(fp)
            }
        )
        log.info(response.text)
        assert response.status_code == 200
        data = response.json()
        job_id = data["id"]
        assert data["id"].startswith("ftjob-")
        assert data["status"] == "queued"
        assert data["id"] in fine_tuning_jobs_db[BYPASS_USER]

        response = client.get("/v1/fine_tuning/jobs")
        assert response.status_code == 200
        assert isinstance(response.json()["data"], list)
        assert not response.json()["has_more"]  # No more jobs since we only added one

        # bad id
        response = client.get("/v1/fine_tuning/jobs/nonexistent_job_id")
        assert response.status_code == 404

        done = False
        # get job
        while not done:
            response = client.get(f"/v1/fine_tuning/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json()["id"] == job_id
            done = response.json()["status"] in ("done", "error")

        assert not response.json().get("error")
        assert response.json()["status"] == "done"

        # We're using "ftjob-1" as it's already in our mock database
        response = client.get("/v1/fine_tuning/jobs/ftjob-1/events")
        assert response.status_code == 200
        events = response.json()

        # Check the events
        assert len(events) == 2
        assert "done" in events[-1]["message"]

        response = client.get("/v1/fine_tuning/jobs/nonexistent_job_id/events")
        assert response.status_code == 404
