import asyncio
import contextlib
import unittest.mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from moto import mock_s3
import boto3
from ai_spider.files import app as router
from ai_spider.util import USER_BUCKET_NAME
from ai_spider.fine_tune import fine_tuning_jobs_db
from ai_spider.fine_tune import fine_tuning_events_db
from ai_spider.workers import get_reg_mgr
from util import set_bypass_token

set_bypass_token()
app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    return {
        "aws_access_key_id": "testing",
        "aws_secret_access_key": "testing",
        "aws_session_token": "testing",
    }


@pytest.fixture
def s3_client(aws_credentials):
    with mock_s3():
        cli = boto3.client('s3', **aws_credentials)
        cli.create_bucket(Bucket=USER_BUCKET_NAME)
        yield cli


class MockQueueSocket:
    def __init__(self, predefined_responses):
        self.queue = asyncio.Queue()
        for resp in predefined_responses:
            self.queue.put_nowait(resp)
        self.results = self.queue
        self.info = {}  # If ws.info is used anywhere, provide a mock value

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@contextlib.contextmanager
def mock_sock(predef=[{}]):
    # You can adjust these predefined responses to simulate different scenarios.
    def mock_get_socket_for_inference(*args, **kwargs):
        return MockQueueSocket(predef)

    mgr = get_reg_mgr()
    with unittest.mock.patch.object(mgr, 'get_socket_for_inference', mock_get_socket_for_inference):
        yield


def test_create_fine_tuning_job(tmp_path):
    fp = tmp_path / "train"
    with open(fp, "w") as fh:
        fh.write("""
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}
""")

    with mock_sock([{"status": "done"}, {}]):
        response = client.post(
            "/fine_tuning/jobs",
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.1",
                "training_file": str(fp)
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"].startswith("ftjob-")
        assert data["status"] == "queued"
        assert data["id"] in fine_tuning_jobs_db["valid_user"]


def test_list_fine_tuning_jobs():
    response = client.get("/fine_tuning/jobs")
    assert response.status_code == 200
    assert isinstance(response.json()["data"], list)
    assert not response.json()["has_more"]  # No more jobs since we only added one


def test_retrieve_existing_fine_tuning_job():
    # Create a job to retrieve later
    response = client.post(
        "/fine_tuning/jobs",
        json={
            "model_dump": "some_dump",
            "training_file": "sample_file.txt"
        }
    )
    job_id = response.json()["id"]

    # Retrieve the created job
    response = client.get(f"/fine_tuning/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["id"] == job_id


def test_retrieve_nonexistent_fine_tuning_job():
    response = client.get("/fine_tuning/jobs/nonexistent_job_id")
    assert response.status_code == 404


def test_list_fine_tuning_events_for_existing_job():
    # We're using "ftjob-1" as it's already in our mock database
    response = client.get("/fine_tuning/jobs/ftjob-1/events")
    assert response.status_code == 200
    events = response.json()

    # Check the events
    assert len(events) == 2
    assert events[0]["event_id"] == "event1"
    assert events[1]["event_id"] == "event2"


def test_list_fine_tuning_events_for_nonexistent_job():
    response = client.get("/fine_tuning/jobs/nonexistent_job_id/events")
    assert response.status_code == 404
