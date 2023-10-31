import os
import time
from base64 import urlsafe_b64encode as b64encode

from fastapi.testclient import TestClient
import logging as log

from util import set_bypass_token, mock_sock

set_bypass_token()  # noqa

from ai_spider.util import BYPASS_USER
from ai_spider.fine_tune import fine_tuning_jobs_db
from ai_spider.app import app

from tests.util import s3_server  # noqa

client = TestClient(app)
token = os.environ["BYPASS_TOKEN"]
client.headers = {"authorization": "bearer: " + token}


def test_create_fine_tuning_job(tmp_path, s3_server):
    fp = tmp_path / "train"
    with open(fp, "w") as fh:
        fh.write("""
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}
""")

    with mock_sock([
        {"status": "lora", "size": 8 * 1024 * 1024},
        {"status": "lora", "chunk": b64encode(b'fil1' * 1024 * 1024).decode()},
        {"status": "lora", "chunk": b64encode(b'fil2' * 1024 * 1024).decode()},
        {"status": "gguf", "size": 8},
        {"status": "gguf", "chunk": b64encode(b'fil1').decode()},
        {"status": "gguf", "chunk": b64encode(b'fil2').decode()},
        {"status": "done"},
        {}
    ]):
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
        assert data["id"]
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
        to = time.monotonic() + 10
        # get job
        while not done:
            response = client.get(f"/v1/fine_tuning/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json()["id"] == job_id
            done = response.json()["status"] in ("done", "error")
            time.sleep(0.1)
            if time.monotonic() > to:
                raise TimeoutError

        assert not response.json().get("error")
        assert response.json()["status"] == "done"

        response = client.get(f"/v1/fine_tuning/jobs/{job_id}/events")
        assert response.status_code == 200
        events = response.json()

        # Check the events
        assert len(events) == 2
        assert "done" in events[-1]["message"]

        response = client.get("/v1/fine_tuning/jobs/nonexistent_job_id/events")
        assert response.status_code == 404
