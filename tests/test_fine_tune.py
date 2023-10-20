import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from moto import mock_s3
import boto3
from ai_spider.files import app as router
from ai_spider.util import USER_BUCKET_NAME
from ai_spider.fine_tune import fine_tuning_jobs_db
from ai_spider.fine_tune import fine_tuning_events_db
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


def test_create_fine_tuning_job(tmp_path):
    fp = tmp_path / "train"
    with open(fp, "w") as fh:
        fh.write("""

""")

    response = client.post(
        "/fine_tuning/jobs",
        json={
            "model_dump": "some_dump",
            "training_file": "sample_file.txt"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"].startswith("ftjob-")
    assert data["status"] == "queued"

    # Ensure the job was added to the mock db
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
