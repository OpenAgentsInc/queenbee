import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from moto import mock_s3
import boto3
from ai_spider.files import app as router
from ai_spider.util import USER_BUCKET_NAME
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


def test_file_operations(s3_client):
    # Upload a file
    token = os.environ["BYPASS_TOKEN"]
    headers = {"authorization": "bearer: " + token}

    response = client.post("/v1/files", files={"file": ("test_file.txt", "some content")}, data={"purpose": "fine-tune"}, headers=headers)
    assert response.status_code == 200
    file_id = response.json()["id"]

    # List files and check if the uploaded file is listed
    response = client.get("/v1/files", headers=headers)
    assert response.status_code == 200
    assert any(f["id"] == file_id for f in response.json()["data"])

    # Get the content of the uploaded file
    response = client.get(f"/v1/files/{file_id}/content", headers=headers)
    assert response.status_code == 200
    assert response.content == b"some content"

    # Delete the uploaded file
    response = client.delete(f"/v1/files/{file_id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["deleted"] == True

    # Check if the file is deleted by trying to fetch its content
    response = client.get(f"/v1/files/{file_id}/content", headers=headers)
    assert response.status_code == 404
