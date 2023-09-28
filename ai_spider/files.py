import os
from functools import wraps

import httpx
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import File, UploadFile, HTTPException, Depends, Request, APIRouter
from fastapi.responses import StreamingResponse
import boto3

from ai_spider.util import get_bill_to, BILLING_URL

app = APIRouter()

load_dotenv()

g_s3 = None


def s3():
    global g_s3
    if not g_s3:
        g_s3 = boto3.client('s3',
                            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))

    return g_s3

bucket_name = 'gputopia-user-files'


def handle_aws_exceptions(route_function):
    @wraps(route_function)
    async def wrapper(*args, **kwargs):
        try:
            return await route_function(*args, **kwargs)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == "NoSuchKey":
                raise HTTPException(status_code=404, detail=str(e))
            else:
                raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return wrapper

async def check_bearer_token(request: Request) -> str:
    bill_to_token = get_bill_to(request)

    if bill_to_token == os.environ.get("BYPASS_TOKEN"):
        return "bypass"

    command = dict(
        command="check",
        bill_to_token=bill_to_token,
    )

    try:
        res = httpx.post(BILLING_URL, json=command, timeout=10)
    except Exception as ex:
        raise HTTPException(status_code=500, detail="billing endpoint error: %s" % ex)

    js = res.json()

    if js.get("user_id"):
        return js.get("user_id")

    raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/v1/files")
@handle_aws_exceptions
async def list_files(user_id: str = Depends(check_bearer_token)):
    user_folder = f"{user_id}/"
    file_objects = s3().list_objects(Bucket=bucket_name, Prefix=user_folder)['Contents']
    return {"data": [{"id": obj["Key"][len(user_folder):], "bytes": obj["Size"], "created_at": obj["LastModified"].timestamp()} for obj in
                     file_objects], "object": "list"}


@app.post("/v1/files")
@handle_aws_exceptions
async def upload_file(file: UploadFile = File(...), purpose: str = "", user_id: str = Depends(check_bearer_token)):
    user_folder = f"{user_id}/"
    s3().upload_fileobj(file.file, bucket_name, f"{user_folder}{file.filename}")
    # You may also need to store additional metadata, such as purpose, in a database
    return {"id": file.filename, "object": "file", "bytes": file.size, "filename": file.filename, "purpose": purpose,
            "status": "uploaded"}


@app.delete("/v1/files/{file_id}")
@handle_aws_exceptions
async def delete_file(file_id: str, user_id: str = Depends(check_bearer_token)):
    try:
        user_folder = f"{user_id}/"
        s3().delete_object(Bucket=bucket_name, Key=f"{user_folder}{file_id}")
        return {"id": file_id, "object": "file", "deleted": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files/{file_id}")
@handle_aws_exceptions
async def retrieve_file(file_id: str, user_id: str = Depends(check_bearer_token)):
    try:
        user_folder = f"{user_id}/"
        # Retrieve file metadata from S3 and potentially other metadata from the database
        file_meta = s3().head_object(Bucket=bucket_name, Key=f"{user_folder}{file_id}")
        return {"id": file_id, "object": "file", "bytes": file_meta["ContentLength"],
                "created_at": file_meta["LastModified"].timestamp(), "filename": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files/{file_id}/content")
@handle_aws_exceptions
async def retrieve_file_content(file_id: str, user_id: str = Depends(check_bearer_token)):
    user_folder = f"{user_id}/"
    file_content = s3().get_object(Bucket=bucket_name, Key=f"{user_folder}{file_id}")['Body']
    return StreamingResponse(content=file_content, media_type="application/octet-stream")
