import logging
import uuid
from datetime import datetime
from functools import wraps

from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import File, UploadFile, HTTPException, Depends, APIRouter, Header
from fastapi.responses import StreamingResponse, JSONResponse

from ai_spider.s3 import get_s3
from ai_spider.util import check_bearer_token, USER_BUCKET_NAME

app = APIRouter()

load_dotenv()
log = logging.getLogger(__name__)


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


@app.get("/files")
@handle_aws_exceptions
async def list_files(user_id: str = Depends(check_bearer_token)):
    s3 = await get_s3()
    user_folder = f"{user_id}/"
    file_objects = (await s3.list_objects(Bucket=USER_BUCKET_NAME, Prefix=user_folder))['Contents']
    return {"data": [
        {"id": obj["Key"][len(user_folder):], "bytes": obj["Size"], "created_at": int(obj["LastModified"].timestamp())} for
        obj in
        file_objects], "object": "list"}


@app.post("/files")
@handle_aws_exceptions
async def upload_file(file: UploadFile = File(...), purpose: str = "", user_id: str = Depends(check_bearer_token), x_request_id: str = Header(None)):

    # Generate a random x-request-id if it is not provided
    if not x_request_id:
        x_request_id = str(uuid.uuid4())

    # for now, this is a quick way of dealing with identical uploads
    file_name = f"file_{file.size}"
    user_folder = f"{user_id}/"
    await (await get_s3()).upload_fileobj(file.file, USER_BUCKET_NAME, f"{user_folder}{file_name}")
    # You may also need to store additional metadata, such as purpose, in a database
    return JSONResponse(
        content={
            "id": file_name,
            "bytes": file.size,
            "created_at": int(datetime.now().timestamp()),
            "filename": file_name,
            "object": "file",
            "purpose": purpose,
            "status": "uploaded",
            "status_details": ""
        },
        headers={"x-request-id": x_request_id}
    )


@app.delete("/files/{file_id}")
@handle_aws_exceptions
async def delete_file(file_id: str, user_id: str = Depends(check_bearer_token)):
    try:
        user_folder = f"{user_id}/"
        await (await get_s3()).delete_object(Bucket=USER_BUCKET_NAME, Key=f"{user_folder}{file_id}")
        return {"id": file_id, "object": "file", "deleted": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{file_id}")
@handle_aws_exceptions
async def retrieve_file(file_id: str, user_id: str = Depends(check_bearer_token)):
    try:
        user_folder = f"{user_id}/"
        # Retrieve file metadata from S3 and potentially other metadata from the database
        file_meta = await (await get_s3()).head_object(Bucket=USER_BUCKET_NAME, Key=f"{user_folder}{file_id}")
        return {"id": file_id, "object": "file", "bytes": file_meta["ContentLength"],
                "created_at": int(file_meta["LastModified"].timestamp()), "filename": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{file_id}/content")
@handle_aws_exceptions
async def retrieve_file_content(file_id: str, user_id: str = Depends(check_bearer_token)):
    user_folder = f"{user_id}/"
    file_content = (await (await get_s3()).get_object(Bucket=USER_BUCKET_NAME, Key=f"{user_folder}{file_id}"))['Body']
    return StreamingResponse(content=file_content, media_type="application/octet-stream")
