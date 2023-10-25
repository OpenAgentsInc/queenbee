import asyncio
import json
import logging
import os
import time
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Generator

from ai_spider.util import get_model_size, bill_usage, check_bearer_token, \
    optional_bearer_token, USER_BUCKET_NAME, schedule_task, b64dec
from ai_spider.workers import get_reg_mgr, QueueSocket, do_model_job
from ai_spider.s3 import get_s3

AWS_MINIMUM_PART_SIZE = 1024 * 1024 * 5

app = APIRouter()

log = logging.getLogger(__name__)

PUNISH_BUSY_SECS = 30

load_dotenv()


# Define request models
class CreateFineTuningJobRequest(BaseModel):
    model: str
    training_file: str
    validation_file: Optional[str] = ""
    hyperparameters: dict = {}
    suffix: str = ""
    gpu_filter: Optional[dict] = {}
    checkpoint: Optional[str] = ""


class CancelFineTuningJobResponse(BaseModel):
    id: str
    model: str
    status: str
    validation_file: str = None
    training_file: str


# Define response models
class FineTuningJobResponse(CreateFineTuningJobRequest):
    object: str = "fine_tuning.job"
    id: str
    created_at: int
    finished_at: Optional[int] = None
    fine_tuned_model: Optional[str] = None
    organization_id: Optional[str] = None
    result_files: List[str] = []
    status: str
    error: Optional[str] = None
    trained_tokens: int = None


class ListFineTuningJobsResponse(BaseModel):
    object: str = "list"
    data: List[FineTuningJobResponse]
    has_more: bool


class FineTuningEventResponse(BaseModel):
    object: str = "event"
    id: str
    created_at: int
    level: str
    message: str
    data: Optional[dict] = None
    type: str


# Dummy database to store fine-tuning jobs
fine_tuning_jobs_db = defaultdict(lambda: {})

# Dummy database to store fine-tuning events
fine_tuning_events_db = defaultdict(lambda: defaultdict(lambda: []))


async def do_fine_tune(body: CreateFineTuningJobRequest, state: dict, ws: "QueueSocket") \
        -> Generator[tuple[dict, float], None, None]:
    req = body.model_dump(mode="json")
    req["state"] = state
    async for js, job_time in do_model_job("/v1/fine_tuning/jobs", req, ws, stream=True, stream_timeout=-1):
        if not js or "status" not in js:
            raise HTTPException(status_code=500, detail="Invalid worker response")
        yield js, job_time


# Create Fine-tuning Job
@app.post("/fine_tuning/jobs", response_model=FineTuningJobResponse)
async def create_fine_tuning_job(request: Request, body: CreateFineTuningJobRequest, tasks: BackgroundTasks,
                                 user_id: str = Depends(check_bearer_token)):
    job_id = os.urandom(16).hex()

    job = fine_tuning_jobs_db[user_id][job_id] = body.model_dump(mode="json")
    job["status"] = "init"
    job["created_at"] = int(time.time())

    log.info("fine tune: new job %s", job)

    # user can specify any url here, including one with a username:token, for example
    # todo: manage access to training data, allowing only the requested files for the duration of the job
    if not body.training_file.startswith("https:"):
        body.training_file = f"https://{USER_BUCKET_NAME}.s3.amazonaws.com/{user_id}/{body.training_file}"

    # queue task
    tasks.add_task(fine_tune_task, request, body, job_id, user_id)

    fine_tuning_events_db[user_id][job_id].append(dict(
        id="ft-event-" + os.urandom(16).hex(),
        created_at=int(time.time()),
        level="info",
        message="pending worker allocation",
        type="message"
    ))

    return {**{"id": job_id, "created_at": 1692661014, "status": "queued"}, **body.model_dump(mode="json")}


async def fine_tune_task(request, body, job_id, user_id):
    s3 = await get_s3()
    gpu_filter = body.gpu_filter or {}
    msize = get_model_size(body.model)
    mgr = get_reg_mgr()
    state = {}
    gpu_filter["min_version"] = "0.2.0"
    gpu_filter["capabilities"] = ["llama-fine-tune"]
    job = fine_tuning_jobs_db[user_id][job_id]
    mbase = body.model.split("/")[-1]
    q_level = job.get("hyperparameters", {}).get("q_level", "q5_1")
    final_name = f"{user_id}/{mbase}:{q_level}:{job_id}"
    lora_key = f"{final_name}.lora.gz"
    gguf_key = f"{final_name}.gguf"
    try:
        upload = {
            "lora": {
                "key": lora_key
            },
            "gguf": {
                "key": gguf_key
            }
        }
        with mgr.get_socket_for_inference(msize, "cli", gpu_filter) as ws:
            async for js, job_time in do_fine_tune(body, state, ws):
                if job["status"] == "cancelled":
                    break
                job["status"] = js["status"]
                if js.get("error"):
                    raise HTTPException(408, detail=json.dumps(js))
                elif js["status"] in ("done",):
                    log.info("fine tune %s: %s / %s", job_id, js, job_time)

                    schedule_task(bill_usage(request, msize, {"job": "fine_tune"}, ws.info, job_time))

                if js["status"] not in ("lora", "gguf"):
                    log.info("fine tune %s: %s / %s", job_id, js, job_time)
                    fine_tuning_events_db[user_id][job_id].append(dict(
                        id="ft-event-" + os.urandom(16).hex(),
                        created_at=int(time.time()),
                        level="info",
                        message=json.dumps(js),
                        data=js,
                        type="message"
                    ))
                else:
                    chunk = js.pop("chunk", None)
                    upl = upload[js["status"]]
                    if not upl.get("id"):
                        log.info("fine tune %s: start upload: %s", job_id, upl["key"])
                        upl_id = \
                        (await s3.create_multipart_upload(Bucket=USER_BUCKET_NAME, Key=upl["key"]))[
                            'UploadId']
                        upl.update({
                            "id": upl_id,
                            "parts": [],
                            "bytes": b""
                        })

                    await process_upload_chunk(chunk, s3, upl)
                state = js

        if state.get("status") == "done":
            log.info("fine tune %s: finalize upload", job_id)
            for fil in ["lora", "gguf"]:
                await process_upload_chunk(b"", s3, upload[fil], final=True)
        elif state.get("status") not in ("error", "cancelled"):
            assert state.get("status") == "error", f"fine tune %s: error invalid end state: {state.get('status')}"

        log.info("fine tune %s: done", state)
    except HTTPException as ex:
        log.error("fine tune %s: error %s", job_id, repr(ex))
        job["status"] = "error"
        job["error"] = repr(ex)
        fine_tuning_events_db[user_id][job_id].append(dict(
            id="ft-event-" + os.urandom(16).hex(),
            created_at=int(time.time()),
            level="error",
            message=repr(ex),
            type="error"
        ))
    except (Exception, asyncio.CancelledError) as ex:
        log.exception("fine tune %s: error %s", job_id, repr(ex))
        job["status"] = "error"
        job["error"] = repr(ex)
        fine_tuning_events_db[user_id][job_id].append(dict(
            id="ft-event-" + os.urandom(16).hex(),
            created_at=int(time.time()),
            level="error",
            message=repr(ex),
            type="error"
        ))


async def process_upload_chunk(chunk, s3, upl, final=False):
    if chunk:
        try:
            upl["bytes"] += b64dec(chunk)
        except Exception as ex:
            log.error("wtf is this: %s", chunk)

    if final or len(upl["bytes"]) > AWS_MINIMUM_PART_SIZE:
        log.debug("upload min part size")
        part_num = len(upl["parts"]) + 1
        if part_num == 1 or upl["bytes"]:
            response = await s3.upload_part(
                Bucket=USER_BUCKET_NAME,
                Key=upl["key"],
                UploadId=upl["id"],
                PartNumber=part_num,
                Body=upl["bytes"]
            )
            upl["parts"].append({'PartNumber': part_num, 'ETag': response['ETag']})
            # clear out ram
            upl["bytes"] = b''

        if final:
            await s3.complete_multipart_upload(
                Bucket=USER_BUCKET_NAME,
                Key=upl["key"],
                UploadId=upl["id"],
                MultipartUpload={'Parts': upl["parts"]})


@app.get("/fine_tuning/jobs", response_model=ListFineTuningJobsResponse)
async def list_fine_tuning_jobs(after: str = None, limit: int = 20, user_id: str = Depends(check_bearer_token)):
    jobs = [
        FineTuningJobResponse(
            object="fine_tuning_job",
            id=job_id,
            model=job["model"],
            created_at=job["created_at"],
            finished_at=None,
            fine_tuned_model=None,
            organization_id="",
            result_files=[],
            status=job["status"],
            validation_file=job.get("validation_file"),
            error=job.get("error"),
            training_file=job["training_file"],
            hyperparameters=job["hyperparameters"],
            trained_tokens=0
        ) for job_id, job in
        fine_tuning_jobs_db[user_id].items()]
    return {"object": "list", "data": jobs, "has_more": False}


# Retrieve Fine-tuning Job
@app.get("/fine_tuning/jobs/{fine_tuning_job_id}", response_model=FineTuningJobResponse)
async def retrieve_fine_tuning_job(fine_tuning_job_id: str, user_id: str = Depends(check_bearer_token)):
    job = fine_tuning_jobs_db[user_id].get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    return {**{"id": fine_tuning_job_id, "created_at": 1692661014, "status": job.get("status")}, **job}


# Cancel Fine-tuning Job
@app.post("/fine_tuning/jobs/{fine_tuning_job_id}/cancel", response_model=CancelFineTuningJobResponse)
async def cancel_fine_tuning_job(fine_tuning_job_id: str, user_id: str = Depends(check_bearer_token)):
    job = fine_tuning_jobs_db[user_id].get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    if job["status"] not in ["error", "done", "cancelled"]:
        fine_tuning_events_db[user_id][fine_tuning_job_id].append(dict(
            id="ft-event-" + os.urandom(16).hex(),
            created_at=int(time.time()),
            level="error",
            message="cancelled job",
            type="message"
        ))
        job["status"] = "cancelled"
    return {**{"id": fine_tuning_job_id, "status": "cancelled"}, **job}


# List Fine-tuning Events
@app.get("/fine_tuning/jobs/{fine_tuning_job_id}/events", response_model=List[FineTuningEventResponse])
async def list_fine_tuning_events(fine_tuning_job_id: str, after: str = None, limit: int = 20,
                                  user_id: str = Depends(check_bearer_token)):
    if fine_tuning_job_id not in fine_tuning_events_db[user_id]:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    return fine_tuning_events_db[user_id][fine_tuning_job_id]


# Dummy database to store models
models_db = {}


# Model response schema
class ModelResponse(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class DeleteResponse(BaseModel):
    id: str
    object: str = "model"
    deleted: bool


class ListResponse(BaseModel):
    object: str = "list"
    data: list[ModelResponse] = []


# List Models
@app.get("/models", response_model=ListResponse)
async def list_models(user_id: str = Depends(optional_bearer_token)):
    user_folder = f"{user_id}/"
    file_objects = (await (await get_s3()).list_objects(Bucket=USER_BUCKET_NAME, Prefix=user_folder))['Contents']

    # all uploads
    uploads = [
        {"file": obj["Key"][len(user_folder):], "size": obj["Size"], "created_at": obj["LastModified"].timestamp()} for
        obj in file_objects]

    # endswith gguf ? we can probably use it
    uploads = [ent for ent in uploads if ent["file"].endswith(".gguf")]

    # user owned models, if any
    models = [ModelResponse(id=f"um:{ent['file']}", owned_by=user_id, created=ent["created_at"]) for ent in uploads]

    # web/hf default models
    models.append(ModelResponse(id="vicuna-v1-7b-q4f32_0", owned_by="hf", created=0))

    # hf only model
    models.append(ModelResponse(id="TheBloke/zephyr-7B-alpha-GGUF:Q4_K_S", owned_by="hf", created=0))
    models.append(ModelResponse(id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF:Q4_K_S", owned_by="hf", created=0))
    models.append(ModelResponse(id="TheBloke/CodeLlama-13B-Instruct-GGUF:Q4_K_S", owned_by="hf", created=0))

    return ListResponse(data=models)


@app.get("/models/{model_id}", response_model=ModelResponse)
async def retrieve_model(model_id: str, user_id: str = Depends(check_bearer_token)):
    """Get model info, including the file_id if it's a user-owned fine-tune.

    NOTE: This should never return non-public user-owned fine-tunes, unless the bearer token matches the owner.
    """
    # get fine tune
    model = models_db.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelResponse(id=model_id, owned_by="user", created=0)


@app.delete("/models/{model_id}")
async def delete_model(model_id: str, user_id: str = Depends(check_bearer_token)):
    """Remove model from the system."""
    model = models_db.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if the user has the required permissions to delete the model (role check)
    # Replace this with your actual permission logic

    # Delete the model
    del models_db[model_id]
    return DeleteResponse(id=model_id, deleted=True)
