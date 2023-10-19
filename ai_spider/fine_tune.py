import asyncio
import logging

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, WebSocketDisconnect, Request, Depends
from pydantic import BaseModel
from typing import List, Optional, Generator

from ai_spider.util import get_model_size, bill_usage, check_bearer_token, USER_BUCKET_NAME
from ai_spider.workers import get_reg_mgr, QueueSocket, do_model_job

app = APIRouter()

log = logging.getLogger(__name__)

PUNISH_BUSY_SECS = 30

load_dotenv()


# Define request models
class CreateFineTuningJobRequest(BaseModel):
    model: str
    training_file: str
    hyperparameters: dict = {}
    suffix: str = ""
    validation_file: str = ""
    gpu_filter: Optional[dict] = {}
    checkpoint: Optional[str] = ""

class CancelFineTuningJobResponse(BaseModel):
    id: str
    model: str
    status: str
    validation_file: str = None
    training_file: str


# Define response models
class FineTuningJobResponse(BaseModel):
    object: str = "fine_tuning.job"
    id: str
    model: str
    created_at: int
    finished_at: int = None
    fine_tuned_model: str = None
    organization_id: Optional[str] = ""
    result_files: List[str] = []
    status: str
    validation_file: str = None
    training_file: str
    hyperparameters: dict
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


# Dummy database to store fine-tuning jobs
fine_tuning_jobs_db = {}

# Dummy database to store fine-tuning events
fine_tuning_events_db = {}


async def do_fine_tune(body: CreateFineTuningJobRequest, state: dict, ws: "QueueSocket") -> Generator[tuple[dict, float]]:
    req = body.model_dump(mode="json")
    req["state"] = state
    async for js, job_time in do_model_job("/v1/fine_tuning/jobs", req, ws, stream=True):
        if "status" not in js:
            raise HTTPException(status_code=500, detail="Invalid worker response")
        yield js, job_time


# Create Fine-tuning Job
@app.post("/fine_tuning/jobs", response_model=FineTuningJobResponse)
async def create_fine_tuning_job(request: Request, body: CreateFineTuningJobRequest, user_id: str = Depends(check_bearer_token)):
    job_id = f"ftjob-{len(fine_tuning_jobs_db) + 1}"
    fine_tuning_jobs_db[job_id] = body.model_dump(mode="json")

    # user can specify any url here, including one with a username:token, for example
    # todo: manage access to training data, allowing only the requested files for the duration of the job
    if not body.training_file.startswith("https:"):
        body.training_file = f"https://{USER_BUCKET_NAME}.s3.amazonaws.com/{user_id}/{body.training_file}"

    gpu_filter = body.gpu_filter or {}
    msize = get_model_size(body.model)
    mgr = get_reg_mgr()
    state = {}
    gpu_filter["min_version"] = "0.2.0"
    try:
        while "done" not in state:
            try:
                with mgr.get_socket_for_inference(msize, "cli", gpu_filter) as ws:
                    async for js, job_time in do_fine_tune(body, state, ws):
                        log.info("fine tune: %s / %s", js, job_time)
                        if js["status"] in ("done", "checkpoint"):
                            asyncio.create_task(bill_usage(request, msize, {"job": "fine_tune"}, ws.info, job_time))
                        state = js
            except WebSocketDisconnect:
                pass
    except HTTPException as ex:
        log.error("inference failed : %s", repr(ex))
        raise
    except TimeoutError as ex:
        log.error("inference failed : %s", repr(ex))
        raise HTTPException(408, detail=repr(ex))
    except AssertionError as ex:
        log.error("inference failed : %s", repr(ex))
        raise HTTPException(400, detail=repr(ex))
    except Exception as ex:
        log.exception("unknown error : %s", repr(ex))
        raise HTTPException(500, detail=repr(ex))

    return {**{"id": job_id, "created_at": 1692661014, "status": "queued"}, **body.model_dump(mode="json")}


# List Fine-tuning Jobs
@app.get("/fine_tuning/jobs", response_model=ListFineTuningJobsResponse)
async def list_fine_tuning_jobs(after: str = None, limit: int = 20, user_id: str = Depends(check_bearer_token)):
    jobs = [{"id": job_id, "created_at": 1692661014, "status": "succeeded", **job} for job_id, job in
            fine_tuning_jobs_db.items()]
    return {"data": jobs, "has_more": False}


# Retrieve Fine-tuning Job
@app.get("/fine_tuning/jobs/{fine_tuning_job_id}", response_model=FineTuningJobResponse)
async def retrieve_fine_tuning_job(fine_tuning_job_id: str, user_id: str = Depends(check_bearer_token)):
    job = fine_tuning_jobs_db.get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    return {**{"id": fine_tuning_job_id, "created_at": 1692661014, "status": "succeeded"}, **job}


# Cancel Fine-tuning Job
@app.post("/fine_tuning/jobs/{fine_tuning_job_id}/cancel", response_model=CancelFineTuningJobResponse)
async def cancel_fine_tuning_job(fine_tuning_job_id: str, user_id: str = Depends(check_bearer_token)):
    job = fine_tuning_jobs_db.get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    job["status"] = "cancelled"
    return {**{"id": fine_tuning_job_id, "status": "cancelled"}, **job}


# List Fine-tuning Events
@app.get("/fine_tuning/jobs/{fine_tuning_job_id}/events", response_model=List[FineTuningEventResponse])
async def list_fine_tuning_events(fine_tuning_job_id: str, after: str = None, limit: int = 20, user_id: str = Depends(check_bearer_token)):
    return []


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
async def list_models(user_id: str = Depends(check_bearer_token)):
    # user fine-tunes
    models = [ModelResponse(id=model_id, owned_by="user", created=0) for model_id, owned_by in models_db.items()]

    # web/hf models
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
