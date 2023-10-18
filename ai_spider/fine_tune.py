from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = APIRouter()

load_dotenv()


# Define request models
class CreateFineTuningJobRequest(BaseModel):
    model: str
    training_file: str
    hyperparameters: dict = {}
    suffix: str = ""
    validation_file: str = ""


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


# Create Fine-tuning Job
@app.post("/fine_tuning/jobs", response_model=FineTuningJobResponse)
async def create_fine_tuning_job(request: CreateFineTuningJobRequest):
    job_id = f"ftjob-{len(fine_tuning_jobs_db) + 1}"
    fine_tuning_jobs_db[job_id] = request.model_dump(mode="json")
    return {**{"id": job_id, "created_at": 1692661014, "status": "queued"}, **request.model_dump(mode="json")}


# List Fine-tuning Jobs
@app.get("/fine_tuning/jobs", response_model=ListFineTuningJobsResponse)
async def list_fine_tuning_jobs(after: str = None, limit: int = 20):
    jobs = [{"id": job_id, "created_at": 1692661014, "status": "succeeded", **job} for job_id, job in
            fine_tuning_jobs_db.items()]
    return {"data": jobs, "has_more": False}


# Retrieve Fine-tuning Job
@app.get("/fine_tuning/jobs/{fine_tuning_job_id}", response_model=FineTuningJobResponse)
async def retrieve_fine_tuning_job(fine_tuning_job_id: str):
    job = fine_tuning_jobs_db.get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    return {**{"id": fine_tuning_job_id, "created_at": 1692661014, "status": "succeeded"}, **job}


# Cancel Fine-tuning Job
@app.post("/fine_tuning/jobs/{fine_tuning_job_id}/cancel", response_model=CancelFineTuningJobResponse)
async def cancel_fine_tuning_job(fine_tuning_job_id: str):
    job = fine_tuning_jobs_db.get(fine_tuning_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")
    job["status"] = "cancelled"
    return {**{"id": fine_tuning_job_id, "status": "cancelled"}, **job}


# List Fine-tuning Events
@app.get("/fine_tuning/jobs/{fine_tuning_job_id}/events", response_model=List[FineTuningEventResponse])
async def list_fine_tuning_events(fine_tuning_job_id: str, after: str = None, limit: int = 20):
    return []


# Dummy database to store models
models_db = {}


# Model response schema
class ModelResponse(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "user"


class DeleteResponse(BaseModel):
    id: str
    object: str = "model"
    deleted: bool


class ListResponse(BaseModel):
    object: str = "list"
    data: list[ModelResponse] = []


# List Models
@app.get("/models", response_model=ListResponse)
async def list_models():
    models = [ModelResponse(id=model_id, owned_by=owned_by) for model_id, owned_by in models_db.items()]
    return ListResponse(data=models)


# Retrieve Model
@app.get("/models/{model_id}", response_model=ModelResponse)
async def retrieve_model(model_id: str):
    model = models_db.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelResponse(model_id=model_id, owned_by=model)


# Delete Fine-tuned Model
@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    model = models_db.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if the user has the required permissions to delete the model (role check)
    # Replace this with your actual permission logic

    # Delete the model
    del models_db[model_id]
    return DeleteResponse(id=model_id, deleted=True)
