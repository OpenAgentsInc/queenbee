# GPUtopia Queenbee

Allows multiple AI workers to register and make their services available.

Routes user requests to workers.

User requests are `/v1/embeddings`, `/v1/chat/completion` , with openai syntax.

Models must be in the format:  `hf-user/hf-repo:filter`

Only GGML/GGUF supported for now.

Registration as a worker is "open to all", no need to sign up.  The first job will always be free (proof of work).


Workflow documented here:

https://github.com/ArcadeLabsInc/workerbee/wiki

## Tech stack
- Python
- FastAPI

## Statistics Collection

 - We use an EMA to track worker performance as seconds-per-token, with weighted input and completion tokens.
 - These EMA values are stored for each incremental step $in model size.   (7b model completion stats are stored separate from 13b, for example)
 - TODO: Values for image inference will be stored under a separate key.
 - The set of EMA values are serialized along with the worker's identifier, gpu count and # of inferences performed 
 - The actual identifier is never exposed via an endpoint
 - A user who knows one or more of their worker's identifiers can query for them specifically
 - Code for managing live and serialized EMA values are located in ai_spider/stats.py
 - Serialization is used to coordinate live performance tracking across a cluster of queenbee instances

## Running locally

Create a .env file or edit env vars:

```
SECRET_KEY=<random, 32-byte, 64-character hex string>
BILLING_URL=<url for billing endpoint, can use http://localhost:3000/api/worker, for example to hit the local.   can be a fake url>
BYPASS_TOKEN=<random, 32-byte, 16-character hex string, use this and a bearer token to ignore the billing url mechanism> 
AWS_ACCESS_KEY_ID=<your s3 bucket key id for fine-tune uploads>
AWS_SECRET_ACCESS_KEY=<your s3 bucket secret key for fine-tune uploads>
AWS_USER_BUCKET=<your s3 bucket name for fine-tune uploads>
```

Clone repo and run

```
git clone git@github.com:ArcadeLabsInc/queenbee.git
cd queenbee
poetry install
poetry run uvicorn ai_spider:app --reload --log-config logging.conf
```


## End to end testing

 - Follow the "running locally:
 - Install a "workerbee" locally https://github.com/ArcadeLabsInc/workerbee
 - Run the workerbee with `--queen_url=ws://127.0.0.1:8000/worker`
 - Use the api.
```
openai.api_key = "8e93a0836dcc39f8d600df04b1be61c1"   # local bypass token
openai.api_base = "http://127.0.0.1:8000/v1"
res = openai.ChatCompletion.create(
  model="vicuna-v1-7b-q4f32_0",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write one sentence about a silly frog."},
    ],
  max_tokens=200,
)
print(res.choices[0].message['content'].strip())
```

## Running tests

To run all tests, run this command after `poetry install`:

```
BILLING_URL=https://gputopia.ai/api/worker SECRET_KEY=asdfasdfasdfasdf poetry run pytest
```


## Fine tuning 

The openai endpoint has the same specs as openai `/v1/fine_tuning/jobs`.

Fine tuning jobs can take a long time, only work with linux, cuda workers, and we require larger amounts of NVRAM.

Status updates arrive via websockets every 25 training steps.   

Checkpoints are saved to the associated S3 (or other configured storage) bucket if available.

Info about LORA hyperparameters is found in [[FINETUNE.md]].
