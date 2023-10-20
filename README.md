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

```
git clone git@github.com:ArcadeLabsInc/queenbee.git
cd queenbee
poetry install
poetry run uvicorn ai_spider:app --reload --log-config logging.conf
```

## Running tests

To run all tests, run this command after `poetry install`:

```
BILLING_URL=https://gputopia.ai/api/worker SECRET_KEY=asdfasdfasdfasdf poetry run pytest
```
