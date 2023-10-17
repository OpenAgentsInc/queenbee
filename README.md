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
