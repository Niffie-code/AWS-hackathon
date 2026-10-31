# Gemma on Bedrock Mantle

Python 3.12 Lambda with an API Gateway HTTP API (`POST /chat`) and a native
Chat Completions request to
`https://bedrock-mantle.<region>.api.aws/v1/chat/completions`.

The handler uses **boto3's credential chain and botocore SigV4 signing** with
service name `bedrock`. It sends `model`, `messages`, `max_tokens`, and `stream`
directly; it does not wrap the payload in `InvokeModel` or use Converse.
Temporary Lambda execution-role credentials include the session token.

## Model prerequisite

`google.gemma-4-31b` is the requested default, **not a verified model ID**.
Before deployment, confirm that AWS offers this exact ID through Mantle in
your chosen region and that your account has access. Otherwise pass a supported
ID as the `ModelId` SAM parameter. Mocked tests do not establish model availability,
account authorization, endpoint availability, or live signing compatibility.
The current boto3 service definitions have no native Mantle chat operation;
the handler therefore uses the signed REST endpoint.

## Local verification

From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
```

Tests use fake credentials and a mocked HTTPS transport, with real botocore
request signing. No AWS calls, API keys, or charges are required. They cover
the native payload, signed headers, successful completions, input validation,
base64 HTTP events, model/region overrides, and upstream failures.

## Deployment

Install the AWS SAM CLI and configure AWS credentials with permission to create
CloudFormation, Lambda, API Gateway, and IAM resources. From the repository root:

```sh
sam validate --region us-east-1
sam build
sam deploy --guided --region us-east-1 --parameter-overrides ModelId=google.gemma-4-31b
```

Select a region and model combination that actually supports Mantle. The template
grants the execution role `bedrock:InvokeModel` on the configured foundation model
in that region. SAM provides the Lambda logging role and API invocation permission.
Deployment creates a **public, unauthenticated** hackathon endpoint; add API
authorization before using it with private data or opening it broadly.

Use the `ChatEndpoint` stack output:

```sh
curl --fail-with-body "$CHAT_ENDPOINT" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain serverless computing in one sentence.","max_tokens":128}'
```

Success returns the native completion JSON, including `choices` and any `usage`
provided by the model. Invalid input returns 400; model throttling returns 429;
other upstream failures return 502. Upstream error bodies and prompts are not logged.
The HTTPS timeout is 25 seconds within the 30-second Lambda timeout. This is a
non-streaming endpoint intended for short hackathon responses.

AWS deployment and live model inference must be verified separately; the local
suite does not perform either action.

## LexiGemma frontend

The React dashboard runs independently with local mock data:

```sh
npm install
npm run dev
```

Use `npm run build` to verify a production build. The mock vocabulary is fixed for demonstration, regardless of the goal entered. To connect a backend, set `USE_MOCK` to `false` in `src/api.js` and provide `POST /api/generate` and `POST /api/evaluate` endpoints returning the response shapes used by `vocab_prompt.py`. These endpoints are not supplied by the existing Lambda template.
