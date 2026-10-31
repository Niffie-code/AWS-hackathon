# LexiGemma

## Vercel deployment with Gemma (no AWS)

Import this repository into Vercel. In the project settings, add `GEMINI_API_KEY` as a sensitive environment variable for the environments you deploy. Obtain the key from Google AI Studio, restrict it to the Generative Language API, and set usage limits. Optionally set `GEMMA_MODEL_ID` to a Gemma model available to your key; the default is `gemma-3-27b-it`. Verify the model ID and availability in Google AI Studio. Redeploy after changing environment variables. Do not add the key to source control or client-side `VITE_` variables.

Vercel builds the frontend with `VITE_USE_MOCK=false`; `/api/generate` and `/api/evaluate` run Python serverless functions and call the Google Generative Language API on the server. The app still defaults to mock data when run locally. Live calls require model access, quota and a working key; no live inference has been verified. The public API has no authentication or rate limiting, so protect it and set limits before sharing to avoid unwanted use and charges. Vercel function timeouts and Google model availability depend on your account and plan.

## Optional legacy AWS deployment

LexiGemma is a React vocabulary coach backed by AWS Bedrock Converse. The SAM deployment serves the built frontend and the generation/evaluation APIs from one Lambda and API Gateway HTTP API. The deployed frontend calls the APIs on the same origin.

## Run locally with mock data

Requires Node.js 20.19+ or 22.12+ and npm.

```sh
npm ci
npm run dev
```

The local dashboard defaults to mock data. The mock dataset is fixed regardless of the goal. The mock and backend tests need no AWS credentials:

```sh
python vocab_prompt.py
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Deploy to AWS

Prerequisites: Node.js and npm, Python 3.12, AWS CLI, AWS SAM CLI, and an AWS identity permitted to deploy CloudFormation, Lambda, API Gateway, IAM and S3 resources and invoke the selected Bedrock model. Configure credentials using your organization's AWS CLI login method and confirm the target account with `aws sts get-caller-identity`.

1. In the Bedrock console, confirm the **exact model ID**, region, account access, and support for the **Converse API**. The default `google.gemma-4-31b` is a placeholder requested for this project; its availability and compatibility have not been verified. The template's IAM policy is scoped to a regional foundation-model ARN; inference profiles or other model resource types require an appropriate policy change.
2. From the repository root, run:

   ```sh
   ./scripts/prepare-deployment.sh
   sam validate --lint --region YOUR_REGION
   sam build
   sam deploy --guided --region YOUR_REGION --parameter-overrides ModelId=YOUR_MODEL_ID
   ```

   The preparation script installs npm dependencies using the lockfile, builds the frontend with `VITE_USE_MOCK=false`, and copies it and the Python handler to `.deploy/`. SAM packages the Python requirements. Re-run the script and `sam build` whenever frontend or backend code changes. The build requires a local Python 3.12 interpreter.
3. Open the `WebUrl` CloudFormation output. Enter a goal and generate vocabulary; select a word and submit a sentence for evaluation. Both requests call Bedrock through the Lambda execution role. Check Lambda CloudWatch logs if a request fails.

**The URL is public and has no authentication or rate limiting.** Anyone who can reach it can invoke the model and incur charges. Add an authorizer and appropriate usage controls before sharing it broadly. API Gateway and Lambda impose request and response size limits; this deployment is intended for a small hackathon app, not large static assets. Its API request timeout is bounded by API Gateway and the Lambda's 29-second timeout; long model responses may fail.

AWS deployment and live inference have not been tested in this repository's development environment. The frontend returns a generic error on model failure; check logs and AWS configuration without exposing private prompts or credentials.

## Structure

- `vocab_prompt.py`: generation/evaluation prompts, Converse client, response parsing and offline mock tests.
- `web_handler.py`: API routes and static-file serving for API Gateway HTTP API events.
- `src/api.js`: mock API for local development; `VITE_USE_MOCK=false` selects real same-origin requests at build time.
- `src/App.jsx`, `src/style.css`, `src/mockData.js`: dashboard and mock data.
- `template.yaml`: Lambda, API Gateway and Bedrock IAM policy.

The older `src/app.py` Mantle chat handler remains in the repository for reference; the current template deploys `web_handler.py` and uses Converse.
