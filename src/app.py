"""API Gateway HTTP API -> Bedrock Mantle native Chat Completions."""

import base64
import binascii
import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError

logger = logging.getLogger(__name__)


def chat_completion(prompt, max_tokens):
    """Use boto3 credentials to sign Mantle's OpenAI-compatible REST request.

    Mantle is a separate endpoint, not bedrock-runtime.invoke_model/Converse.
    boto3 currently has no native Mantle chat-completions operation.
    """
    session = boto3.Session()
    region = os.environ.get("AWS_REGION") or session.region_name or "us-east-1"
    endpoint = f"https://bedrock-mantle.{region}.api.aws/v1/chat/completions"
    payload = {
        "model": os.environ.get("MODEL_ID", "google.gemma-4-31b"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials are unavailable")
    signed = AWSRequest(
        method="POST", url=endpoint, data=data,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(credentials.get_frozen_credentials(), "bedrock", region).add_auth(signed)
    request = Request(endpoint, data=data, headers=dict(signed.headers), method="POST")
    with urlopen(request, timeout=25) as response:
        result = json.load(response)
    # Do not report a successful generation for a malformed upstream response.
    if not isinstance(result, dict) or not result.get("choices"):
        raise ValueError("Missing chat completion choices")
    return result


def _response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event, context):
    """Accept {prompt: str, max_tokens?: int} through an HTTP API proxy event."""
    try:
        raw = event.get("body", "")
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw, validate=True).decode("utf-8")
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("Expected an object")
        prompt = body.get("prompt")
        max_tokens = body.get("max_tokens", 512)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
            raise ValueError("Invalid prompt")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("Invalid max_tokens")
    except (ValueError, TypeError, UnicodeError, binascii.Error):
        return _response(400, {"error": "Provide a nonempty prompt (up to 32000 characters) and max_tokens from 1 to 4096."})

    try:
        return _response(200, chat_completion(prompt, max_tokens))
    except HTTPError as exc:
        # Do not return upstream bodies, which may contain sensitive details.
        logger.warning("Mantle returned HTTP %s", exc.code)
        status = 429 if exc.code == 429 else 502
        return _response(status, {"error": "Model request failed", "upstream_status": exc.code})
    except (URLError, TimeoutError, BotoCoreError, RuntimeError, ValueError, OSError):
        logger.warning("Mantle request failed")
        return _response(502, {"error": "Model service unavailable"})
