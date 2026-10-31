import base64
import io
import json
from unittest.mock import MagicMock, Mock
from urllib.error import HTTPError, URLError

import pytest
from botocore.credentials import Credentials

import app


COMPLETION = {
    "id": "chatcmpl-test", "object": "chat.completion", "model": "google.gemma-4-31b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from Gemma!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
}


@pytest.fixture
def mantle(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("MODEL_ID", "google.gemma-4-31b")
    session = Mock()
    session.get_credentials.return_value = Credentials("test-access-key", "test-secret", "test-token")
    monkeypatch.setattr(app.boto3, "Session", Mock(return_value=session))
    transport = MagicMock()
    transport.return_value.__enter__.return_value = io.BytesIO(json.dumps(COMPLETION).encode())
    monkeypatch.setattr(app, "urlopen", transport)
    return transport


def event(body):
    return {"body": json.dumps(body), "isBase64Encoded": False}


def test_native_payload_and_real_sigv4_signing(mantle):
    result = app.handler(event({"prompt": "Say hello", "max_tokens": 128}), None)
    assert result["statusCode"] == 200
    assert json.loads(result["body"]) == COMPLETION
    request = mantle.call_args.args[0]
    assert request.full_url == "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
    assert json.loads(request.data) == {
        "model": "google.gemma-4-31b", "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 128, "stream": False,
    }
    assert "AWS4-HMAC-SHA256" in request.get_header("Authorization")
    assert "/us-east-1/bedrock/aws4_request" in request.get_header("Authorization")
    assert request.get_header("X-amz-security-token") == "test-token"
    assert mantle.call_args.kwargs["timeout"] == 25


def test_base64_body_and_default_tokens(mantle):
    body = base64.b64encode(json.dumps({"prompt": "Hello"}).encode()).decode()
    assert app.handler({"body": body, "isBase64Encoded": True}, None)["statusCode"] == 200
    assert json.loads(mantle.call_args.args[0].data)["max_tokens"] == 512


@pytest.mark.parametrize("body", [{}, {"prompt": " "}, {"prompt": 3}, [], None,
    {"prompt": "x" * 32001}, {"prompt": "Hello", "max_tokens": True},
    {"prompt": "Hello", "max_tokens": 0}, {"prompt": "Hello", "max_tokens": 4097}])
def test_invalid_input_never_calls_mantle(mantle, body):
    assert app.handler(event(body), None)["statusCode"] == 400
    mantle.assert_not_called()


@pytest.mark.parametrize("bad_event", [{"body": "not json"}, {"body": "!", "isBase64Encoded": True}])
def test_malformed_body(mantle, bad_event):
    assert app.handler(bad_event, None)["statusCode"] == 400
    mantle.assert_not_called()


@pytest.mark.parametrize("upstream,expected", [(403, 502), (404, 502), (429, 429), (500, 502)])
def test_upstream_errors_are_sanitized(mantle, upstream, expected):
    mantle.side_effect = HTTPError("https://example.invalid", upstream, "secret detail", {}, io.BytesIO(b"private"))
    result = app.handler(event({"prompt": "Hello"}), None)
    assert result["statusCode"] == expected
    assert json.loads(result["body"])["upstream_status"] == upstream
    assert "private" not in result["body"] and "secret detail" not in result["body"]


@pytest.mark.parametrize("error", [URLError("offline"), TimeoutError("timeout")])
def test_transport_failure(mantle, error):
    mantle.side_effect = error
    assert app.handler(event({"prompt": "Hello"}), None)["statusCode"] == 502


def test_model_and_region_override(mantle, monkeypatch):
    monkeypatch.setenv("MODEL_ID", "another-model")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert app.handler(event({"prompt": "Hello"}), None)["statusCode"] == 200
    assert json.loads(mantle.call_args.args[0].data)["model"] == "another-model"
    assert "bedrock-mantle.us-west-2.api.aws" in mantle.call_args.args[0].full_url


def test_missing_credentials(mantle, monkeypatch):
    app.boto3.Session.return_value.get_credentials.return_value = None
    assert app.handler(event({"prompt": "Hello"}), None)["statusCode"] == 502
    mantle.assert_not_called()


def test_malformed_upstream_completion(mantle):
    mantle.return_value.__enter__.return_value = io.BytesIO(b'{"choices": []}')
    assert app.handler(event({"prompt": "Hello"}), None)["statusCode"] == 502
