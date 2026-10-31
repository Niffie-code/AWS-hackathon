"""Serve the LexiGemma SPA and its Bedrock-backed HTTP API via Lambda."""
import base64
import binascii
import json
import mimetypes
from pathlib import Path

from vocab_prompt import generate_vocab_response

WEB_ROOT = Path(__file__).resolve().parent / "web"


def respond(status, body, content_type="application/json; charset=utf-8", binary=False):
    return {"statusCode": status, "headers": {"content-type": content_type, "x-content-type-options": "nosniff"},
            "body": body, "isBase64Encoded": binary}


def json_response(status, data):
    return respond(status, json.dumps(data))


def handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "")
    path = event.get("rawPath", "/")
    if method == "POST" and path in ("/api/generate", "/api/evaluate"):
        try:
            body = event.get("body") or ""
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body, validate=True).decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            goal = data.get("user_goal")
            if not isinstance(goal, str) or not goal.strip():
                return json_response(400, {"error": "Please provide a goal."})
            if path == "/api/evaluate":
                sentence, words = data.get("user_input"), data.get("target_words")
                if not isinstance(sentence, str) or not sentence.strip():
                    return json_response(400, {"error": "Please provide a sentence."})
                if not isinstance(words, list) or not words or any(not isinstance(w, str) or not w.strip() for w in words):
                    return json_response(400, {"error": "Please provide the target words."})
                result = generate_vocab_response(goal.strip(), sentence.strip(), words)
            else:
                result = generate_vocab_response(goal.strip())
            if "error" in result:
                # Avoid exposing model replies or internal AWS error details to the public.
                return json_response(502, {"error": "The vocabulary coach is unavailable. Please try again."})
            return json_response(200, result)
        except (ValueError, TypeError, UnicodeError, binascii.Error):
            return json_response(400, {"error": "Please provide valid JSON."})
    if method == "GET" and (path == "/" or path.startswith("/assets/")):
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if not target.is_relative_to(WEB_ROOT.resolve()) or not target.is_file():
            return json_response(404, {"error": "Not found."})
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return respond(200, base64.b64encode(target.read_bytes()).decode("ascii"), content_type, True)
    return json_response(404, {"error": "Not found."})
