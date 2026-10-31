import base64
import json
from pathlib import Path
from unittest.mock import Mock

import web_handler


def event(method, path, body=None, encoded=False):
    return {"requestContext": {"http": {"method": method}}, "rawPath": path,
            "body": body, "isBase64Encoded": encoded}


def test_generation_and_evaluation_route_to_coach(monkeypatch):
    coach = Mock(side_effect=[{"words": [1, 2, 3], "mini_story": "story"}, {"score": 91, "feedback": "Good"}])
    monkeypatch.setattr(web_handler, "generate_vocab_response", coach)
    result = web_handler.handler(event("POST", "/api/generate", json.dumps({"user_goal": " IELTS "})), None)
    assert result["statusCode"] == 200 and len(json.loads(result["body"])["words"]) == 3
    coach.assert_called_once_with("IELTS")
    payload = base64.b64encode(json.dumps({"user_goal": "IELTS", "user_input": "A sentence.", "target_words": ["mitigate"]}).encode()).decode()
    result = web_handler.handler(event("POST", "/api/evaluate", payload, True), None)
    assert result["statusCode"] == 200 and json.loads(result["body"])["score"] == 91
    coach.assert_called_with("IELTS", "A sentence.", ["mitigate"])


def test_bad_requests_and_model_errors(monkeypatch):
    coach = Mock(return_value={"error": "secret AWS detail", "raw_text": "private model output"})
    monkeypatch.setattr(web_handler, "generate_vocab_response", coach)
    for path, body in [("/api/generate", "not json"), ("/api/generate", '{}'),
                       ("/api/evaluate", '{"user_goal":"goal", "user_input":"text"}')]:
        assert web_handler.handler(event("POST", path, body), None)["statusCode"] == 400
    coach.assert_not_called()
    result = web_handler.handler(event("POST", "/api/generate", '{"user_goal":"goal"}'), None)
    assert result["statusCode"] == 502
    assert "secret" not in result["body"] and "private" not in result["body"]


def test_serves_web_assets_and_blocks_traversal(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<h1>LexiGemma</h1>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("console.log('hello')")
    monkeypatch.setattr(web_handler, "WEB_ROOT", tmp_path)
    result = web_handler.handler(event("GET", "/"), None)
    assert result["statusCode"] == 200 and b"LexiGemma" in base64.b64decode(result["body"])
    result = web_handler.handler(event("GET", "/assets/app.js"), None)
    assert result["statusCode"] == 200 and "javascript" in result["headers"]["content-type"]
    assert web_handler.handler(event("GET", "/assets/../index.html"), None)["statusCode"] == 200
    assert web_handler.handler(event("GET", "/assets/../../secret"), None)["statusCode"] == 404
