"""Gemma via Google's Generative Language API; no AWS credentials required."""
import json
import os
import re
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vocab_prompt import SYSTEM_PROMPT_GENERATION, SYSTEM_PROMPT_EVALUATION, _parse_json, _validate_schema


def discover_model(key, excluded=None):
    """Select only a Gemma model advertised for generateContent by Google."""
    token = None
    for _ in range(3):
        query = {"pageSize": 1000}
        if token:
            query["pageToken"] = token
        request = Request("https://generativelanguage.googleapis.com/v1beta/models?" + urlencode(query),
                          headers={"x-goog-api-key": key})
        with urlopen(request, timeout=5) as response:
            catalog = json.load(response)
        for model in catalog.get("models", []):
            name = model.get("name", "").removeprefix("models/")
            if (name.startswith("gemma-") and name != excluded
                    and re.fullmatch(r"[A-Za-z0-9._-]+", name)
                    and "generateContent" in model.get("supportedGenerationMethods", [])):
                return name
        token = catalog.get("nextPageToken")
        if not token:
            break
    return None


def generate(goal, sentence=None, target_words=None):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return {"error": "The coach is not configured. Add GEMINI_API_KEY in Vercel settings."}
    model = os.getenv("GEMMA_MODEL_ID", "").strip().removeprefix("models/")
    if model and not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        return {"error": "The configured model ID is invalid."}
    evaluation = bool(sentence)
    content = {"user_goal": goal}
    if evaluation:
        content.update(user_input=sentence, target_words=target_words)
    else:
        content["task"] = "Generate 3 new words tailored to my goal."
    prompt = SYSTEM_PROMPT_EVALUATION if evaluation else SYSTEM_PROMPT_GENERATION
    body = {"contents": [{"role": "user", "parts": [{"text": prompt + "\n\n" + json.dumps(content, ensure_ascii=False)}]}],
            "generationConfig": {"temperature": 0.2 if evaluation else 0.4, "maxOutputTokens": 2048}}
    try:
        if not model:
            model = discover_model(key)
        if not model:
            return {"error": "Google lists no Gemma model supporting text generation for this API key. Check Gemma availability in Google AI Studio; a different provider may be needed."}
        for attempt in range(2):
            request = Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              data=json.dumps(body).encode(), method="POST",
                              headers={"Content-Type": "application/json", "x-goog-api-key": key})
            try:
                with urlopen(request, timeout=20) as response:
                    data = json.load(response)
                break
            except HTTPError as exc:
                if exc.code != 404 or attempt:
                    raise
                replacement = discover_model(key, excluded=model)
                if not replacement:
                    return {"error": "The selected model is unavailable, and Google lists no alternative Gemma model for this API key. Check Gemma availability in Google AI Studio."}
                model = replacement
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
        result = _parse_json(text)
        _validate_schema(result, evaluation)
        return result
    except HTTPError as exc:
        if exc.code in (401, 403):
            return {"error": "Google rejected access to the model. Check the API key and model access in Google AI Studio."}
        if exc.code == 429:
            return {"error": "The coach is receiving too many requests. Please try again shortly."}
        if exc.code == 400:
            return {"error": "Google rejected the model request (HTTP 400). Check that the configured model supports text generation with the Gemini API."}
        if exc.code == 404:
            return {"error": "Google could not find the configured model (HTTP 404). Check GEMMA_MODEL_ID and model availability for your API key."}
        if 500 <= exc.code < 600:
            return {"error": f"Google's model service returned HTTP {exc.code}. Please try again later."}
        return {"error": f"Google's model request failed (HTTP {exc.code}). Check model and API settings."}
    except (URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError):
        return {"error": "The coach could not produce a valid response. Please try again."}
