"""Gemma via Google's Generative Language API; no AWS credentials required."""
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vocab_prompt import SYSTEM_PROMPT_GENERATION, SYSTEM_PROMPT_EVALUATION, _parse_json, _validate_schema


def generate(goal, sentence=None, target_words=None):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return {"error": "The coach is not configured. Add GEMINI_API_KEY in Vercel settings."}
    model = os.getenv("GEMMA_MODEL_ID", "gemma-3-27b-it")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
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
    request = Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                      data=json.dumps(body).encode(), method="POST",
                      headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urlopen(request, timeout=25) as response:
            data = json.load(response)
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
