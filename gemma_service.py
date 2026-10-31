"""Gemma via Google's Generative Language API; no AWS credentials required."""
import json
import logging
import os
import re
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vocab_prompt import SYSTEM_PROMPT_GENERATION, SYSTEM_PROMPT_EVALUATION, _parse_json, _validate_schema


logger = logging.getLogger(__name__)


def model_rank(name):
    size = re.search(r"(?:^|[-])(?:e)?(\d+(?:\.\d+)?)b(?:[-]|$)", name)
    return (float(size.group(1)) if size else float('inf'), 0 if name.endswith('-it') else 1, name)


def discover_model(key, excluded=None):
    """Select only a Gemma model advertised for generateContent by Google."""
    token = None
    available = []
    for _ in range(3):
        query = {"pageSize": 1000}
        if token:
            query["pageToken"] = token
        request = Request("https://generativelanguage.googleapis.com/v1beta/models?" + urlencode(query),
                          headers={"x-goog-api-key": key})
        with urlopen(request, timeout=15) as response:
            catalog = json.load(response)
        for model in catalog.get("models", []):
            name = model.get("name", "").removeprefix("models/")
            if (name.startswith("gemma-") and name != excluded
                    and re.fullmatch(r"[A-Za-z0-9._-]+", name)
                    and "generateContent" in model.get("supportedGenerationMethods", [])):
                available.append(name)
        token = catalog.get("nextPageToken")
        if not token:
            break
    # Prefer an advertised smaller instruction model to reduce latency.
    return min(available, key=model_rank) if available else None


def complete(key, model, body, deadline):
    """Retry transient provider errors, with at most three generation calls."""
    for attempt in range(3):
        remaining = deadline - time.monotonic()
        if remaining < 2:
            raise TimeoutError("Generation deadline exceeded")
        request = Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          data=json.dumps(body).encode(), method="POST",
                          headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urlopen(request, timeout=min(45, remaining)) as response:
                return model, json.load(response)
        except HTTPError as exc:
            # Log only provider codes; never credentials, request bodies, or arbitrary error text.
            provider_status = "UNKNOWN"
            try:
                status = json.loads(exc.read(8192)).get("error", {}).get("status", "")
                if re.fullmatch(r"[A-Z_]{1,60}", status):
                    provider_status = status
            except (ValueError, AttributeError, OSError):
                pass
            logger.warning("GEMMA_UPSTREAM model=%s attempt=%s http=%s provider_status=%s", model, attempt + 1, exc.code, provider_status)
            if attempt == 2 or exc.code not in (404, 500, 502, 503, 504):
                raise
            if exc.code == 404 or attempt == 1:
                try:
                    replacement = discover_model(key, excluded=model)
                except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                    replacement = None
                if replacement:
                    model = replacement
                elif exc.code == 404:
                    raise exc
            time.sleep(0.5 * (attempt + 1))


def generate(goal, sentence=None, target_words=None):
    """Use a backup credential only for service/connection/access failures."""
    keys = []
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_BACKUP"):
        value = os.getenv(name, "").strip()
        if value and value not in keys:
            keys.append(value)
    if not keys:
        return {"error": "The coach is not configured. Add GEMINI_API_KEY in Vercel settings."}
    total_deadline = time.monotonic() + 150
    for index, key in enumerate(keys):
        if time.monotonic() >= total_deadline - 5:
            break
        budget = 70 if len(keys) > 1 else 150
        result = _generate_with_key(key, goal, sentence, target_words,
                                    min(total_deadline, time.monotonic() + budget))
        retryable = result.pop("_can_failover", False)
        if not retryable or index == len(keys) - 1:
            return result
        logger.warning("GEMMA_KEY_FAILOVER from_slot=%s to_slot=%s", index + 1, index + 2)
    return {"error": "Google did not complete the request within the available time. Please retry."}


def _generate_with_key(key, goal, sentence, target_words, deadline):
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
            "generationConfig": {"temperature": 0.2 if evaluation else 0.4, "maxOutputTokens": 4096}}
    stage = "model discovery"
    try:
        if not model:
            model = discover_model(key)
        if not model:
            return {"error": "Google lists no Gemma model supporting text generation for this API key. Check Gemma availability in Google AI Studio; a different provider may be needed."}
        stage = "Gemma generation"
        model, data = complete(key, model, body, deadline)
        for repair_attempt in range(2):
            if data.get("promptFeedback", {}).get("blockReason"):
                return {"error": "Google blocked this prompt. Please try a different learning goal."}
            candidates = data.get("candidates") or []
            candidate = candidates[0] if candidates else {}
            if candidate.get("finishReason") in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT"):
                return {"error": "Google declined this response. Please rephrase your goal or sentence."}
            try:
                parts = candidate.get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
                result = _parse_json(text)
                _validate_schema(result, evaluation)
                return result
            except (ValueError, KeyError, IndexError, TypeError):
                if repair_attempt or deadline - time.monotonic() < 5:
                    reason = "cut off" if candidate.get("finishReason") == "MAX_TOKENS" else "not valid JSON in the required format"
                    return {"error": f"Gemma's answer was {reason}, even after recovery. Please retry with a shorter goal."}
                body["contents"][0]["parts"][0]["text"] += (
                    "\nIMPORTANT: Output ONLY the requested JSON object. Keep each field concise. "
                    "Include every required key. Do not explain your reasoning."
                )
                body["generationConfig"]["temperature"] = 0.1
                model, data = complete(key, model, body, deadline)
    except HTTPError as exc:
        if exc.code in (401, 403):
            return {"error": "Google rejected access to the model. Check the API key and model access in Google AI Studio.", "_can_failover": True}
        if exc.code == 429:
            return {"error": "The coach is receiving too many requests. Please try again shortly."}
        if exc.code == 400:
            return {"error": "Google rejected the model request (HTTP 400). Check that the configured model supports text generation with the Gemini API."}
        if exc.code == 404:
            return {"error": "Google could not find the configured model (HTTP 404). Check GEMMA_MODEL_ID and model availability for your API key."}
        if 500 <= exc.code < 600:
            return {"error": f"Google's model service is unavailable (HTTP {exc.code}) after automatic recovery attempts. Your saved progress is safe; please try again later.", "_can_failover": True}
        return {"error": f"Google's model request failed (HTTP {exc.code}). Check model and API settings."}
    except TimeoutError:
        return {"error": f"Google timed out during {stage}. Please retry or select a smaller advertised Gemma model.", "_can_failover": True}
    except (URLError, OSError):
        return {"error": f"The Google connection failed during {stage}. Check Vercel runtime logs for network restrictions.", "_can_failover": True}
    except (ValueError, KeyError, IndexError, TypeError):
        return {"error": "Google returned an unreadable API response. Please retry."}
