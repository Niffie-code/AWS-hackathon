"""LexiGemma vocabulary coaching via Bedrock Converse.

Run this file directly for offline tests; neither boto3 nor AWS credentials
are needed for those tests. Confirm the configured model ID in Bedrock.
"""

import json
import os
import re
from functools import lru_cache

try:
    from botocore.exceptions import ClientError
except ImportError:
    # A compatible test exception when the optional AWS SDK is not installed.
    # A real Bedrock client always uses botocore's ClientError.
    class ClientError(Exception):
        def __init__(self, error_response, operation_name):
            self.response = error_response
            self.operation_name = operation_name
            error = error_response.get("Error", {})
            super().__init__(f"{error.get('Code', 'Unknown')}: {error.get('Message', '')}")


MODEL_ID = os.getenv("LEXIGEMMA_MODEL_ID", "google.gemma-4-31b")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

SYSTEM_PROMPT_GENERATION = """
# ROLE
You are Lexi, a witty, encouraging English vocabulary coach. You teach advanced words as precision tools, not flashcard entries.

# METHOD
1. Context over rote: convey each word's flavor and register (literary, academic, corporate, casual) and give a contrasting synonym to show exact boundaries.
2. Etymological hooks: anchor each word with a brief history, root, or visual metaphor.
3. Anti-robot: sound like an articulate coach, never a dictionary.

# RULES
- Return exactly 3 C1-C2 level words, tailored to the user's stated goal.
- Each "challenge" must be specific to the user's goal and require them to write one original sentence using the word.
- The "mini_story" must use all 3 words naturally, in 80-120 words.

# OUTPUT
Respond with ONE valid JSON object and nothing else: no markdown fences, no text before or after.
{
  "words": [
    {
      "word": "string",
      "pronunciation": "string, phonetic respelling",
      "definition": "string, one vivid sentence",
      "nuance": "string, connotation, register, common collocations",
      "spectrum_comparison": "string, contrast with a common synonym",
      "challenge": "string, a task requiring the user to write a sentence with the word"
    }
  ],
  "mini_story": "string"
}
"""

SYSTEM_PROMPT_EVALUATION = """
# ROLE
You are Lexi, an encouraging, witty English professor evaluating a learner's sentence.

# METHOD
- Check the sentence against the target words and the user's goal.
- Assess grammar, collocations and prepositions, connotation, and register.
- Praise what works first, then show how to make the sentence more precise.

# RULES
- Score 0-100: 90+ means precise and idiomatic; 70-89 means correct with minor tone or collocation slips; below 70 means misused word or significant errors.
- If a target word is missing or misused, say so plainly and kindly.

# OUTPUT
Respond with ONE valid JSON object and nothing else: no markdown fences, no text before or after.
{
  "score": <integer 0-100>,
  "feedback": "string, 3-5 sentences",
  "improved_sentence": "string, a stronger rewrite that keeps the user's intent",
  "words_used_correctly": ["string"]
}
"""


@lru_cache(maxsize=1)
def _get_client():
    """Create and retain the client only when inference is requested."""
    import boto3

    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def _parse_json(raw_text):
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, count=1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first, last = text.find("{"), text.rfind("}")
        if first >= 0 and last >= first:
            return json.loads(text[first:last + 1])
        raise


def _validate_schema(result, evaluation):
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    if evaluation:
        score = result.get("score")
        if type(score) is not int or not 0 <= score <= 100:
            raise ValueError("score must be an integer from 0 to 100")
        if not isinstance(result.get("feedback"), str):
            raise ValueError("feedback must be a string")
    else:
        words = result.get("words")
        if not isinstance(words, list) or len(words) != 3:
            raise ValueError("words must contain exactly 3 entries")
        keys = ("word", "pronunciation", "definition", "nuance", "spectrum_comparison", "challenge")
        for index, word in enumerate(words):
            if not isinstance(word, dict):
                raise ValueError(f"words[{index}] must be an object")
            for key in keys:
                if not isinstance(word.get(key), str):
                    raise ValueError(f"words[{index}].{key} must be a string")
        if not isinstance(result.get("mini_story"), str):
            raise ValueError("mini_story must be a string")


def generate_vocab_response(
    user_goal: str,
    user_input: str | None = None,
    target_words: list[str] | None = None,
) -> dict:
    """Generate three words or evaluate a sentence; return errors as dictionaries."""
    raw_text = None
    try:
        if not isinstance(user_goal, str):
            return {"error": "Invalid input: user_goal must be a string"}
        if user_input is not None and not isinstance(user_input, str):
            return {"error": "Invalid input: user_input must be a string or None"}
        if target_words is not None and (
            not isinstance(target_words, list)
            or any(not isinstance(word, str) for word in target_words)
        ):
            return {"error": "Invalid input: target_words must be a list of strings or None"}

        evaluation = bool(user_input and user_input.strip())
        system_prompt = SYSTEM_PROMPT_EVALUATION if evaluation else SYSTEM_PROMPT_GENERATION
        user_message = {"user_goal": user_goal}
        if evaluation:
            user_message.update(user_input=user_input, target_words=target_words or [])
        else:
            user_message["task"] = "Generate 3 new words tailored to my goal."
        user_text = json.dumps(user_message, ensure_ascii=False)
        request = {
            "modelId": MODEL_ID,
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
            "inferenceConfig": {"temperature": 0.2 if evaluation else 0.4, "maxTokens": 2048},
        }
        client = _get_client()
        try:
            response = client.converse(**request)
        except ClientError as exc:
            error = exc.response.get("Error", {})
            if error.get("Code") != "ValidationException" or "system" not in error.get("Message", "").lower():
                raise
            request.pop("system")
            request["messages"] = [{"role": "user", "content": [
                {"text": system_prompt + "\n\n" + user_text}
            ]}]
            response = client.converse(**request)

        content = response["output"]["message"]["content"]
        if not isinstance(content, list):
            raise TypeError("content must be a list")
        text_blocks = [block["text"] for block in content if "text" in block]
        if not text_blocks:
            raise KeyError("text")
        raw_text = "".join(text_blocks)
        result = _parse_json(raw_text)
        try:
            _validate_schema(result, evaluation)
        except ValueError as exc:
            return {"error": f"Schema validation failed: {exc}", "raw_text": raw_text}
        return result
    except ClientError as exc:
        result = {"error": f"Bedrock ClientError: {exc}"}
    except (KeyError, IndexError) as exc:
        result = {"error": f"Invalid Converse response shape: {exc}"}
    except json.JSONDecodeError as exc:
        result = {"error": f"JSON parsing failed: {exc}"}
    except Exception as exc:
        result = {"error": f"Vocabulary response failed: {type(exc).__name__}: {exc}"}
    if raw_text is not None:
        result["raw_text"] = raw_text
    return result


if __name__ == "__main__":
    from unittest.mock import Mock, patch

    generation = {
        "words": [
            {
                "word": word, "pronunciation": pronunciation,
                "definition": definition,
                "nuance": "Formal workplace English; suitable for a project review.",
                "spectrum_comparison": comparison,
                "challenge": f"Write one original sentence using {word} to describe a decision at work.",
            }
            for word, pronunciation, definition, comparison in [
                ("perspicacious", "pur-spih-KAY-shus", "Quick to spot what others miss.", "Sharper in judgment than merely observant."),
                ("equivocal", "ih-KWIV-uh-kul", "Open to more than one interpretation.", "More ambiguous than simply uncertain."),
                ("intransigent", "in-TRAN-sih-junt", "Unwilling to compromise a position.", "More resistant to compromise than merely stubborn."),
            ]
        ],
        "mini_story": (
            "At Monday's review, our perspicacious analyst noticed that the launch forecast "
            "counted the same customers twice. The sales lead offered an equivocal answer "
            "when asked whether the targets were realistic, leaving everyone unsure what "
            "he actually believed. Our usually intransigent director surprised us by "
            "agreeing to a smaller pilot. Instead of defending their departments, the "
            "team spent the afternoon checking assumptions together. By Friday, we had "
            "a modest plan that everyone could explain clearly, and a much better reason "
            "to trust the numbers behind it."
        ),
    }
    evaluation = {
        "score": 94,
        "feedback": "You used perspicacious accurately. The workplace register fits your goal. Name the insight to make the praise more precise.",
        "improved_sentence": "Our perspicacious analyst spotted a hidden flaw in the forecast.",
        "words_used_correctly": ["perspicacious"],
    }

    def reply(text):
        return {"output": {"message": {"content": [{"text": text}]}}}

    with patch("__main__._get_client") as getter:
        client = Mock()
        getter.return_value = client

        client.converse.return_value = reply(json.dumps(generation))
        result = generate_vocab_response("Speak precisely at work")
        print("1. Generation:")
        print(json.dumps(result, indent=2))
        assert len(result["words"]) == 3 and result["mini_story"]

        client.converse.return_value = reply(json.dumps(evaluation))
        result = generate_vocab_response("Speak precisely at work", "Our analyst is perspicacious.", ["perspicacious"])
        print("2. Evaluation:")
        print(json.dumps(result, indent=2))
        assert type(result["score"]) is int
        request = client.converse.call_args.kwargs
        assert json.loads(request["messages"][0]["content"][0]["text"])["target_words"] == ["perspicacious"]

        client.converse.return_value = reply("```json\n" + json.dumps(generation) + "\n```")
        result = generate_vocab_response("Speak precisely at work", "   ")
        print("3. Fenced JSON:")
        print(json.dumps(result, indent=2))
        assert len(result["words"]) == 3

        client.converse.return_value = reply("Here are some wonderful words!")
        result = generate_vocab_response("Speak precisely at work")
        print("4. Malformed output:")
        print(json.dumps(result, indent=2))
        assert "error" in result and "raw_text" in result

        client.converse.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Mock access denied"}}, "Converse"
        )
        result = generate_vocab_response("Speak precisely at work")
        print("5. ClientError:")
        print(json.dumps(result, indent=2))
        assert "error" in result

    print("All mock tests passed.")
