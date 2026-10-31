from http.server import BaseHTTPRequestHandler
import json
import logging

from gemma_service import generate


class CoachHandler(BaseHTTPRequestHandler):
    evaluation = False

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                return self.respond(400, {"error": "Please provide a valid request."})
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            goal = data.get("user_goal")
            if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
                return self.respond(400, {"error": "Please provide a valid goal."})
            if self.evaluation:
                sentence, words = data.get("user_input"), data.get("target_words")
                if not isinstance(sentence, str) or not sentence.strip() or len(sentence) > 8000:
                    return self.respond(400, {"error": "Please provide a valid sentence."})
                if not isinstance(words, list) or not 1 <= len(words) <= 10 or any(not isinstance(w, str) or not w.strip() or len(w) > 100 for w in words):
                    return self.respond(400, {"error": "Please provide valid target words."})
                result = generate(goal.strip(), sentence.strip(), words)
            else:
                result = generate(goal.strip())
            if "error" in result:
                logging.getLogger(__name__).warning("GEMMA_REQUEST_FAILED route=%s reason=%s", "evaluate" if self.evaluation else "generate", result["error"])
            return self.respond(502 if "error" in result else 200, result)
        except (ValueError, TypeError, UnicodeError):
            return self.respond(400, {"error": "Please provide valid JSON."})
        except Exception as exc:
            logging.getLogger(__name__).error("GEMMA_HANDLER_FAILURE type=%s", type(exc).__name__)
            return self.respond(500, {"error": "The coach is temporarily unavailable."})

    def respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
