"""Read-only Google model discovery check. Never exposes the API key."""
from http.server import BaseHTTPRequestHandler
import json
import os
from urllib.error import HTTPError

from gemma_service import discover_model


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        key = os.getenv('GEMINI_API_KEY')
        status = 200
        try:
            if not key:
                result = {'ok': False, 'stage': 'configuration', 'error': 'GEMINI_API_KEY is missing'}
            else:
                model = discover_model(key)
                result = {'ok': bool(model), 'stage': 'model discovery', 'model': model,
                          'note': 'This checks model discovery only, not generation.'}
        except HTTPError as exc:
            status = 502
            result = {'ok': False, 'stage': 'model discovery', 'google_status': exc.code}
        except Exception as exc:
            status = 502
            result = {'ok': False, 'stage': 'model discovery', 'error_type': type(exc).__name__}
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
