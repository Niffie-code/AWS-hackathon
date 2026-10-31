"""Adapt Vercel Python HTTP requests to the shared API Gateway handler."""
from http.server import BaseHTTPRequestHandler

from web_handler import handler as app_handler


class ApiHandler(BaseHTTPRequestHandler):
    route = ""

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 65536:
                self.send_error(413)
                return
            body = self.rfile.read(length).decode("utf-8")
        except (ValueError, UnicodeError):
            self.send_error(400)
            return
        result = app_handler({"requestContext": {"http": {"method": "POST"}},
                              "rawPath": self.route, "body": body}, None)
        self.send_response(result["statusCode"])
        for key, value in result["headers"].items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(result["body"].encode("utf-8"))
