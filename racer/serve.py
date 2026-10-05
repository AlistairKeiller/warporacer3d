"""HTTP front end for the viewer; the simulation loop stays in Mojo.

GET / serves viewer.html, /info the scene description, /map the raw track array.
POST /step hands its body to Mojo through `requests` and answers with whatever
Mojo puts on `replies` (a binary state snapshot). The server is single-threaded,
so steps are handled one at a time in request order.
"""

import http.server
import json
import queue
import threading


class Bridge:
    def __init__(self, port, page, info, track):
        page, info = page.encode(), json.dumps(info).encode()
        self.requests = queue.Queue()
        self.replies = queue.Queue()
        bridge = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, body, kind):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/":
                    self.send(page, "text/html; charset=utf-8")
                elif self.path == "/info":
                    self.send(info, "application/json")
                elif self.path == "/map":
                    self.send(track, "application/octet-stream")
                else:
                    self.send_error(404)

            def do_POST(self):
                bridge.requests.put(self.rfile.read(int(self.headers["Content-Length"])).decode())
                self.send(bridge.replies.get(), "application/octet-stream")

        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
