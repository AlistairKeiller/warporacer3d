"""HTTP front end for the viewer; the simulation loop stays in Mojo.

GET / serves viewer.html, /info the scene description, /map the raw .wrmap.
POST /step hands its body to Mojo through a queue and answers with whatever
Mojo replies (a binary state snapshot).
"""
import http.server
import queue
import threading


class Bridge:
    def __init__(self, port, page, info, map_bytes):
        page, info = page.encode(), info.encode()
        self.requests = queue.Queue()
        self.replies = queue.Queue()
        self.lock = threading.Lock()
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
                    self.send(map_bytes, "application/octet-stream")
                else:
                    self.send_error(404)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                with bridge.lock:  # one step at a time, in request order
                    bridge.requests.put(body.decode())
                    self.send(bridge.replies.get(), "application/octet-stream")

        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def wait(self):
        return self.requests.get()

    def reply(self, data):
        self.replies.put(data)
