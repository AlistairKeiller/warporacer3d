"""Browser viewer and keyboard controls for the native racer."""

import argparse
import gzip
import json
import shutil
import subprocess
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from prepare import compile_track
from track import Track

ROOT = Path(__file__).resolve().parent


def build(cpu_only=False):
    compiler = shutil.which("mojo") or str(ROOT / ".venv/bin/mojo")
    binary = ROOT / "build" / ("racer-cpu" if cpu_only else "racer")
    sources = [ROOT / "main.mojo", *ROOT.glob("racer/*.mojo")]
    if not binary.exists() or any(
        p.stat().st_mtime > binary.stat().st_mtime for p in sources
    ):
        binary.parent.mkdir(exist_ok=True)
        command = [compiler, "build", str(ROOT / "main.mojo"), "-o", str(binary)]
        if cpu_only:
            command += ["-D", "CPU_ONLY=true"]
        subprocess.run(command, cwd=ROOT, check=True)
    return binary


def scene(track, data, cars, policy):
    # Colour is per triangle; static geometry uploads once in the browser.
    vertices, colours = [], []
    for part in track.parts:
        triangles = part.vertices[part.triangles]
        normal = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        light = 0.65 + 0.35 * abs(normal @ np.array([0.3, -0.4, 0.86]))
        colour = np.array([0.65, 0.25, 0.18] if part.obstacle else [0.29, 0.36, 0.39])
        vertices.append(triangles.reshape(-1, 3))
        colours.append(np.repeat(light[:, None] * colour, 3, axis=0))
    nx, ny = map(int, data[2:4])
    fields = data[16 : 16 + 4 * nx * ny].reshape(4, ny, nx)
    lo, hi = track.bounds
    return {
        "name": track.name,
        "cars": cars,
        "policy": policy,
        "vertices": np.concatenate(vertices).ravel().tolist(),
        "colours": np.concatenate(colours).ravel().tolist(),
        "route": track.route.points.ravel().tolist(),
        "closed": track.route.closed,
        "centre": ((lo + hi) / 2).tolist(),
        "extent": float(max(hi[:2] - lo[:2])),
        "grid": {
            "nx": nx,
            "ny": ny,
            "x": float(data[6]),
            "y": float(data[7]),
            "cell": float(data[8]),
            "height": fields[1].ravel().tolist(),
            "sx": fields[2].ravel().tolist(),
            "sy": fields[3].ravel().tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path, nargs="?", default=Path("ramp"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--cars", type=int, default=1)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cars <= 64:
        parser.error("--cars must be 1..64")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be 1..65535")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error("checkpoint does not exist")
    track = Track.load(args.map)
    data = compile_track(track)
    binary = build(args.device == "cpu")
    compiled = ROOT / "build" / "viewer.wrmap"
    data.tofile(compiled)
    process = subprocess.Popen(
        [
            str(binary),
            "serve",
            str(compiled),
            args.device,
            str(args.cars),
            "1",
            str(args.checkpoint.resolve()) if args.checkpoint else "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Mojo runtime exited during startup ({process.poll()})")
        if line.startswith("{"):
            runtime = json.loads(line)
            if runtime.pop("ready", False):
                break
        print(line.rstrip())
    payload = gzip.compress(
        json.dumps(
            scene(track, data, args.cars, bool(args.checkpoint)) | runtime,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, data, content_type, status=200, compressed=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.reply(
                    (ROOT / "viewer.html").read_bytes(), "text/html; charset=utf-8"
                )
            elif self.path == "/scene":
                self.reply(payload, "application/json", compressed=True)
            else:
                self.reply(b"Not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/step":
                self.reply(b"Not found", "text/plain", 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("invalid request size")
                command = json.loads(self.rfile.read(length))
                if not isinstance(command, dict):
                    raise TypeError("controls must be a JSON object")
                steering = max(-100, min(100, int(command.get("steering", 0))))
                throttle = max(-100, min(100, int(command.get("throttle", 0))))
                policy = int(
                    bool(command.get("policy", False)) and bool(args.checkpoint)
                )
                reset = int(bool(command.get("reset", False)))
                process.stdin.write(f"{steering} {throttle} {reset} {policy}\n")
                process.stdin.flush()
                result = process.stdout.readline()
                if not result:
                    raise RuntimeError("Mojo runtime stopped")
                self.reply(result.encode(), "application/json")
            except (
                ValueError,
                TypeError,
                OverflowError,
                RuntimeError,
                BrokenPipeError,
            ) as error:
                self.reply(
                    json.dumps({"error": str(error)}).encode(), "application/json", 400
                )

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"{url} — Ctrl+C to stop")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        process.terminate()
        process.wait(timeout=5)


if __name__ == "__main__":
    main()
