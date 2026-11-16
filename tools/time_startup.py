"""Measure a fresh local process to HTTP readiness, keeping the worker available for warm requests."""

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def available(port):
    with socket.socket() as server:
        try:
            server.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--studio",
        action="store_true",
        help="Start the private studio launcher and additionally time its web app.",
    )
    parser.add_argument("--studio-port", type=int, default=8000)
    args = parser.parse_args()
    routes = [("worker_ready_s", args.port, "/health")]
    if args.studio:
        if not Path("run_local.py").is_file():
            parser.error("Use --studio from the full studio repository.")
        command = [
            sys.executable,
            "run_local.py",
            "--model",
            "klein",
            "--port",
            str(args.studio_port),
            "--model-port",
            str(args.port),
        ]
        routes.append(("studio_ready_s", args.studio_port, "/api/health"))
    else:
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "gpu_service.klein_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
    if not all(available(port) for _, port, _ in routes):
        parser.error("The requested port is already in use.")
    if args.output.exists():
        parser.error("Choose a fresh output file.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "definition": "Process launch to HTTP readiness. Downloaded checkpoints remain on disk.",
        "utc": datetime.now(UTC).isoformat(),
    }
    start = time.perf_counter()
    worker = subprocess.Popen(command)
    try:
        for name, port, path in routes:
            while time.perf_counter() - start < 240:
                if worker.poll() is not None:
                    raise RuntimeError("Worker stopped before readiness.")
                try:
                    with urlopen(
                        f"http://127.0.0.1:{port}{path}", timeout=1
                    ) as response:
                        body = json.load(response)
                    if body.get("ready") or body.get("status") == "ok":
                        record[name] = round(time.perf_counter() - start, 3)
                        if body.get("instance_id"):
                            record["instance_id"] = body["instance_id"]
                        break
                except (OSError, URLError, ValueError):
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("Worker did not become ready.")
        with args.output.open("x") as output:
            output.write(json.dumps(record, indent=2) + "\n")
        print(record, flush=True)
        worker.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if worker.poll() is None:
            if args.studio:
                import signal

                worker.send_signal(signal.SIGINT)
            else:
                worker.terminate()
            worker.wait(timeout=30)


if __name__ == "__main__":
    main()
