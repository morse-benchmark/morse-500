"""Runs inside the network/filesystem namespace; never receives a real key."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    config = json.loads(Path("/config.json").read_text())
    port = config.get("relay_port", 8080)
    home = config.get("home", "/home/agent")
    relay_binary = config.get("relay_binary", "/usr/bin/socat")
    gateway_socket = config.get("gateway_socket", "/gateway/api.sock")
    relay = subprocess.Popen([
        relay_binary, f"TCP-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork",
        f"UNIX-CONNECT:{gateway_socket}",
    ], stdout=subprocess.DEVNULL, stderr=sys.stderr)
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.1):
                    break
            except OSError:
                if relay.poll() is not None:
                    raise RuntimeError("Local gateway relay failed")
                time.sleep(.02)
        else:
            raise RuntimeError("Local gateway relay did not become ready")
        os.environ["HOME"] = home
        for directory in config.get("directories", []):
            Path(directory).mkdir(parents=True, exist_ok=True)
        os.environ.update(config.get("environment", {}))
        result = subprocess.run(config["command"], input=config["input"], text=True)
        return result.returncode
    finally:
        relay.terminate()
        relay.wait()


if __name__ == "__main__":
    sys.exit(main())
