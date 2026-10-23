"""Small MCP HTTP client. Credentials are read only from the process environment."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

class Client:
    def __init__(self):
        self.url = os.environ["MCP_URL"]
        self.token = os.environ["MCP_TOKEN"]
        self.session = None
        self.version = "2024-11-05"
        self.counter = 0

    def request(self, method, params=None, notification=False):
        self.counter += 1
        body = {"jsonrpc": "2.0", "method": method}
        if not notification:
            body["id"] = self.counter
        if params is not None:
            body["params"] = params
        headers = {"Authorization": "Bearer " + self.token,
                   "Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if method != "initialize":
            headers["MCP-Protocol-Version"] = self.version
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.headers.get("Mcp-Session-Id"):
                self.session = response.headers["Mcp-Session-Id"]
            if notification:
                return None
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                parts = []
                for line in response:
                    line = line.decode().rstrip("\r\n")
                    if line.startswith("data:"):
                        parts.append(line[5:].lstrip())
                    elif not line and parts:
                        data = json.loads("\n".join(parts))
                        parts = []
                        if data.get("id") == body["id"]:
                            break
                else:
                    raise RuntimeError("MCP stream ended before response")
            else:
                data = json.load(response)
        if "error" in data:
            raise RuntimeError("MCP JSON-RPC error code " + str(data["error"].get("code")))
        return data["result"]

    def initialize(self):
        result = self.request("initialize", {"protocolVersion": self.version,
            "capabilities": {}, "clientInfo": {"name": "codex-patientphex", "version": "1.0"}})
        self.version = result["protocolVersion"]
        self.request("notifications/initialized", notification=True)
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["initialize", "tools/list", "tools/call"])
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()
    try:
        client = Client()
        initialized = client.initialize()
        result = initialized if args.method == "initialize" else client.request(args.method, json.loads(args.params))
        output = json.dumps(result, ensure_ascii=False, indent=2)
        for name in ("MCP_TOKEN", "MCP_URL"):
            value = os.environ.get(name)
            if value:
                output = output.replace(value, "[REDACTED]")
        print(output)
        if isinstance(result, dict) and result.get("isError"):
            sys.exit(1)
    except urllib.error.HTTPError as exc:
        print("MCP HTTP error: " + str(exc.code), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print("MCP request failed: " + type(exc).__name__, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
