import json
import os

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000/api")
TOKEN = os.getenv("API_TOKEN", "")


def stream_alerts():
    params = {}
    if TOKEN:
        params["token"] = TOKEN

    with requests.get(f"{API_BASE_URL}/stream/alerts", params=params, stream=True, timeout=3600) as response:
        response.raise_for_status()
        print("Connected to alert stream. Waiting for events...")

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("data: "):
                payload = raw_line.replace("data: ", "", 1)
                try:
                    parsed = json.loads(payload)
                    print(json.dumps(parsed, indent=2))
                except json.JSONDecodeError:
                    print(payload)


if __name__ == "__main__":
    stream_alerts()
