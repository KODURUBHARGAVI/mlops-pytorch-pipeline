"""Exercise the running inference API: /health, /metadata and /predict.
    python scripts/smoke_test.py
    python scripts/smoke_test.py --base-url http://localhost:8000   # if the port was changed
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def get_json(url: str) -> tuple[int, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body.decode(errors="replace")[:200]


def post_image(url: str, image_path: Path) -> tuple[int, object]:
    """POST one file as multipart/form-data under the field name `image`."""
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            image_path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def show(label: str, status: int, payload: object) -> None:
    print(f"\n=== {label}  [HTTP {status}] ===")
    print(json.dumps(payload, indent=2) if isinstance(payload, (dict, list)) else payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--image", default="test_image.png")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    try:
        health_status, health = get_json(f"{base}/health")
    except (urllib.error.URLError, TimeoutError) as exc:
        sys.exit(f"Could not reach {base} - is serve.py running there? ({exc})")

    show("GET /health", health_status, health)
    if isinstance(health, dict) and "scheduler" in health:
        sys.exit(
            f"\n{base} is answering, but that is not this project's API "
            "(it looks like Airflow). Start serve.py on a free port and pass "
            "--base-url to point at it."
        )
    if health_status != 200:
        sys.exit("\n/health is not 200 - the checkpoint has not loaded. Train a model first.")

    show("GET /metadata", *get_json(f"{base}/metadata"))

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"\n{image_path} not found - run: python scripts/make_test_image.py --from-dataset")
    show(f"POST /predict  ({image_path.name})", *post_image(f"{base}/predict", image_path))
    print()


if __name__ == "__main__":
    main()
