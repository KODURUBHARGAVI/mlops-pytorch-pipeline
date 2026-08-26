"""Fetch a dataset once, quickly, with progress and resume support.
python scripts/fetch_data.py --dataset fashionmnist
python scripts/fetch_data.py --dataset cifar10
python scripts/fetch_data.py --dataset cifar10 --url https://your/mirror.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Mirrors are tried in order. The first is a public mirror of the identical
# archive; the last is torchvision's own (slow) origin as a fallback.
CIFAR10_URLS = [
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]
CIFAR10_FILENAME = "cifar-10-python.tar.gz"
CHUNK = 1 << 20  # 1 MiB


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity check, not a security control
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_with_resume(url: str, target: Path, timeout: int = 30) -> None:
    """Stream `url` into `target`, resuming from whatever is already there."""
    existing = target.stat().st_size if target.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "mlops-pipeline/1.0"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")

    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        resuming = response.status == 206
        if existing and not resuming:
            print("  server ignored the resume request; starting over")
            existing = 0

        remaining = int(response.headers.get("Content-Length", 0))
        total = existing + remaining
        if existing:
            print(f"  resuming at {human(existing)} of {human(total)}")

        mode = "ab" if resuming and existing else "wb"
        downloaded, started, last_report = existing, time.time(), 0.0

        with open(target, mode) as f:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_report >= 1.0 or downloaded == total:
                    last_report = now
                    speed = (downloaded - existing) / max(now - started, 1e-6)
                    pct = f"{downloaded / total * 100:5.1f}%" if total else "  ?  "
                    eta = (total - downloaded) / speed if speed > 0 and total else 0
                    sys.stdout.write(
                        f"\r  {pct}  {human(downloaded)}/{human(total)}  "
                        f"{human(speed)}/s  ETA {int(eta // 60)}m{int(eta % 60):02d}s   "
                    )
                    sys.stdout.flush()
    print()


def fetch_cifar10(data_dir: Path, urls: list[str]) -> None:
    from torchvision import datasets

    expected_md5 = datasets.CIFAR10.tgz_md5  # torchvision's own checksum
    target = data_dir / CIFAR10_FILENAME
    data_dir.mkdir(parents=True, exist_ok=True)

    if target.exists() and md5_of(target) == expected_md5:
        print(f"Archive already present and valid: {target}")
    else:
        for url in urls:
            print(f"Downloading CIFAR-10 from {url}")
            try:
                download_with_resume(url, target)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"  failed: {exc}")
                continue

            actual = md5_of(target)
            if actual == expected_md5:
                print("  checksum OK")
                break
            print(f"  checksum mismatch (got {actual}, want {expected_md5}); discarding")
            os.replace(target, target.with_suffix(".bad"))
        else:
            raise SystemExit("Every mirror failed. Pass a working one with --url.")

    print("Extracting and verifying with torchvision ...")
    datasets.CIFAR10(root=str(data_dir), train=True, download=True)
    datasets.CIFAR10(root=str(data_dir), train=False, download=True)
    print(f"CIFAR-10 ready in {data_dir}")


def fetch_fashion_mnist(data_dir: Path) -> None:
    from torchvision import datasets

    # ~30 MB from a fast S3 mirror - no hand-rolled downloader needed.
    print("Downloading Fashion-MNIST (~30 MB) ...")
    datasets.FashionMNIST(root=str(data_dir), train=True, download=True)
    datasets.FashionMNIST(root=str(data_dir), train=False, download=True)
    print(f"Fashion-MNIST ready in {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10", "fashionmnist"], default="cifar10")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--url", help="Override the mirror for cifar10.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.dataset == "fashionmnist":
        fetch_fashion_mnist(data_dir)
    else:
        fetch_cifar10(data_dir, [args.url] if args.url else CIFAR10_URLS)


if __name__ == "__main__":
    main()
