"""Creates a test image that can be sent to the /predict endpoint.

    python scripts/make_test_image.py --from-dataset
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import CLASSES, IMAGE_MODE, IMAGE_SIZE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-dataset",
        action="store_true",
        help="Use a real image from the validation set instead of a random one.",
    )
    parser.add_argument("--index", type=int, default=0, help="Which image to use.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="test_image.png")
    args = parser.parse_args()

    if args.from_dataset:
        # Uses the copy already in --data-dir, so nothing is downloaded again.
        dataset = datasets.FashionMNIST(root=args.data_dir, train=False, download=True)
        image, label = dataset[args.index]
        image.save(args.out)
        print(f"Saved {args.out}. The actual class is: {CLASSES[label]}")
        return

    image = Image.new(IMAGE_MODE, (IMAGE_SIZE, IMAGE_SIZE))
    image.putdata([random.randint(0, 255) for _ in range(IMAGE_SIZE**2)])
    image.save(args.out)
    print(f"Saved {args.out}. This is a random pattern, not a real image.")


if __name__ == "__main__":
    main()
