"""Create a test image to POST at the /predict endpoint.
python scripts/make_test_image.py                      # random noise
python scripts/make_test_image.py --from-dataset       # a real sample
python scripts/make_test_image.py --from-dataset --dataset cifar10 --index 7
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import get_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="fashionmnist",
        help="Which dataset the model was trained on (default: fashionmnist).",
    )
    parser.add_argument(
        "--from-dataset",
        action="store_true",
        help="Use a real image from the test split instead of random noise.",
    )
    parser.add_argument("--index", type=int, default=0, help="Which test-set image to use.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="test_image.png")
    args = parser.parse_args()

    spec = get_spec(args.dataset)

    if args.from_dataset:
        # Reuses the already-downloaded copy in --data-dir; nothing is refetched.
        dataset = spec.torchvision_class(root=args.data_dir, train=False, download=True)
        image, label = dataset[args.index]
        image.save(args.out)
        print(f"Saved {args.out} from {spec.name}[{args.index}] - true class: {spec.classes[label]}")
        return

    image = Image.new(spec.image_mode, (spec.image_size, spec.image_size))
    if spec.in_channels == 1:
        pixels = [random.randint(0, 255) for _ in range(spec.image_size**2)]
    else:
        pixels = [
            tuple(random.randint(0, 255) for _ in range(3)) for _ in range(spec.image_size**2)
        ]
    image.putdata(pixels)
    image.save(args.out)
    print(f"Saved {args.out} - random noise, {spec.image_size}x{spec.image_size} {spec.image_mode}")


if __name__ == "__main__":
    main()
