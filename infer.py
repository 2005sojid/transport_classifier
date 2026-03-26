"""
Production inference entry point.

Usage:
  python infer.py path/to/image.jpg
  python infer.py path/to/image.jpg --k 11 --verbose
"""
import argparse
import os
import sys
import time
from PIL import Image

import config
import classifier


def main():
    parser = argparse.ArgumentParser(description="Transport classifier inference")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--k", type=int, default=config.K_NEAREST)
    parser.add_argument("--verbose", action="store_true",
                        help="Print top neighbors and all class scores")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    img = Image.open(args.image).convert("RGB")
    clf = classifier.GalleryKNN()
    clf.predict(img, k=args.k)  # warm-up: loads model weights

    t0 = time.perf_counter()
    pred = clf.predict(img, k=args.k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Main result
    status = "LOW CONFIDENCE" if pred.low_confidence else "OK"
    print(f"\nResult: {pred.label}  ({pred.confidence:.1%})  [{status}]")
    print(f"Inference: {elapsed_ms:.0f} ms")

    if args.verbose:
        print("\nClass scores:")
        for cls, score in sorted(pred.scores.items(), key=lambda x: -x[1]):
            bar = "#" * int(score * 40)
            print(f"  {cls:<15} {score:.4f}  {bar}")

        print(f"\nTop-{len(pred.top_neighbors)} neighbors:")
        for n in pred.top_neighbors:
            print(f"  #{n['rank']}  {n['class']:<15}  sim={n['similarity']:.4f}  "
                  f"{os.path.basename(n['path'])}")


if __name__ == "__main__":
    main()
