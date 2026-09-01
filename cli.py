import argparse
import math

import cv2

from benchmark import benchmark_until, edge_for_elapsed, estimate_elapsed
from morph import dual_resize, morph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("-o", "--output", default="reconstructed.jpg")

    size = parser.add_mutually_exclusive_group()
    size.add_argument("-d", "--downsample", type=float, default=1)
    size.add_argument("-e", "--edge-length", type=int)
    size.add_argument("-t", "--target-runtime", type=float)

    parser.add_argument("--estimate-runtime", action="store_true")
    parser.add_argument("--benchmark-seconds", type=float, default=10)
    args = parser.parse_args()

    source = cv2.imread(args.source)
    target = cv2.imread(args.target)
    if source is None or target is None:
        parser.error("could not read source or target image")

    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    if args.target_runtime is not None:
        results = benchmark_until(
            source,
            target,
            total_budget=args.benchmark_seconds,
            single_budget=args.benchmark_seconds,
        )
        edge_length = edge_for_elapsed(results, args.target_runtime)
        estimate = estimate_elapsed(results, edge_length)
        print(f"Selected edge length: {edge_length}px ({estimate:.2f}s estimated)")
        source, target = dual_resize(source, target, edge_length=edge_length)
    elif args.edge_length is not None:
        source, target = dual_resize(source, target, edge_length=args.edge_length)
    else:
        source, target = dual_resize(source, target, args.downsample)

    if args.estimate_runtime and args.target_runtime is None:
        height, width = source.shape[:2]
        aspect_ratio = max(height / width, width / height)
        edge_length = max(height, width)
        results = benchmark_until(
            source,
            target,
            n_start=math.ceil(aspect_ratio),
            total_budget=args.benchmark_seconds,
            single_budget=args.benchmark_seconds,
            max_edge=edge_length,
        )
        estimate = estimate_elapsed(results, edge_length)
        print(f"Estimated morph time: {estimate:.2f}s")

    result = morph(source, target)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    cv2.imwrite(args.output, result)


if __name__ == "__main__":
    main()
