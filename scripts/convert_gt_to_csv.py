#!/usr/bin/env python3
"""Convert MOT gt.txt files to CSV format with class remapping.

This script transforms ground truth annotations from MOT format to CSV format,
remapping class IDs:
- Class 1 and 2 -> Class 0 (pedestrians/obstacles)
- All other classes -> Class 2 (others)
"""

import argparse
from pathlib import Path

import pandas as pd


def remap_class_id(original_class: int) -> int:
    """Remap class IDs according to the specified mapping.

    Args:
        original_class: Original class ID from gt.txt

    Returns:
        Remapped class ID (1 for pedestrians, 2 for others)
    """
    if original_class in [1, 2]:
        return 0
    else:
        return 1


def convert_gt_to_csv(
    gt_path: Path,
    output_path: Path | None = None,
    remap_classes: bool = True
) -> None:
    """Convert gt.txt to CSV format with optional class remapping.

    Args:
        gt_path: Path to input gt.txt file
        output_path: Path to output CSV file (default: same dir as input, named gt.csv)
        remap_classes: Whether to remap class IDs (default: True)
    """
    # MOT format: frame_id, track_id, x, y, width, height, confidence, class_id, visibility
    column_names = [
        "frame_id",
        "track_id",
        "x",
        "y",
        "width",
        "height",
        "confidence",
        "class_id",
        "visibility"
    ]

    # Read gt.txt (comma-separated, no header)
    df = pd.read_csv(gt_path, header=None, names=column_names)

    # Remap class IDs if requested
    if remap_classes:
        df["class_id"] = df["class_id"].apply(remap_class_id)

    # Determine output path
    if output_path is None:
        output_path = gt_path.parent / "gt.csv"

    # Write to CSV with header
    df.to_csv(output_path, index=False)

    print(f"Converted {gt_path} -> {output_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Class distribution:")
    print(df["class_id"].value_counts().sort_index())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MOT gt.txt files to CSV with class remapping"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to gt.txt file or directory containing gt.txt files"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: gt.csv in same directory)"
    )
    parser.add_argument(
        "--no-remap",
        action="store_true",
        help="Disable class ID remapping"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively process all gt.txt files in directory"
    )

    args = parser.parse_args()

    input_path = args.input

    # Handle directory input
    if input_path.is_dir():
        if args.recursive:
            gt_files = list(input_path.rglob("gt.txt"))
        else:
            gt_files = list(input_path.glob("gt.txt"))

        if not gt_files:
            print(f"No gt.txt files found in {input_path}")
            return

        print(f"Found {len(gt_files)} gt.txt file(s)")
        for gt_file in gt_files:
            convert_gt_to_csv(
                gt_file,
                output_path=None,  # Auto-generate output path
                remap_classes=not args.no_remap
            )
            print()
    else:
        # Single file
        convert_gt_to_csv(
            input_path,
            output_path=args.output,
            remap_classes=not args.no_remap
        )


if __name__ == "__main__":
    main()
