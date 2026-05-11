import argparse
import os
import sys
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Check missing image files referenced by CSV.")
    parser.add_argument("--csv", type=str, required=True, help="./datasets/research/dept8/qdou/data/RSNA-ICH/training.c")
    parser.add_argument("--root", type=str, required=True, help="./datasets/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train")
    parser.add_argument("--image-col", type=str, default="ImageID", help="CSV column name for image filenames")
    parser.add_argument("--save-missing", type=str, default="missing_files.csv", help="Output CSV path for missing files")
    parser.add_argument("--show", type=int, default=20, help="30")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)
    if not os.path.isdir(args.root):
        print(f"[ERROR] Root directory not found: {args.root}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if args.image_col not in df.columns:
        print(f"[ERROR] Column '{args.image_col}' not found in CSV. Available columns: {list(df.columns)}")
        sys.exit(1)

    names = df[args.image_col].astype(str).tolist()
    missing = []
    for name in names:
        path = os.path.join(args.root, name)
        if not os.path.exists(path):
            missing.append(name)

    total = len(names)
    miss_n = len(missing)
    miss_ratio = miss_n / total if total > 0 else 0.0

    print("=" * 60)
    print("Missing File Check")
    print("=" * 60)
    print(f"CSV: {args.csv}")
    print(f"Root: {args.root}")
    print(f"Image column: {args.image_col}")
    print(f"Total entries: {total}")
    print(f"Missing entries: {miss_n}")
    print(f"Missing ratio: {miss_ratio:.4%}")
    print("-" * 60)

    if miss_n > 0:
        show_n = min(args.show, miss_n)
        print(f"First {show_n} missing examples:")
        for x in missing[:show_n]:
            print(x)

        out_df = pd.DataFrame({args.image_col: missing})
        out_df.to_csv(args.save_missing, index=False)
        print("-" * 60)
        print(f"Saved missing list to: {args.save_missing}")
    else:
        print("No missing files found.")

    print("=" * 60)


if __name__ == "__main__":
    main()
