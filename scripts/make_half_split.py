"""Half split: cut each MOT17 training video into a first half and a second half.

We tune settings on the first half ("train") and report numbers on the second half ("val"),
so the reported numbers come from frames we never tuned on.
Each half becomes its own small video, with frames renumbered from 1:
    outputs/half/train/<sequence>/  and  outputs/half/val/<sequence>/
each with det/det.txt, gt/gt.txt and seqinfo.ini, so run_tracker.py and evaluate.py work on it unchanged.

Example (from the repo folder):
    python -m scripts.make_half_split
"""
import argparse
import configparser
from pathlib import Path

import numpy as np

FRCNN = [f"MOT17-{n:02d}-FRCNN" for n in (2, 4, 5, 9, 10, 11, 13)]


def write_rows(path, rows, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt=fmt, delimiter=",")


def split_sequence(src, out, name):
    info = configparser.ConfigParser()
    info.read(src / "seqinfo.ini")
    n_frames = int(info["Sequence"]["seqLength"])
    half = n_frames // 2                      # first half: frames 1..half, second half: half+1..n_frames

    det = np.loadtxt(src / "det" / "det.txt", delimiter=",", ndmin=2)
    gt = np.loadtxt(src / "gt" / "gt.txt", delimiter=",", ndmin=2)

    for part, first, last in (("train", 1, half), ("val", half + 1, n_frames)):
        dst = out / part / name
        shift = first - 1                     # renumber so that every half starts at frame 1
        for rows, sub, fmt in ((det, "det/det.txt", ["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%.4f"]),
                               (gt, "gt/gt.txt", ["%d"] * 2 + ["%.2f"] * 4 + ["%d", "%d", "%.4f"])):
            keep = rows[(rows[:, 0] >= first) & (rows[:, 0] <= last)].copy()
            keep[:, 0] -= shift
            write_rows(dst / sub, keep[:, :len(fmt)], fmt)
        part_info = configparser.ConfigParser()
        part_info.optionxform = str           # keep "seqLength" with its capital L
        part_info.read(src / "seqinfo.ini")
        part_info["Sequence"]["seqLength"] = str(last - first + 1)
        with open(dst / "seqinfo.ini", "w") as f:
            part_info.write(f)
    return half, n_frames - half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", nargs="*", default=FRCNN, help="default: the 7 FRCNN training videos")
    parser.add_argument("--data", default="data/MOT17/train")
    parser.add_argument("--out", default="outputs/half")
    args = parser.parse_args()

    for name in args.sequences:
        n_train, n_val = split_sequence(Path(args.data) / name, Path(args.out), name)
        print(f"{name}: first half {n_train} frames, second half {n_val} frames")


if __name__ == "__main__":
    main()
