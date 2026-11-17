"""Oracle test input: turn the ground-truth boxes of one video into a det.txt.

If the pipeline is right, the tracker fed with these perfect boxes must score almost 100%.
Example (from the repo folder):
    python -m scripts.make_oracle MOT17-09-FRCNN
Then:
    python -m scripts.run_tracker MOT17-09-FRCNN --data outputs/oracle_input --out outputs/oracle
    python -m scripts.evaluate oracle MOT17-09-FRCNN
"""
import argparse
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("--data", default="data/MOT17/train")
    parser.add_argument("--out", default="outputs/oracle_input")
    args = parser.parse_args()

    src = Path(args.data) / args.sequence
    dst = Path(args.out) / args.sequence
    (dst / "det").mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "seqinfo.ini", dst / "seqinfo.ini")

    gt = np.loadtxt(src / "gt" / "gt.txt", delimiter=",", ndmin=2)
    # gt row: frame, id, left, top, width, height, flag, class, visibility
    # keep only boxes that count in the evaluation: flag 1 and class 1 (pedestrian)
    keep = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    with open(dst / "det" / "det.txt", "w") as f:
        for frame, _, left, top, width, height, *_ in keep:
            # the true id is thrown away (-1): the tracker must find the IDs itself
            f.write(f"{int(frame)},-1,{left:.2f},{top:.2f},{width:.2f},{height:.2f},1\n")
    print(f"{args.sequence}: {len(keep)} ground-truth boxes -> {dst / 'det' / 'det.txt'}")


if __name__ == "__main__":
    main()
