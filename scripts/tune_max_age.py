"""Try several max_age values on the FIRST halves (train) and print one line per value.

We choose the value here, then report it on the second halves (val) only once.
Rule, fixed before looking: the highest IDF1 wins, but a larger max_age must beat the best so far
by more than 0.1 IDF1 (otherwise we keep the smaller, simpler value).
Example (from the repo folder):
    python -m scripts.tune_max_age --stage 1
"""
import argparse
import subprocess
import sys
from pathlib import Path

SEQUENCES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
             "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]


def read_summary(path):
    """pedestrian_summary.txt = a header line + a values line. Returns {name: value}."""
    header, values = Path(path).read_text().splitlines()[:2]
    return dict(zip(header.split(), (float(v) for v in values.split())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[0, 1])
    parser.add_argument("--values", type=int, nargs="+", default=[1, 5, 10, 30, 60, 90, 150])
    parser.add_argument("--data", default="outputs/half/train")   # tune on the FIRST halves only
    args = parser.parse_args()

    print(f"stage {args.stage}, first halves, {len(SEQUENCES)} videos", flush=True)
    print("max_age   IDSW    IDF1    HOTA    IDs", flush=True)
    best_age, best_idf1 = None, -1.0
    for max_age in args.values:
        name = f"tune_stage{args.stage}_age{max_age}"
        for seq in SEQUENCES:
            subprocess.run([sys.executable, "-m", "scripts.run_tracker", seq, "--stage", str(args.stage),
                            "--max-age", str(max_age), "--data", args.data, "--out", f"outputs/{name}"],
                           check=True, capture_output=True)
        # TrackEval prints a lot: hide it, then read the numbers from its summary file
        subprocess.run([sys.executable, "-m", "scripts.evaluate", name, *SEQUENCES, "--gt", args.data],
                       check=True, capture_output=True)
        s = read_summary(f"outputs/{name}/pedestrian_summary.txt")
        print(f"{max_age:7d} {s['IDSW']:6.0f} {s['IDF1']:7.1f} {s['HOTA']:7.1f} {s['IDs']:6.0f}", flush=True)
        if s["IDF1"] > best_idf1 + 0.1:   # values go from small to large, so a tie keeps the smaller one
            best_age, best_idf1 = max_age, s["IDF1"]
    print(f"chosen max_age = {best_age} (IDF1 {best_idf1:.1f} on the first halves)", flush=True)


if __name__ == "__main__":
    main()
