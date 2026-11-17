"""Score tracker output files against MOT17 ground truth with TrackEval.

Example (from the repo folder):
    python -m scripts.evaluate stage0 MOT17-09-FRCNN
It reads outputs/<tracker>/<sequence>.txt and data/MOT17/train/<sequence>/gt/gt.txt.
"""
import argparse
from pathlib import Path

import trackeval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracker", help="folder name inside outputs/, for example stage0")
    parser.add_argument("sequences", nargs="+", help="for example MOT17-09-FRCNN")
    parser.add_argument("--gt", default="data/MOT17/train")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()

    # TrackEval reads the list of videos from a "seqmap" file: a header line, then one name per line
    seqmap = Path(args.outputs) / args.tracker / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(args.sequences) + "\n")

    dataset = trackeval.datasets.MotChallenge2DBox({
        **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
        "GT_FOLDER": args.gt,                # <gt>/<sequence>/gt/gt.txt and seqinfo.ini
        "TRACKERS_FOLDER": args.outputs,     # <outputs>/<tracker>/<sequence>.txt
        "TRACKERS_TO_EVAL": [args.tracker],
        "TRACKER_SUB_FOLDER": "",
        "SEQMAP_FILE": str(seqmap),
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": False,
    })
    evaluator = trackeval.Evaluator({
        **trackeval.Evaluator.get_default_eval_config(),
        "USE_PARALLEL": False,
        "PRINT_CONFIG": False,
        "PLOT_CURVES": False,
    })
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator.evaluate([dataset], metrics)


if __name__ == "__main__":
    main()
