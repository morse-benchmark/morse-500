"""Write a model-specific input CSV containing only failed/missing predictions."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import read_csv, save_csv, successful


def select_failed_rows(source_rows, prediction_rows):
    if not source_rows:
        raise ValueError("Source input CSV is empty")
    if not prediction_rows:
        raise ValueError("Prediction CSV is empty")

    source_by_video = {}
    for row in source_rows:
        video = row.get("video", "")
        if not video or video in source_by_video:
            raise ValueError(f"Source input has missing or duplicate video: {video!r}")
        source_by_video[video] = row

    prediction_by_video = {}
    for row in prediction_rows:
        video = row.get("video", "")
        if not video or video in prediction_by_video:
            raise ValueError(
                f"Prediction CSV has missing or duplicate video: {video!r}"
            )
        if video not in source_by_video:
            raise ValueError(f"Prediction CSV has unexpected video: {video}")
        prediction_by_video[video] = row

    failed = []
    for index, row in enumerate(source_rows, 1):
        prediction = prediction_by_video.get(row["video"], {})
        if not successful(prediction.get("prediction", "")):
            failed.append(dict(row, idx=str(row.get("idx") or row.get("id") or index)))
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original benchmark CSV")
    parser.add_argument(
        "--predictions", type=Path, required=True, help="Original model prediction CSV"
    )
    parser.add_argument("--output", type=Path, required=True, help="Failed-only input CSV")
    args = parser.parse_args(argv)

    if args.output.resolve() in {args.input.resolve(), args.predictions.resolve()}:
        parser.error("--output must differ from the source and prediction CSVs")

    source_rows = read_csv(args.input)
    prediction_rows = read_csv(args.predictions)
    failed = select_failed_rows(source_rows, prediction_rows)
    if not failed:
        args.output.unlink(missing_ok=True)
        print(f"No failed rows in {args.predictions}; no recovery input written")
        return

    fieldnames = list(source_rows[0])
    if "idx" not in fieldnames:
        fieldnames.append("idx")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_csv(args.output, failed, fieldnames)
    print(f"Prepared {len(failed)} failed rows: {args.output}")


if __name__ == "__main__":
    main()
