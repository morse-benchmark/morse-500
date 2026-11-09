"""Sample, cache, and validate frames shared by model and agent runs."""

import argparse
import base64
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import sys

from moviepy import VideoFileClip
import numpy as np
from PIL import Image


def frame_times(duration, fps, max_frames):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid clip duration: {duration}")
    times = np.arange(0, duration, 1 / fps)
    if len(times) > max_frames:
        times = times[np.linspace(0, len(times) - 1, max_frames, dtype=int)]
    return times


def encode_frame(frame, max_side):
    image = Image.fromarray(frame).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def sample_video(path, fps, max_frames, max_side):
    with VideoFileClip(str(path), audio=False) as clip:
        times = frame_times(clip.duration, fps, max_frames)
        return [
            encode_frame(clip.get_frame(float(t)), max_side)
            for t in times
        ]


def cache_policy(args):
    return {
        "version": 1, "fps": args.fps, "max_frames": args.max_frames,
        "max_side": args.max_side, "selection": "uniform_full_clip_grid",
        "jpeg_quality": 90,
    }


def cache_path(args, video):
    name = hashlib.sha256(video.encode("utf-8")).hexdigest()
    return args.frame_cache / f"{name}.json"


def source_identity(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_cached_frames(args, video):
    path = cache_path(args, video)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frame cache for {video}; run prepare first: {path}")
    record = json.loads(path.read_text())
    if (
        record["policy"] != cache_policy(args)
        or record["video"] != video
        or record["source"] != source_identity(args.video_root / video)
    ):
        raise ValueError(f"Stale/incompatible frame cache for {video}; use a new cache directory")
    frames = record["frames"]
    if not 1 <= len(frames) <= args.max_frames:
        raise ValueError(f"Invalid cached frame count for {video}")
    digest = hashlib.sha256("".join(frames).encode("ascii")).hexdigest()
    if digest != record["frames_sha256"]:
        raise ValueError(f"Corrupt frame cache for {video}")
    return frames


def prepare_cache(args, rows):
    args.frame_cache.mkdir(parents=True, exist_ok=True)
    # Serialize cache writers across provider jobs; readers see atomic entries.
    with (args.frame_cache / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, row in enumerate(rows, 1):
            video = row["video"]
            destination = cache_path(args, video)
            if destination.exists():
                load_cached_frames(args, video)
                print(f"Cache verified {index}/{len(rows)}: {video}", flush=True)
                continue
            source = args.video_root / video
            identity = source_identity(source)
            frames = sample_video(source, args.fps, args.max_frames, args.max_side)
            if source_identity(source) != identity:
                raise ValueError(f"Video changed during preprocessing: {source}")
            record = {
                "policy": cache_policy(args), "video": video,
                "source": identity, "frames": frames,
                "frames_sha256": hashlib.sha256("".join(frames).encode("ascii")).hexdigest(),
            }
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(record) + "\n")
            temporary.replace(destination)
            print(
                f"Cached {index}/{len(rows)}: {video} ({len(frames)} frames)",
                flush=True,
            )


def main(argv=None):
    from lib import common

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    common.add_data_arguments(parser)
    args = parser.parse_args(argv)
    if not args.frame_cache:
        parser.error("prepare requires --frame-cache")
    common.validate_positive_arguments(
        parser, args, ("fps", "max_frames", "max_side", "limit", "expected_count")
    )
    prepare_cache(args, common.load_rows(args))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
