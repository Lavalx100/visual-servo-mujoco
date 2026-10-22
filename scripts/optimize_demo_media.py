"""Reduce checked-in simulation-video size while retaining linked demo files.

Requires ffmpeg in PATH. Use --paths to limit the operation to a reviewed set;
with no paths, every MP4 and GIF below assets/demos is optimized in place.
MP4 dimensions and frame rate are preserved. GIF previews are resized to at
most 768 pixels wide and five frames per second.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def _encode_mp4(source: Path, destination: Path, crf: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def _encode_gif(source: Path, destination: Path, width: int, fps: int) -> None:
    filters = (
        f"fps={fps},scale='min({width},iw)':-2:flags=lanczos,"
        "split[base][palette_source];"
        "[palette_source]palettegen=stats_mode=diff:max_colors=128[palette];"
        "[base][palette]paletteuse=dither=bayer:bayer_scale=3"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            filters,
            "-loop",
            "0",
            str(destination),
        ],
        check=True,
    )


def optimize(path: Path, *, crf: int, gif_width: int, gif_fps: int) -> tuple[int, int]:
    path = path.resolve()
    if path.suffix.lower() not in {".mp4", ".gif"}:
        raise ValueError(f"unsupported demo-media format: {path}")
    original_size = path.stat().st_size
    with tempfile.TemporaryDirectory(prefix="demo-media-optimization-", dir=path.parent) as scratch:
        suffix = ".mp4" if path.suffix.lower() == ".mp4" else ".gif"
        encoded = Path(scratch) / f"optimized{suffix}"
        if suffix == ".mp4":
            _encode_mp4(path, encoded, crf)
        else:
            _encode_gif(path, encoded, gif_width, gif_fps)
        encoded_size = encoded.stat().st_size
        if encoded_size < original_size:
            encoded.replace(path)
            return original_size, encoded_size
        return original_size, original_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", type=Path, help="specific demo files to process")
    parser.add_argument(
        "--untracked-only",
        action="store_true",
        help="process only untracked files below the demo directory (requires Git)",
    )
    parser.add_argument("--directory", type=Path, default=Path("assets/demos"))
    parser.add_argument("--mp4-crf", type=int, default=24)
    parser.add_argument("--gif-width", type=int, default=768)
    parser.add_argument("--gif-fps", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.mp4_crf <= 51 or args.gif_width <= 0 or args.gif_fps <= 0:
        parser.error("CRF must be 0–51 and GIF width/FPS must be positive")
    if args.paths and args.untracked_only:
        parser.error("use either --paths or --untracked-only")
    if args.paths:
        paths = args.paths
    elif args.untracked_only:
        output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", str(args.directory)],
            check=True,
            capture_output=True,
            text=True,
        )
        paths = sorted(
            Path(line)
            for line in output.stdout.splitlines()
            if Path(line).suffix.lower() in {".mp4", ".gif"}
        )
    else:
        paths = sorted(
            path for path in args.directory.rglob("*") if path.suffix.lower() in {".mp4", ".gif"}
        )
    total_before = total_after = 0
    for path in paths:
        before, after = optimize(
            path, crf=args.mp4_crf, gif_width=args.gif_width, gif_fps=args.gif_fps
        )
        total_before += before
        total_after += after
        print(f"{path}: {before / (1024 * 1024):.2f} MiB -> {after / (1024 * 1024):.2f} MiB")
    print(
        f"Total: {total_before / (1024 * 1024):.2f} MiB -> "
        f"{total_after / (1024 * 1024):.2f} MiB "
        f"({100 * (total_before - total_after) / total_before:.1f}% saved)"
        if total_before
        else "No media files selected"
    )


if __name__ == "__main__":
    main()
