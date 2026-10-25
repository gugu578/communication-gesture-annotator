# SPDX-License-Identifier: Apache-2.0
"""PNG transport preserving the recorded FFmpeg-decoded RGB24 raster."""

from hashlib import sha256
from pathlib import Path
import struct
import subprocess


def optimize_png(source: Path, width: int, height: int) -> tuple[bytes, str]:
    """Recompress generated RGB PNGs, verifying every decoded pixel."""
    import cv2
    import numpy as np

    original = source.read_bytes()
    if not original.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("size recovery requires PNG input")
    pixels = cv2.imdecode(np.frombuffer(original, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if (
        pixels is None
        or pixels.shape != (height, width, 3)
        or pixels.dtype != np.uint8
    ):
        raise ValueError("size recovery requires an RGB8 PNG of the recorded dimensions")
    ok, encoded = cv2.imencode(
        ".png", pixels,
        [cv2.IMWRITE_PNG_COMPRESSION, 9,
         cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_FILTERED],
    )
    if not ok:
        raise ValueError("PNG compression failed")
    # Some already compressed images cannot shrink; never make those larger.
    result = encoded.tobytes() if len(encoded) < len(original) else original
    decoded = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if not np.array_equal(pixels, decoded):
        raise ValueError("PNG compression changed decoded pixels")
    rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB).tobytes()
    return result, sha256(rgb).hexdigest()


def transcode(source: Path, width: int, height: int, ffmpeg: str) -> tuple[bytes, str]:
    """Preserve the source's FFmpeg-decoded rgb24 raster without scaling."""
    if (
        type(width) is not int
        or type(height) is not int
        or (not (0 < width <= 8192 and 0 < height <= 8192))
    ):
        raise ValueError("unsupported image dimensions")
    if width * height > 16777216:
        raise ValueError("diagnostic image exceeds pixel budget")

    def run(arguments: list[str], data: bytes | None = None) -> bytes:
        try:
            result = subprocess.run(
                [ffmpeg, "-v", "error", "-nostdin", "-threads", "1", *arguments],
                input=data,
                capture_output=True,
                check=True,
                timeout=60,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise ValueError("local image conversion or verification failed") from exc
        return result.stdout

    rgb = run(
        [
            "-i",
            str(source.resolve()),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    if len(rgb) != width * height * 3:
        raise ValueError("decoded image dimensions do not match the manifest")
    png = run(
        [
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-c:v",
            "png",
            "-f",
            "image2pipe",
            "pipe:1",
        ],
        rgb,
    )
    if png[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", png[16:24]) != (width, height):
        raise ValueError("encoded PNG dimensions do not match")
    decoded = run(
        ["-i", "pipe:0", "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], png
    )
    if decoded != rgb:
        raise ValueError("PNG did not preserve every decoded RGB pixel")
    return (png, sha256(rgb).hexdigest())


def resize_png(source: Path, width: int, height: int, percent: int) -> tuple[bytes, str]:
    """Area-resample the whole raster; PNG encoding then preserves that raster.

    This is spatially lossy, not a lossless compression operation. No crop or
    temporal sampling is performed. Callers record both source and output sizes.
    """
    import cv2
    import numpy as np

    if percent not in (85, 70):
        raise ValueError("unsupported PNG size-recovery scale")
    pixels = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.shape != (height, width, 3) or pixels.dtype != np.uint8:
        raise ValueError("spatial recovery requires an RGB8 PNG of the recorded dimensions")
    size = (max(1, (width * percent + 50) // 100), max(1, (height * percent + 50) // 100))
    resized = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", resized, [cv2.IMWRITE_PNG_COMPRESSION, 9,
        cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_FILTERED])
    if not ok:
        raise ValueError("resized PNG encoding failed")
    if not np.array_equal(resized, cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)):
        raise ValueError("PNG did not preserve the resized raster")
    return encoded.tobytes(), sha256(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
