from __future__ import annotations

import http.client
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from .io import sha256_file


USER_AGENT = "ROBird-Bench public licensed image download (academic research)"


def inspect_image(path: Path) -> dict[str, Any]:
    source = Path(path)
    with Image.open(source) as image:
        image.verify()
    with Image.open(source) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions for {source}: {width}x{height}")
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "width": int(width),
        "height": int(height),
        "bytes": int(source.stat().st_size),
    }


def download_image(url: str, destination: Path, attempts: int, timeout: float) -> dict[str, Any]:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return {**inspect_image(target), "reused_existing": True}

    errors: list[str] = []
    for attempt in range(attempts):
        temporary: Path | None = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(
                request, timeout=timeout
            ) as response:
                while block := response.read(1024 * 1024):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            inspection = inspect_image(temporary)
            if target.exists():
                temporary.unlink(missing_ok=True)
                return {**inspect_image(target), "reused_existing": True}
            os.replace(temporary, target)
            return {
                **inspection,
                "path": str(target.resolve()),
                "reused_existing": False,
            }
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            OSError,
            ValueError,
        ) as error:
            errors.append(f"{type(error).__name__}: {error}")
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"Image download failed after {attempts} attempts: {errors[-1]}")
