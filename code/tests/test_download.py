from __future__ import annotations

from PIL import Image

from robird.download import inspect_image


def test_inspect_image_reports_dimensions_and_hash(tmp_path) -> None:
    path = tmp_path / "test.jpg"
    Image.new("RGB", (17, 11), color=(20, 40, 60)).save(path)
    result = inspect_image(path)
    assert result["width"] == 17
    assert result["height"] == 11
    assert result["bytes"] > 0
    assert len(result["sha256"]) == 64
