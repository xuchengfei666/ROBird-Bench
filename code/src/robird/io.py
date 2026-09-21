from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    value["_config_path"] = str(source)
    value["_config_hash"] = sha256_file(source)
    return value


def resolve_config_path(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    source = Path(str(config["_config_path"]))
    return (source.parent / path).resolve()


def protocol_status(config: Mapping[str, Any]) -> str:
    return str(config.get("protocol", {}).get("status", "missing")).strip().lower()


def require_frozen_protocol(config: Mapping[str, Any]) -> None:
    status = protocol_status(config)
    if status != "frozen":
        raise RuntimeError(
            f"Protocol is {status!r}, not 'frozen'. Execution and scientific decisions are disabled."
        )
    protocol = config.get("protocol", {})
    record_value = protocol.get("freeze_record")
    required_values = protocol.get("freeze_required", [])
    if not record_value or not isinstance(required_values, list) or not required_values:
        raise RuntimeError("Frozen protocol must declare freeze_record and freeze_required")
    record_path = resolve_config_path(config, str(record_value))
    if not record_path.is_file():
        raise RuntimeError(f"Protocol freeze record is missing: {record_path}")

    recorded: dict[Path, str] = {}
    with record_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise RuntimeError(f"Invalid freeze record line {line_number}: {raw_line.rstrip()}")
            expected_hash, recorded_name = parts
            recorded_name = recorded_name.lstrip("*").strip()
            target = Path(recorded_name)
            if not target.is_absolute():
                target = (record_path.parent / target).resolve()
            else:
                target = target.resolve()
            if target in recorded:
                raise RuntimeError(f"Duplicate freeze record target: {target}")
            recorded[target] = expected_hash.lower()

    required_paths = [resolve_config_path(config, str(value)) for value in required_values]
    for required_path in required_paths:
        expected = recorded.get(required_path)
        if expected is None:
            raise RuntimeError(f"Required file is absent from freeze record: {required_path}")
        actual = sha256_file(required_path)
        if actual.lower() != expected:
            raise RuntimeError(
                f"Frozen file hash mismatch for {required_path}: expected {expected}, got {actual}"
            )


def _atomic_text(path: Path, text: str, refuse_if_exists: bool) -> None:
    destination = Path(path)
    if refuse_if_exists and destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(
    path: Path, value: Mapping[str, Any] | list[Any], *, refuse_if_exists: bool = False
) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_text(Path(path), text, refuse_if_exists)


def atomic_write_csv(path: Path, frame: pd.DataFrame, *, refuse_if_exists: bool = False) -> None:
    _atomic_text(Path(path), frame.to_csv(index=False), refuse_if_exists)
