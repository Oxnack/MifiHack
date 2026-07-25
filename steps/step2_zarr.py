"""
Step 2 — Упаковка в Zarr + валидация.
- Конвертация/проверка локального Zarr
- Валидация: подсчёт файлов, проверка размерностей, сверка каналов
- Создание манифеста (SHA256)
"""
import sys
import json
import hashlib
import numpy as np
import xarray as xr
from pathlib import Path

from config import (
    ZARR_DIR, CHANNEL_NAMES, N_CHANNELS, STATIC_FIELDS,
    GRID_025, GRID_05,
)
from utils.logger import get_logger, log_exception
from utils.checkpoints import is_step_done, mark_step_done, mark_step_failed

logger = get_logger(__name__, "step2_zarr.log")
STEP_NAME = "step2_zarr"

SUBSET_ZARR = ZARR_DIR / "era5_28ch_0p25_6h.zarr"
MANIFEST_FILE = ZARR_DIR / "manifest.json"
VALIDATION_REPORT = ZARR_DIR / "validation_report.json"


def _compute_sha256(filepath: Path) -> str:
    """SHA256 хеш файла."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _create_manifest(zarr_path: Path) -> dict:
    """Создаёт манифест Zarr-хранилища (список файлов + хеши)."""
    logger.info(f"Создание манифеста для {zarr_path} ...")
    manifest = {
        "zarr_path": str(zarr_path),
        "files": {},
    }
    all_files = sorted(zarr_path.rglob("*"))
    for fp in all_files:
        if fp.is_file():
            rel = str(fp.relative_to(zarr_path))
            sha = _compute_sha256(fp)
            size = fp.stat().st_size
            manifest["files"][rel] = {"sha256": sha, "size_bytes": size}

    manifest["total_files"] = len(manifest["files"])
    manifest["total_size_bytes"] = sum(v["size_bytes"] for v in manifest["files"].values())

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Манифест сохранён: {MANIFEST_FILE} (файлов: {manifest['total_files']})")
    return manifest


def _validate_zarr(ds: xr.Dataset) -> dict:
    """
    Валидирует Zarr-датасет.

    Проверяет:
    - наличие всех 28 каналов
    - размерности (time, latitude, longitude)
    - отсутствие NaN-заполнений
    - размеры сетки 0.25°
    """
    logger.info("Валидация Zarr-датасета...")
    report = {"valid": True, "warnings": [], "errors": []}

    # Проверка размерностей
    expected_dims = {"time", "latitude", "longitude"}
    actual_dims = set(ds.dims)
    report["dims"] = list(actual_dims)

    if not expected_dims.issubset(actual_dims):
        missing = expected_dims - actual_dims
        report["valid"] = False
        report["errors"].append(f"Отсутствуют размерности: {missing}")

    # Проверка размеров сетки 0.25°
    lat_size = ds.sizes.get("latitude", 0)
    lon_size = ds.sizes.get("longitude", 0)
    report["grid"] = {"lat": lat_size, "lon": lon_size}

    if lat_size != GRID_025["lat"] or lon_size != GRID_025["lon"]:
        report["warnings"].append(
            f"Размер сетки ({lat_size}×{lon_size}) != ожидаемой 0.25° "
            f"({GRID_025['lat']}×{GRID_025['lon']})"
        )

    # Проверка каналов
    found_channels = []
    missing_channels = []
    for ch in CHANNEL_NAMES:
        if ch in ds.data_vars:
            found_channels.append(ch)
        else:
            missing_channels.append(ch)

    report["n_channels_found"] = len(found_channels)
    report["channels_found"] = found_channels
    report["channels_missing"] = missing_channels

    if missing_channels:
        report["warnings"].append(f"Отсутствуют каналы: {missing_channels}")

    # Проверка статических полей
    static_found = [s for s in STATIC_FIELDS if s in ds.data_vars]
    report["static_fields_found"] = static_found

    # Проверка временного ряда
    if "time" in ds.dims:
        time_vals = ds.time.values
        report["time_start"] = str(time_vals[0])
        report["time_end"] = str(time_vals[-1])
        report["time_steps"] = len(time_vals)

        # Проверка на дубликаты/пропуски
        time_diffs = np.diff(time_vals.astype(np.int64))
        unique_diffs = np.unique(time_diffs)
        report["time_step_ns"] = [int(d) for d in unique_diffs]

    # Проверка NaN для нескольких каналов (выборочно)
    sample_vars = found_channels[:4] if found_channels else []
    for var in sample_vars:
        data = ds[var].isel(time=0).values
        nan_ratio = np.isnan(data).mean()
        if nan_ratio > 0.5:
            report["errors"].append(f"Канал {var}: {nan_ratio:.1%} NaN значений!")

    if report["errors"]:
        report["valid"] = False

    VALIDATION_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    status = "OK" if report["valid"] else "FAILED"
    logger.info(f"Валидация: {status} (каналов: {len(found_channels)}/{N_CHANNELS})")

    return report


def run() -> bool:
    """Выполняет шаг 2."""
    if is_step_done(STEP_NAME):
        logger.info(f"[{STEP_NAME}] Уже выполнен, пропускаю.")
        return True

    logger.info(f"=== [{STEP_NAME}] Запуск ===")

    try:
        # Проверяем наличие подмножества из шага 1
        if not SUBSET_ZARR.exists():
            logger.error(f"Подмножество не найдено: {SUBSET_ZARR}")
            logger.error("Сначала запусти step1_download.py")
            mark_step_failed(STEP_NAME, "Subset Zarr not found")
            return False

        # Открываем Zarr
        logger.info(f"Открытие {SUBSET_ZARR} ...")
        ds = xr.open_zarr(str(SUBSET_ZARR))
        logger.info(f"Датасет открыт: {dict(ds.dims)}, переменных: {len(ds.data_vars)}")

        # Валидация
        report = _validate_zarr(ds)

        # Манифест
        manifest = _create_manifest(SUBSET_ZARR)

        if not report["valid"]:
            logger.error("Валидация не пройдена! См. validation_report.json")
            mark_step_failed(STEP_NAME, "Validation failed")
            return False

        extra = {
            "n_channels": report["n_channels_found"],
            "time_steps": report.get("time_steps", 0),
            "grid": report["grid"],
            "manifest_files": manifest["total_files"],
            "manifest_size_mb": round(manifest["total_size_bytes"] / 1024 / 1024, 1),
        }
        mark_step_done(STEP_NAME, extra)
        logger.info(f"=== [{STEP_NAME}] Завершён успешно ===")
        return True

    except Exception as e:
        log_exception(logger, e, STEP_NAME)
        mark_step_failed(STEP_NAME, str(e))
        return False


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)