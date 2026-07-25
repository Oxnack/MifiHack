"""
Step 1 — Загрузка датасетов (LIGHT: последние 2 года).
Скачивает только 2022–2023 из WeatherBench2 (анонимный GCS доступ),
сразу извлекает 28 каналов + статику, сохраняет в локальный Zarr.
Без промежуточной загрузки полного датасета — работает потоково через dask.

Итоговый размер: ~30–60 GB (0.25°, сжатый Zarr, зависит от компрессора).
"""
import sys
import json
import numpy as np
import xarray as xr
import gcsfs
import zarr
from pathlib import Path
from datetime import datetime

from config import (
    WB2_ERA5_ZARR, ZARR_DIR,
    SURFACE_VARS, PRESSURE_VARS, PRESSURE_LEVELS,
    STATIC_FIELDS, CHANNEL_NAMES,
)
from utils.logger import get_logger, log_exception
from utils.checkpoints import is_step_done, mark_step_done, mark_step_failed

logger = get_logger(__name__, "step1_download.log")
STEP_NAME = "step1_download"

# ── Что скачиваем ─────────────────────────────────────
TIME_START = "2022"
TIME_END   = "2023"
OUTPUT_ZARR = ZARR_DIR / "era5_28ch_0p25_6h_2022_2023.zarr"
MANIFEST_FILE = ZARR_DIR / "manifest_2022_2023.json"

# Сжатие: zstd level 3 — хороший баланс скорость/размер
COMPRESSOR = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE)

# Размер чанка: 1 месяц (≈120 таймстепов по 6ч) × полный грид
CHUNK_TIME = 120   # ~1 месяц
CHUNK_LAT = 721    # полная высота
CHUNK_LON = 1440   # полная ширина


def _open_remote():
    """Подключается к удалённому WeatherBench2 Zarr (анонимно)."""
    logger.info(f"Подключение к GCS: {WB2_ERA5_ZARR}")
    fs = gcsfs.GCSFileSystem(token="anon")
    path = WB2_ERA5_ZARR.replace("gs://", "")
    mapper = fs.get_mapper(path)
    ds = xr.open_zarr(mapper, consolidated=True)
    logger.info(f"Удалённый датасет открыт: dims={dict(ds.dims)}")
    return ds


def _extract_subset(ds: xr.Dataset) -> Path:
    """
    Извлекает 28 каналов + статику за TIME_START–TIME_END
    и потоково сохраняет в локальный Zarr.
    """
    if OUTPUT_ZARR.exists():
        logger.info(f"Выходной Zarr уже существует: {OUTPUT_ZARR}")
        return OUTPUT_ZARR

    # ── Фильтр по времени ──
    ds_period = ds.sel(time=slice(TIME_START, TIME_END))
    n_times = ds_period.sizes["time"]
    logger.info(f"Временной диапазон: {TIME_START}–{TIME_END}, таймстепов: {n_times}")

    # ── Сбор переменных ──
    data_vars = {}

    # Surface (8 каналов)
    for var in SURFACE_VARS:
        if var in ds.data_vars:
            data_vars[var] = ds_period[var]
            logger.info(f"  ✓ surface: {var}")
        else:
            logger.warning(f"  ✗ surface: {var} — не найдена!")

    # Pressure (5 переменных × 4 уровня = 20 каналов)
    for var in PRESSURE_VARS:
        if var in ds.data_vars:
            var_data = ds_period[var]
            if "level" in var_data.dims:
                var_sel = var_data.sel(level=PRESSURE_LEVELS)
                for lev in PRESSURE_LEVELS:
                    lev_name = f"{var}_{lev}"
                    lev_data = var_sel.sel(level=lev, drop=True)
                    data_vars[lev_name] = lev_data
                    logger.info(f"  ✓ pressure: {lev_name}")
            else:
                logger.warning(f"  ✗ pressure: {var} — нет размерности level")
        else:
            logger.warning(f"  ✗ pressure: {var} — не найдена!")

    # Статические поля
    for sfield in STATIC_FIELDS:
        if sfield in ds.data_vars:
            data_vars[sfield] = ds[sfield]
            logger.info(f"  ✓ static: {sfield}")
        elif sfield == "sin_lat":
            lat = ds.latitude.values
            lat_rad = np.deg2rad(lat)
            data_vars["sin_lat"] = xr.DataArray(
                np.sin(lat_rad), dims=["latitude"], coords={"latitude": lat}
            )
            logger.info("  ✓ static: sin_lat (вычислен)")
        elif sfield == "cos_lat":
            lat = ds.latitude.values
            lat_rad = np.deg2rad(lat)
            data_vars["cos_lat"] = xr.DataArray(
                np.cos(lat_rad), dims=["latitude"], coords={"latitude": lat}
            )
            logger.info("  ✓ static: cos_lat (вычислен)")

    # ── Сборка датасета ──
    subset_ds = xr.Dataset(data_vars, attrs=ds.attrs)
    subset_ds.attrs["source"] = WB2_ERA5_ZARR
    subset_ds.attrs["time_range"] = f"{TIME_START}-{TIME_END}"
    subset_ds.attrs["extracted_at"] = datetime.now().isoformat()
    subset_ds.attrs["n_channels"] = len(data_vars)
    subset_ds.attrs["channel_names"] = json.dumps(list(data_vars.keys()))

    n_vars = len(data_vars)
    logger.info(f"Собрано {n_vars} переменных (ожидается ≥ 28)")

    # ── Чанкинг и сохранение (потоково через dask) ──
    subset_ds = subset_ds.chunk({
        "time": CHUNK_TIME,
        "latitude": CHUNK_LAT,
        "longitude": CHUNK_LON,
    })

    # Создаём encoding со сжатием для каждой переменной
    encoding = {}
    for vname in subset_ds.data_vars:
        encoding[vname] = {"compressor": COMPRESSOR}

    logger.info(f"Сохранение в {OUTPUT_ZARR} (потоково, со сжатием zstd-3)...")
    logger.info("  Это может занять 30–120 минут в зависимости от канала связи.")
    logger.info("  Данные качаются чанками, прогресс виден в логах dask.")

    # Запускаем вычисление — dask сам подтянет чанки по сети
    subset_ds.to_zarr(str(OUTPUT_ZARR), mode="w", encoding=encoding)

    # Оценка размера
    total_size = sum(
        f.stat().st_size for f in OUTPUT_ZARR.rglob("*") if f.is_file()
    )
    logger.info(f"Сохранено: {OUTPUT_ZARR} ({total_size / 1024**3:.1f} GB)")

    return OUTPUT_ZARR


def _create_manifest(zarr_path: Path):
    """Создаёт JSON-манифест с метаданными."""
    ds = xr.open_zarr(str(zarr_path))
    total_size = sum(
        f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
    )

    manifest = {
        "zarr_path": str(zarr_path),
        "source": WB2_ERA5_ZARR,
        "time_range": f"{TIME_START}–{TIME_END}",
        "time_steps": int(ds.sizes["time"]),
        "grid": {
            "lat": int(ds.sizes["latitude"]),
            "lon": int(ds.sizes["longitude"]),
        },
        "n_variables": len(ds.data_vars),
        "variables": sorted(ds.data_vars.keys()),
        "channel_names_28ch": CHANNEL_NAMES,
        "static_fields": STATIC_FIELDS,
        "total_size_gb": round(total_size / 1024**3, 2),
        "compressor": "zstd level=3 shuffle=SHUFFLE",
        "created_at": datetime.now().isoformat(),
        "time_values": {
            "first": str(ds.time.values[0]),
            "last": str(ds.time.values[-1]),
        },
    }

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Манифест: {MANIFEST_FILE}")

    ds.close()
    return manifest


def validate():
    """Быстрая валидация скачанного датасета."""
    if not OUTPUT_ZARR.exists():
        logger.error(f"Датасет не найден: {OUTPUT_ZARR}")
        return False

    ds = xr.open_zarr(str(OUTPUT_ZARR))
    ok = True

    # Размерности
    if ds.sizes.get("latitude") != 721 or ds.sizes.get("longitude") != 1440:
        logger.warning(f"Сетка {ds.sizes.get('latitude')}×{ds.sizes.get('longitude')} != 721×1440")
        ok = False

    # Каналы
    missing = [ch for ch in CHANNEL_NAMES if ch not in ds.data_vars]
    if missing:
        logger.warning(f"Отсутствуют каналы: {missing}")
        ok = False

    n_times = ds.sizes.get("time", 0)
    n_vars = len(ds.data_vars)
    logger.info(f"Валидация: {n_times} таймстепов, {n_vars} переменных → {'OK' if ok else 'WARN'}")

    ds.close()
    return ok


def run(skip_if_exists: bool = True) -> bool:
    """
    Шаг 1: скачать 2 года ERA5 (28 каналов + статика) из WeatherBench2.

    Args:
        skip_if_exists: пропустить, если OUTPUT_ZARR уже существует

    Returns:
        True если всё OK
    """
    if is_step_done(STEP_NAME):
        logger.info(f"[{STEP_NAME}] Уже выполнен, пропускаю.")
        return True

    if skip_if_exists and OUTPUT_ZARR.exists():
        logger.info(f"[{STEP_NAME}] {OUTPUT_ZARR} уже существует, пропускаю загрузку.")
        mark_step_done(STEP_NAME, {"zarr": str(OUTPUT_ZARR), "status": "already_exists"})
        return True

    logger.info(f"=== [{STEP_NAME}] Запуск (LIGHT: {TIME_START}–{TIME_END}) ===")

    try:
        ds = _open_remote()
        zarr_path = _extract_subset(ds)
        _create_manifest(zarr_path)
        valid = validate()

        extra = {
            "zarr": str(zarr_path),
            "time_range": f"{TIME_START}–{TIME_END}",
            "valid": valid,
        }
        mark_step_done(STEP_NAME, extra)
        logger.info(f"=== [{STEP_NAME}] Завершён успешно ===")
        return valid

    except Exception as e:
        log_exception(logger, e, STEP_NAME)
        mark_step_failed(STEP_NAME, str(e))
        return False


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)