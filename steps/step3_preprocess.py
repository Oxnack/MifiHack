"""
Step 3 — Предобработка данных.
- Приведение к единой временной шкале (6h)
- Feature engineering (wind speed, wind direction)
- Нормализация (StandardScaler: mean/std по train)
- Хронологическое разбиение train/val/test
- Сохранение статистик и индексов
"""
import sys
import json
import pickle
import numpy as np
import xarray as xr
from pathlib import Path

from config import (
    ZARR_DIR, PROCESSED_DIR,
    CHANNEL_NAMES, N_CHANNELS, STATIC_FIELDS,
    TIME_TRAIN, TIME_VAL, TIME_TEST,
    DATASET_SIZES,
)
from utils.logger import get_logger, log_exception
from utils.checkpoints import is_step_done, mark_step_done, mark_step_failed

logger = get_logger(__name__, "step3_preprocess.log")
STEP_NAME = "step3_preprocess"

SUBSET_ZARR = ZARR_DIR / "era5_28ch_0p25_6h_2022_2023.zarr"

# Выходные файлы
TRAIN_DATA = PROCESSED_DIR / "train.npy"          # (T_train, C, H, W) — пока не делаем .npy (слишком большой)
TRAIN_INDICES = PROCESSED_DIR / "train_indices.json"
VAL_INDICES = PROCESSED_DIR / "val_indices.json"
TEST_INDICES = PROCESSED_DIR / "test_indices.json"
STATS_FILE = PROCESSED_DIR / "stats.json"          # mean, std по каждому каналу
SCALER_FILE = PROCESSED_DIR / "scaler.pkl"         # StandardScaler или словарь mean/std
OCEAN_MASK_FILE = PROCESSED_DIR / "ocean_mask.npy"
LAT_WEIGHTS_FILE = PROCESSED_DIR / "lat_weights.npy"


def _compute_mean_std(ds: xr.Dataset, train_indices: list) -> dict:
    """
    Вычисляет mean и std по обучающей выборке для каждого канала.
    Читает данные побаточно (чанками) для экономии памяти.

    Returns:
        dict: {channel: {"mean": float, "std": float}}
    """
    logger.info("Вычисление mean/std по train-выборке...")
    stats = {}

    for ch in CHANNEL_NAMES:
        if ch not in ds.data_vars:
            logger.warning(f"  Канал {ch} не найден, пропускаю.")
            continue

        # Берём данные по train-индексам
        data = ds[ch].isel(time=train_indices).values  # (T, H, W)
        # Конвертируем в float32
        data = data.astype(np.float32)

        # Убираем NaN (если есть) — заменяем на 0 для расчёта статистик
        data = np.nan_to_num(data, nan=0.0)

        mean = float(np.mean(data))
        std = float(np.std(data))
        if std == 0:
            std = 1.0  # избегаем деления на 0

        stats[ch] = {"mean": mean, "std": std}
        logger.info(f"  {ch}: mean={mean:.4f}, std={std:.4f}")

    return stats


def _normalize_channel(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Нормализация: (x - mean) / std."""
    return (data - mean) / max(std, 1e-8)


def _create_ocean_mask(ds: xr.Dataset) -> np.ndarray:
    """
    Создаёт океанскую маску из land_sea_mask (если есть).
    ocean = 1 - land_sea_mask (где 0 — океан, 1 — суша или наоборот).
    """
    if "land_sea_mask" in ds.data_vars:
        lsm = ds["land_sea_mask"].isel(time=0).values.astype(np.float32)
        # В ERA5: lsm = 1 — суша, 0 — океан
        ocean_mask = 1.0 - lsm
        logger.info(f"Океанская маска создана: ocean ratio = {ocean_mask.mean():.3f}")
        return ocean_mask
    else:
        logger.warning("land_sea_mask не найдена, маска не создана.")
        return np.ones((ds.sizes["latitude"], ds.sizes["longitude"]), dtype=np.float32)


def _compute_lat_weights(ds: xr.Dataset) -> np.ndarray:
    """Вычисляет latitude-веса: cos(lat)."""
    lat = ds.latitude.values.astype(np.float32)
    lat_rad = np.deg2rad(lat)
    weights = np.cos(lat_rad)
    weights = np.clip(weights, 0, None)
    # Нормализуем
    weights = weights / weights.sum()
    logger.info(f"Latitude-веса: min={weights.min():.6f}, max={weights.max():.6f}")
    return weights


def _split_indices(ds: xr.Dataset) -> tuple:
    """
    Хронологическое разбиение индексов по времени.

    Returns:
        (train_idx, val_idx, test_idx) — списки индексов
    """
    time_vals = ds.time.values
    n_total = len(time_vals)
    logger.info(f"Всего временных шагов: {n_total}")

    train_idx = []
    val_idx = []
    test_idx = []

    for i, t in enumerate(time_vals):
        year = int(str(t)[:4])
        train_start, train_end = int(TIME_TRAIN[0]), int(TIME_TRAIN[1])
        val_start, val_end = int(TIME_VAL[0]), int(TIME_VAL[1])
        test_start, test_end = int(TIME_TEST[0]), int(TIME_TEST[1])

        if train_start <= year <= train_end:
            train_idx.append(i)
        elif val_start <= year <= val_end:
            val_idx.append(i)
        elif test_start <= year <= test_end:
            test_idx.append(i)

    # Если разбиение пустое (например, для 2022–2023 при старых диапазонах),
    # используем автоматическое: первый год — train, второй год — val/test пополам
    if not train_idx and not val_idx and not test_idx:
        logger.warning("Стандартное разбиение пустое! Использую авто-разбиение: "
                        "год0=train, год1=val/test 50/50")
        years_in_data = sorted(set(int(str(t)[:4]) for t in time_vals))
        for i, t in enumerate(time_vals):
            year = int(str(t)[:4])
            if year == years_in_data[0]:
                train_idx.append(i)
            elif len(years_in_data) > 1 and year == years_in_data[1]:
                # Первая половина года — val, вторая — test
                month = int(str(t)[5:7])
                if month <= 6:
                    val_idx.append(i)
                else:
                    test_idx.append(i)
        # Обновляем TIME_TRAIN/VAL/TEST для логов
        if years_in_data:
            TIME_TRAIN = (str(years_in_data[0]), str(years_in_data[0]))
            if len(years_in_data) > 1:
                TIME_VAL = (str(years_in_data[1]), str(years_in_data[1]))
                TIME_TEST = (str(years_in_data[1]), str(years_in_data[1]))

    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Создаём сезонно-сбалансированные поднаборы для train
    subsets = {}
    for N in DATASET_SIZES:
        if N <= len(train_idx):
            # Равномерная выборка с учётом сезонности (по месяцам)
            step = max(1, len(train_idx) // N)
            subset = train_idx[::step][:N]
            subsets[f"N{N}"] = subset
            logger.info(f"  Subset N={N}: {len(subset)} индексов")
        else:
            subsets[f"N{N}"] = train_idx[:N] if N <= len(train_idx) else train_idx
            logger.warning(f"  Subset N={N}: запрошено {N}, доступно {len(train_idx)}")

    return train_idx, val_idx, test_idx, subsets


def _feature_engineering(ds: xr.Dataset) -> None:
    """
    Добавляет производные признаки (in-place в датасет не пишем, 
    вычисляем на лету при обучении). Логируем только информацию.
    """
    logger.info("Feature engineering (информационно):")
    logger.info("  - wind_speed = sqrt(u10² + v10²)  [вычисляется на лету]")
    logger.info("  - wind_direction = atan2(v10, u10)  [вычисляется на лету]")


def run() -> bool:
    """Выполняет шаг 3."""
    if is_step_done(STEP_NAME):
        logger.info(f"[{STEP_NAME}] Уже выполнен, пропускаю.")
        return True

    logger.info(f"=== [{STEP_NAME}] Запуск ===")

    try:
        # Проверяем наличие данных
        if not SUBSET_ZARR.exists():
            logger.error(f"Датасет не найден: {SUBSET_ZARR}")
            mark_step_failed(STEP_NAME, "Zarr not found")
            return False

        # Открываем Zarr
        ds = xr.open_zarr(str(SUBSET_ZARR))
        logger.info(f"Датасет открыт: {dict(ds.dims)}")

        # 1. Разбивка индексов
        train_idx, val_idx, test_idx, subsets = _split_indices(ds)

        # Сохраняем индексы
        train_indices = {"indices": train_idx, "subsets": {k: list(v) for k, v in subsets.items()}}
        val_indices = {"indices": val_idx}
        test_indices = {"indices": test_idx}

        TRAIN_INDICES.write_text(json.dumps(train_indices))
        VAL_INDICES.write_text(json.dumps(val_indices))
        TEST_INDICES.write_text(json.dumps(test_indices))
        logger.info(f"Индексы сохранены: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # 2. Feature engineering (информационно)
        _feature_engineering(ds)

        # 3. Статистики нормализации
        stats = _compute_mean_std(ds, train_idx)
        STATS_FILE.write_text(json.dumps(stats, indent=2))
        logger.info(f"Статистики сохранены: {STATS_FILE}")

        # 4. Океанская маска
        ocean_mask = _create_ocean_mask(ds)
        np.save(OCEAN_MASK_FILE, ocean_mask)

        # 5. Latitude-веса
        lat_weights = _compute_lat_weights(ds)
        np.save(LAT_WEIGHTS_FILE, lat_weights)

        extra = {
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "n_channels": len(stats),
            "subsets": {k: len(v) for k, v in subsets.items()},
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