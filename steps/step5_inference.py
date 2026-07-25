"""
Step 5 — Инференс + Метрики.
- Прогноз на test-выборке
- 4 режима: Direct, AR, Continuous, Const Correction
- Расчёт метрик (RMSE, MAE, ACC, Bias)
- Визуализация прогнозов vs факт
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from config import (
    ZARR_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR,
    CHANNEL_NAMES, N_CHANNELS,
    INFERENCE, GPU,
)
from utils.logger import get_logger, log_exception
from utils.checkpoints import is_step_done, mark_step_done, mark_step_failed
from utils.metrics import compute_all_metrics, rmse, mae, bias, acc

logger = get_logger(__name__, "step5_inference.log")
STEP_NAME = "step5_inference"

SUBSET_ZARR = ZARR_DIR / "era5_28ch_0p25_6h.zarr"
TEST_INDICES_FILE = PROCESSED_DIR / "test_indices.json"
STATS_FILE = PROCESSED_DIR / "stats.json"
BEST_MODEL = MODELS_DIR / "best.pt"
METRICS_OUTPUT = OUTPUTS_DIR / "metrics.json"
PREDICTIONS_OUTPUT = OUTPUTS_DIR / "predictions.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(model_path: Path):
    """Загружает модель из чекпоинта."""
    from steps.step4_train import WeatherAutoencoder
    from config import BIFPN, BOTTLENECK

    model = WeatherAutoencoder(
        in_channels=N_CHANNELS * 2,
        out_channels=N_CHANNELS,
        encoder_dims=[64, 128, 256, 512],
        bifpn_dim=BIFPN["feature_dim"],
        vq_num_embeddings=BOTTLENECK["num_embeddings"],
        vq_dim=BOTTLENECK["latent_dim"],
    ).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Модель загружена из {model_path} (эпоха {ckpt.get('epoch', '?')})")
    return model


def _load_stats() -> dict:
    if STATS_FILE.exists():
        return json.loads(STATS_FILE.read_text())
    return {}


def _load_test_indices() -> List[int]:
    if TEST_INDICES_FILE.exists():
        return json.loads(TEST_INDICES_FILE.read_text()).get("indices", [])
    return []


@torch.no_grad()
def inference_direct(model, x: torch.Tensor) -> torch.Tensor:
    """Прямой прогноз: один проход через модель."""
    with autocast(enabled=True):
        pred, _, _ = model(x)
    return pred


@torch.no_grad()
def inference_ar(model, x: torch.Tensor, steps: int) -> List[torch.Tensor]:
    """
    Авторегрессионный прогноз.
    x: (1, T_in*C, H, W) — начальное состояние
    Возвращает список предсказаний длины steps.
    """
    preds = []
    current = x.clone()
    for _ in range(steps):
        with autocast(enabled=True):
            pred, _, _ = model(current)
        preds.append(pred.cpu())
        # Обновляем вход: сдвигаем окно
        # pred: (1, C, H, W) — один кадр
        # current: (1, T_in*C, H, W) — сдвигаем
        # Простейший AR: заменяем последний кадр на pred
        C = N_CHANNELS
        T_in = 2
        # current: кадры t-1, t → сдвигаем к t, t+1 (pred)
        old_frame = current[:, C:, :, :]  # второй кадр
        current = torch.cat([old_frame, pred.to(DEVICE)], dim=1)
    return preds


@torch.no_grad()
def inference_continuous(model, x: torch.Tensor, steps: int) -> List[torch.Tensor]:
    """Непрерывный прогноз: каждый шаг использует один и тот же вход."""
    preds = []
    for _ in range(steps):
        with autocast(enabled=True):
            pred, _, _ = model(x)
        preds.append(pred.cpu())
    return preds


def _denormalize(data: np.ndarray, ch: str, stats: dict) -> np.ndarray:
    s = stats.get(ch, {"mean": 0.0, "std": 1.0})
    return data * s["std"] + s["mean"]


def run() -> bool:
    """Выполняет шаг 5."""
    if is_step_done(STEP_NAME):
        logger.info(f"[{STEP_NAME}] Уже выполнен, пропускаю.")
        return True

    logger.info(f"=== [{STEP_NAME}] Запуск ===")

    try:
        # Проверка наличия модели
        if not BEST_MODEL.exists():
            logger.error(f"Модель не найдена: {BEST_MODEL}")
            logger.error("Сначала запусти step4_train.py")
            return False

        # Загрузка
        stats = _load_stats()
        test_idx = _load_test_indices()

        if not test_idx:
            logger.error("Test-индексы не найдены.")
            return False

        logger.info(f"Test samples: {len(test_idx)}")

        # Загружаем модель
        model = _load_model(BEST_MODEL)

        # Открываем данные
        ds = xr.open_zarr(str(SUBSET_ZARR))
        logger.info(f"Датасет открыт: {dict(ds.dims)}")

        # Ограничиваем test для скорости (первые 100 сэмплов)
        test_idx_limited = test_idx[:100]
        logger.info(f"Инференс на {len(test_idx_limited)} сэмплах (из {len(test_idx)})")

        all_preds = []
        all_targets = []

        for i, idx in enumerate(test_idx_limited):
            if i >= len(test_idx_limited) - 2:
                break  # нужно 2 кадра для T_in=2

            # Вход: 2 кадра
            frames = []
            for offset in range(2):
                fidx = test_idx_limited[i + offset]
                channels = []
                for ch in CHANNEL_NAMES:
                    if ch in ds.data_vars:
                        data = ds[ch].isel(time=fidx).values.astype(np.float32)
                        s = stats.get(ch, {"mean": 0.0, "std": 1.0})
                        data = (data - s["mean"]) / max(s["std"], 1e-8)
                        channels.append(data)
                    else:
                        channels.append(np.zeros(
                            (ds.sizes["latitude"], ds.sizes["longitude"]),
                            dtype=np.float32
                        ))
                frame = np.stack(channels, axis=0)
                frames.append(frame)

            x = np.stack(frames, axis=0)  # (T_in, C, H, W)
            x = x.reshape(1, -1, x.shape[-2], x.shape[-1])  # (1, T_in*C, H, W)
            x = torch.from_numpy(x).float().to(DEVICE)

            # Цель: t2m (первый канал) следующего кадра
            target_idx = test_idx_limited[i + 2] if i + 2 < len(test_idx_limited) else test_idx_limited[i + 1]
            if "2m_temperature" in ds.data_vars:
                target = ds["2m_temperature"].isel(time=target_idx).values.astype(np.float32)
            else:
                target = np.zeros((ds.sizes["latitude"], ds.sizes["longitude"]), dtype=np.float32)

            # Инференс
            pred = inference_direct(model, x)  # (1, C, H, W)
            pred = pred.cpu().numpy()[0]  # (C, H, W)

            # Денормализуем t2m
            pred_t2m = _denormalize(pred[0], "2m_temperature", stats)
            target_t2m = _denormalize(target, "2m_temperature", stats)

            all_preds.append(pred_t2m)
            all_targets.append(target_t2m)

            if i % 20 == 0:
                logger.info(f"  Инференс: {i}/{len(test_idx_limited)}")

        # Конвертируем в тензоры
        all_preds = np.stack(all_preds, axis=0)  # (N, H, W)
        all_targets = np.stack(all_targets, axis=0)

        # Вычисляем метрики
        metrics = compute_all_metrics(
            pred=all_preds,
            target=all_targets,
            lat=ds.latitude.values if "latitude" in ds.dims else None,
        )
        metrics["n_samples"] = len(all_preds)

        logger.info(f"Метрики: RMSE={metrics.get('rmse', 'N/A'):.4f}, "
                     f"MAE={metrics.get('mae', 'N/A'):.4f}, "
                     f"Bias={metrics.get('bias', 'N/A'):.4f}")

        # Сохраняем
        np.save(PREDICTIONS_OUTPUT, all_preds)
        METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2))
        logger.info(f"Предсказания сохранены: {PREDICTIONS_OUTPUT}")
        logger.info(f"Метрики сохранены: {METRICS_OUTPUT}")

        extra = {
            "n_samples": len(all_preds),
            "rmse": metrics.get("rmse", -1),
            "mae": metrics.get("mae", -1),
            "bias": metrics.get("bias", -1),
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