"""
Step 6 — Упаковка результатов.
- Сбор submission.csv в нужном формате
- Сохранение логов всех этапов
- Создание итогового отчёта (JSON)
- Архивация результатов
"""
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

from config import (
    OUTPUTS_DIR, LOGS_DIR, MODELS_DIR, CHECKPOINTS_DIR,
    PROCESSED_DIR, ZARR_DIR,
    SUBMISSION_COLUMNS, SUBMISSION_FORMAT,
)
from utils.logger import get_logger, log_exception
from utils.checkpoints import is_step_done, mark_step_done, mark_step_failed, list_checkpoints

logger = get_logger(__name__, "step6_package.log")
STEP_NAME = "step6_package"

METRICS_FILE = OUTPUTS_DIR / "metrics.json"
PREDICTIONS_FILE = OUTPUTS_DIR / "predictions.npy"
SUBMISSION_CSV = OUTPUTS_DIR / "submission.csv"
FINAL_REPORT = OUTPUTS_DIR / "final_report.json"
ARCHIVE_DIR = OUTPUTS_DIR / "archive"


def _create_submission_csv(
    predictions: np.ndarray,
    lat: np.ndarray = None,
    lon: np.ndarray = None,
    forecast_dt: str = "2021-01-01T00:00:00",
) -> Path:
    """
    Создаёт submission.csv в формате:
    forecast_dt, latitude, longitude, t2m_pred
    """
    logger.info("Создание submission.csv ...")

    if predictions.ndim == 3:
        # (T, H, W) → берём последний доступный временной шаг
        pred_2d = predictions[-1]
    elif predictions.ndim == 2:
        pred_2d = predictions
    else:
        logger.error(f"Неожиданная форма predictions: {predictions.shape}")
        return None

    H, W = pred_2d.shape

    # Координаты
    if lat is None:
        lat = np.linspace(90, -90, H, dtype=np.float32)
    if lon is None:
        lon = np.linspace(0, 359.5, W, dtype=np.float32)

    # Создаём DataFrame
    rows = []
    for i in range(H):
        for j in range(W):
            rows.append({
                "forecast_dt": forecast_dt,
                "latitude": float(lat[i]),
                "longitude": float(lon[j]),
                "t2m_pred": float(pred_2d[i, j]),
            })

    df = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    df.to_csv(SUBMISSION_CSV, index=False)
    logger.info(f"Submission сохранён: {SUBMISSION_CSV} ({len(df)} строк)")

    return SUBMISSION_CSV


def _collect_logs() -> Dict[str, str]:
    """Собирает все логи в один словарь."""
    logs = {}
    for log_file in sorted(LOGS_DIR.glob("*.log")):
        try:
            content = log_file.read_text()
            # Берём только последние 100 строк для отчёта
            lines = content.strip().split("\n")
            logs[log_file.name] = "\n".join(lines[-100:])
        except Exception as e:
            logs[log_file.name] = f"[ERROR reading log: {e}]"
    return logs


def _create_final_report(
    metrics: dict,
    checkpoint_status: dict,
    logs_summary: dict,
) -> Dict[str, Any]:
    """Создаёт итоговый отчёт."""
    report = {
        "title": "MifiHack — Final Report",
        "generated_at": datetime.now().isoformat(),
        "steps_status": checkpoint_status,
        "metrics": metrics,
        "files": {
            "submission": str(SUBMISSION_CSV) if SUBMISSION_CSV.exists() else None,
            "predictions": str(PREDICTIONS_FILE) if PREDICTIONS_FILE.exists() else None,
            "best_model": str(MODELS_DIR / "best.pt"),
            "log_files": list(logs_summary.keys()),
        },
        "summary": {
            "all_steps_completed": all(
                v == "done" for v in checkpoint_status.values()
            ) if checkpoint_status else False,
            "rmse": metrics.get("rmse", "N/A"),
            "mae": metrics.get("mae", "N/A"),
            "n_predictions": metrics.get("n_samples", 0),
        },
    }
    FINAL_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"Финальный отчёт сохранён: {FINAL_REPORT}")
    return report


def _create_archive() -> Path:
    """Создаёт архив всех результатов."""
    import tarfile

    archive_name = f"mifihack_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    archive_path = ARCHIVE_DIR / archive_name
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "w:gz") as tar:
        # Добавляем outputs
        if OUTPUTS_DIR.exists():
            for f in OUTPUTS_DIR.glob("*"):
                if f != ARCHIVE_DIR:
                    tar.add(f, arcname=f"outputs/{f.name}")
        # Добавляем логи
        if LOGS_DIR.exists():
            tar.add(LOGS_DIR, arcname="logs")
        # Добавляем модель
        best_model = MODELS_DIR / "best.pt"
        if best_model.exists():
            tar.add(best_model, arcname="models/best.pt")
        # Добавляем чекпоинты
        if CHECKPOINTS_DIR.exists():
            tar.add(CHECKPOINTS_DIR, arcname="checkpoints")

    logger.info(f"Архив создан: {archive_path} ({archive_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return archive_path


def run() -> bool:
    """Выполняет шаг 6."""
    if is_step_done(STEP_NAME):
        logger.info(f"[{STEP_NAME}] Уже выполнен, пропускаю.")
        return True

    logger.info(f"=== [{STEP_NAME}] Запуск ===")

    try:
        # 1. Загружаем метрики
        metrics = {}
        if METRICS_FILE.exists():
            metrics = json.loads(METRICS_FILE.read_text())
            logger.info(f"Метрики загружены: RMSE={metrics.get('rmse', 'N/A')}")

        # 2. Создаём submission.csv
        predictions = None
        if PREDICTIONS_FILE.exists():
            predictions = np.load(PREDICTIONS_FILE)
            logger.info(f"Предсказания загружены: {predictions.shape}")

        if predictions is not None:
            _create_submission_csv(predictions)
        else:
            logger.warning("Предсказания не найдены, submission.csv НЕ создан.")
            # Создаём пустой submission как placeholder
            pd.DataFrame(columns=SUBMISSION_COLUMNS).to_csv(SUBMISSION_CSV, index=False)

        # 3. Статус чекпоинтов
        checkpoint_status = list_checkpoints()
        logger.info(f"Статус этапов: {checkpoint_status}")

        # 4. Сбор логов
        logs_summary = _collect_logs()
        logger.info(f"Собрано лог-файлов: {len(logs_summary)}")

        # 5. Финальный отчёт
        report = _create_final_report(metrics, checkpoint_status, logs_summary)

        # 6. Архивация
        archive_path = _create_archive()

        extra = {
            "submission_created": SUBMISSION_CSV.exists(),
            "archive_path": str(archive_path),
            "all_completed": report["summary"]["all_steps_completed"],
            "rmse": metrics.get("rmse", "N/A"),
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