"""MifiHack — система чекпоинтов для отказоустойчивого пайплайна.
Каждый шаг пишет JSON-флаг в checkpoints/ после успешного завершения.
При повторном запуске шаг пропускается, если флаг существует.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import CHECKPOINTS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def _ckpt_path(step_name: str) -> Path:
    """Путь к файлу чекпоинта для шага."""
    return CHECKPOINTS_DIR / f"{step_name}.json"


def is_step_done(step_name: str) -> bool:
    """
    Проверяет, был ли шаг уже выполнен.

    Args:
        step_name: имя шага (например, 'step0_setup')

    Returns:
        True если чекпоинт существует и валиден
    """
    ckpt = _ckpt_path(step_name)
    if not ckpt.exists():
        return False
    try:
        data = json.loads(ckpt.read_text())
        return data.get("status") == "done"
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Чекпоинт {step_name} повреждён, будет перезаписан.")
        return False


def mark_step_done(step_name: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Отмечает шаг как успешно выполненный.

    Args:
        step_name: имя шага
        extra: дополнительные данные для сохранения (время выполнения, метрики, etc.)
    """
    data = {
        "step": step_name,
        "status": "done",
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        data["extra"] = extra

    ckpt = _ckpt_path(step_name)
    ckpt.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f"Чекпоинт сохранён: {step_name}")


def mark_step_failed(step_name: str, error: str) -> None:
    """
    Отмечает шаг как упавший (для отладки).

    Args:
        step_name: имя шага
        error: текст ошибки
    """
    data = {
        "step": step_name,
        "status": "failed",
        "error": error,
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ckpt_path(step_name).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def reset_step(step_name: str) -> None:
    """Удаляет чекпоинт шага (для принудительного перезапуска)."""
    ckpt = _ckpt_path(step_name)
    if ckpt.exists():
        ckpt.unlink()
        logger.info(f"Чекпоинт сброшен: {step_name}")


def list_checkpoints() -> Dict[str, str]:
    """Возвращает статус всех шагов."""
    status = {}
    for ckpt_file in sorted(CHECKPOINTS_DIR.glob("*.json")):
        step_name = ckpt_file.stem
        try:
            data = json.loads(ckpt_file.read_text())
            status[step_name] = data.get("status", "unknown")
        except Exception:
            status[step_name] = "corrupted"
    return status