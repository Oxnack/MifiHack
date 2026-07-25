"""
MifiHack — отказоустойчивый логгер.
Пишет одновременно в файл и stdout. Ротация по размеру.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config import LOGS_DIR, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Создаёт или возвращает существующий логгер.

    Args:
        name: имя логгера (обычно __name__)
        log_file: имя файла лога (если None — только stdout)

    Returns:
        настроенный logging.Logger
    """
    logger = logging.getLogger(name)

    # Не дублируем handler'ы
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Форматтер
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Файловый handler (опционально)
    if log_file:
        log_path = Path(LOGS_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_exception(logger: logging.Logger, e: Exception, context: str = "") -> None:
    """
    Логирует исключение с полным traceback.

    Args:
        logger: экземпляр логгера
        e: исключение
        context: описание контекста ошибки
    """
    msg = f"[FAIL] {context}: {type(e).__name__}: {e}" if context else f"[FAIL] {type(e).__name__}: {e}"
    logger.error(msg, exc_info=True)