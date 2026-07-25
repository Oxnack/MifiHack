"""
Step 0 — Подготовка окружения.
- Установка Python-зависимостей
- Проверка GPU и VRAM
- Создание структуры директорий
"""
import subprocess
import sys
from pathlib import Path

from config import ROOT, DATA_DIR, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, GPU
from utils.logger import get_logger, log_exception
from utils.gpu_check import check_gpu
from utils.checkpoints import is_step_done, mark_step_done, mark_step_failed

logger = get_logger(__name__, "step0_setup.log")

STEP_NAME = "step0_setup"

# Минимально необходимые пакеты
REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "xarray",
    "zarr",
    "gcsfs",
    "dask",
    "netCDF4",
    "tensorboard",
    "matplotlib",
    "pandas",
    "numpy",
    "scikit-learn",
    "tqdm",
    "timm",          # для Swin Transformer
]


def _create_directories() -> None:
    """Создаёт все необходимые директории."""
    dirs = [
        DATA_DIR / "raw",
        DATA_DIR / "zarr",
        DATA_DIR / "processed",
        MODELS_DIR,
        LOGS_DIR,
        CHECKPOINTS_DIR,
        OUTPUTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Директория: {d} (OK)")


def _check_packages() -> bool:
    """Проверяет наличие минимально необходимых пакетов."""
    logger.info("Проверка установленных пакетов...")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace("-", "_"))
            logger.info(f"  {pkg}: OK")
        except ImportError:
            logger.warning(f"  {pkg}: MISSING")
            missing.append(pkg)

    if missing:
        logger.warning(
            f"Отсутствуют пакеты: {missing}. "
            f"Установи их вручную:\n"
            f"  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            f"  pip install xarray zarr gcsfs dask netCDF4 tensorboard matplotlib pandas numpy scikit-learn tqdm timm"
        )
        return False
    return True


def run() -> bool:
    """
    Выполняет шаг 0.
    Returns: True если всё OK
    """
    if is_step_done(STEP_NAME):
        logger.info(f"[{STEP_NAME}] Уже выполнен, пропускаю.")
        return True

    logger.info(f"=== [{STEP_NAME}] Запуск ===")

    try:
        # 1. Создание директорий
        logger.info("Создание структуры директорий...")
        _create_directories()

        # 2. Проверка пакетов
        pkg_ok = _check_packages()
        if not pkg_ok:
            logger.warning("Часть пакетов отсутствует, но продолжаем (может быть установлено иначе).")

        # 3. Проверка GPU
        gpu_ok = check_gpu(
            required_vram_gb=GPU["required_vram_gb"],
            required_cuda=GPU["cuda_version"],
        )
        if not gpu_ok:
            logger.error("GPU-проверка не пройдена. Исправь проблемы и перезапусти.")
            mark_step_failed(STEP_NAME, "GPU check failed")
            return False

        # 4. Тест PyTorch + CUDA
        import torch
        x = torch.randn(2, 3, 64, 64, device="cuda")
        y = x * 2 + 1
        logger.info(f"PyTorch CUDA test: тензор {tuple(y.shape)} на {y.device} — OK")

        mark_step_done(STEP_NAME, {"gpu_ok": gpu_ok, "packages_ok": pkg_ok})
        logger.info(f"=== [{STEP_NAME}] Завершён успешно ===")
        return True

    except Exception as e:
        log_exception(logger, e, STEP_NAME)
        mark_step_failed(STEP_NAME, str(e))
        return False


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)