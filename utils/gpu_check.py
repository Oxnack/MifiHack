"""MifiHack — проверка доступности GPU и VRAM."""
import sys
import torch
from utils.logger import get_logger

logger = get_logger(__name__)


def _format_bytes(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(b) < 1024.0:
            return f"{b:.1f} {unit}"
        b /= 1024.0
    return f"{b:.1f} PB"


def check_gpu(required_vram_gb: float = 14.0, required_cuda: str = "12.1") -> bool:
    """
    Проверяет наличие GPU c достаточным объёмом VRAM и версией CUDA.

    Args:
        required_vram_gb: минимальный объём VRAM в гигабайтах
        required_cuda: ожидаемая мажорная версия CUDA (например, "12.1")

    Returns:
        True если конфигурация подходит
    """
    logger.info("=== GPU CHECK START ===")

    # PyTorch version
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available (torch): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        logger.error("CUDA недоступна! Проверь драйверы и установку PyTorch.")
        logger.error(
            "Установи: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )
        return False

    cuda_version = torch.version.cuda
    logger.info(f"CUDA version: {cuda_version}")

    if cuda_version is None or not cuda_version.startswith("12"):
        logger.warning(
            f"Ожидалась CUDA {required_cuda}, обнаружена {cuda_version}. "
            f"Может работать, но не протестировано."
        )

    device_count = torch.cuda.device_count()
    logger.info(f"GPU devices found: {device_count}")

    if device_count == 0:
        logger.error("Не найдено ни одного GPU-устройства.")
        return False

    all_ok = True
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        total_vram = torch.cuda.get_device_properties(i).total_mem
        total_vram_gb = total_vram / (1024**3)
        free_vram = (
            total_vram - torch.cuda.memory_allocated(i) - torch.cuda.memory_reserved(i)
        )
        free_vram_gb = free_vram / (1024**3)

        logger.info(
            f"  GPU {i}: {name} | "
            f"Total VRAM: {total_vram_gb:.1f} GB | "
            f"Free VRAM: ~{free_vram_gb:.1f} GB"
        )

        if total_vram_gb < required_vram_gb:
            logger.error(
                f"  GPU {i}: VRAM ({total_vram_gb:.1f} GB) < требуемых "
                f"({required_vram_gb} GB). Модель может не поместиться!"
            )
            all_ok = False
        else:
            logger.info(f"  GPU {i}: VRAM OK ({total_vram_gb:.1f} GB >= {required_vram_gb} GB)")

    # Тестовый аллокация
    try:
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        logger.info("CUDA test allocation: OK")
    except RuntimeError as e:
        logger.error(f"CUDA test allocation FAILED: {e}")
        all_ok = False

    if all_ok:
        logger.info("=== GPU CHECK: ALL OK ===")
    else:
        logger.error("=== GPU CHECK: PROBLEMS FOUND ===")

    return all_ok


if __name__ == "__main__":
    ok = check_gpu()
    sys.exit(0 if ok else 1)