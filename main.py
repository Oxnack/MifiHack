"""
MifiHack — Главный оркестратор ночного прогона.
Запускает все этапы последовательно с контролем чекпоинтов.

Использование:
    python main.py --all           # запустить всё
    python main.py --step step0    # запустить конкретный шаг
    python main.py --reset step0   # сбросить чекпоинт шага
    python main.py --status        # показать статус всех шагов
"""
import sys
import argparse
import time
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import LOGS_DIR
from utils.logger import get_logger, log_exception
from utils.checkpoints import list_checkpoints, reset_step, is_step_done

logger = get_logger("main", "full_run.log")

# Импортируем шаги (лениво, чтобы не крашиться при отсутствии зависимостей)
STEP_MODULES = {
    "step0": "steps.step0_setup",
    "step1": "steps.step1_download",
    "step2": "steps.step2_zarr",
    "step3": "steps.step3_preprocess",
    "step4": "steps.step4_train",
    "step5": "steps.step5_inference",
    "step6": "steps.step6_package",
}

STEP_NAMES = list(STEP_MODULES.keys())


def _import_step(name: str):
    """Импортирует модуль шага."""
    module_name = STEP_MODULES[name]
    import importlib
    return importlib.import_module(module_name)


def run_all() -> bool:
    """Запускает все шаги последовательно."""
    logger.info("=" * 60)
    logger.info("MifiHack — ПОЛНЫЙ НОЧНОЙ ПРОГОН")
    logger.info("=" * 60)

    start_time = time.time()
    results = {}

    for step_name in STEP_NAMES:
        logger.info(f"\n{'─' * 40}")
        logger.info(f"▸ Запуск: {step_name}")
        logger.info(f"{'─' * 40}")

        try:
            step_module = _import_step(step_name)
            ok = step_module.run()

            results[step_name] = "OK" if ok else "FAILED"

            if not ok:
                logger.error(f"❌ Шаг {step_name} завершился с ошибкой. "
                             f"Дальнейшие шаги остановлены.")
                break
            else:
                logger.info(f"✅ Шаг {step_name} выполнен успешно.")

        except ImportError as e:
            logger.error(f"❌ Не удалось импортировать {step_name}: {e}")
            results[step_name] = f"IMPORT_ERROR: {e}"
            break
        except Exception as e:
            log_exception(logger, e, f"Шаг {step_name}")
            results[step_name] = f"CRASH: {e}"
            break

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"ИТОГИ ПРОГОНА (за {hours}ч {minutes}м {seconds}с):")
    for name, status in results.items():
        emoji = "✅" if status == "OK" else "❌"
        logger.info(f"  {emoji} {name}: {status}")
    logger.info(f"{'=' * 60}")

    return all(v == "OK" for v in results.values())


def run_step(step_name: str) -> bool:
    """Запускает конкретный шаг."""
    if step_name not in STEP_MODULES:
        logger.error(f"Неизвестный шаг: {step_name}. Доступные: {STEP_NAMES}")
        return False

    logger.info(f"Запуск шага: {step_name}")
    step_module = _import_step(step_name)
    return step_module.run()


def show_status() -> None:
    """Показывает статус всех шагов."""
    status = list_checkpoints()
    if not status:
        logger.info("Нет выполненных шагов. Все этапы ожидают запуска.")
        return

    logger.info("Статус этапов:")
    for step_name in STEP_NAMES:
        s = status.get(step_name, "not_started")
        emoji = "✅" if s == "done" else ("❌" if s == "failed" else "⬜")
        logger.info(f"  {emoji} {step_name}: {s}")


def main():
    parser = argparse.ArgumentParser(
        description="MifiHack — оркестратор ночного прогона"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Запустить все шаги последовательно"
    )
    parser.add_argument(
        "--step", type=str, metavar="STEP",
        help=f"Запустить конкретный шаг (один из: {STEP_NAMES})"
    )
    parser.add_argument(
        "--reset", type=str, metavar="STEP",
        help=f"Сбросить чекпоинт шага (один из: {STEP_NAMES})"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Показать статус всех шагов"
    )
    parser.add_argument(
        "--dataset_n", type=int, default=1024,
        help="Размер обучающего набора для step4 (по умолчанию: 1024)"
    )
    parser.add_argument(
        "--grid", type=str, default="0p5", choices=["0p25", "0p5"],
        help="Разрешение сетки для step4 (по умолчанию: 0p5)"
    )

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.reset:
        reset_step(args.reset)
        logger.info(f"Чекпоинт {args.reset} сброшен.")
        return

    if args.all:
        ok = run_all()
        sys.exit(0 if ok else 1)

    if args.step:
        ok = run_step(args.step)
        sys.exit(0 if ok else 1)

    # Если ничего не указано — показать помощь
    parser.print_help()


if __name__ == "__main__":
    main()