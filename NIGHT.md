# Ночные GPU-задачи (MifiHack)

Задачи, которые можно запустить на сервере с GPU на ночь — долгие вычисления, не требующие интерактивного участия.

---

## 1. Скачивание и препроцессинг данных

| # | Задача | Оценка времени | Команда/описание |
|---|--------|---------------|------------------|
| 1.1 | Скачать ERA5 Zarr (0.25°, 28 каналов, 2014–2021) из WeatherBench 2 | 2–8 ч (зависит от сети) | `gsutil -m cp -r gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr ./data/` |
| 1.2 | Извлечь 28 каналов + маски + статику → `era5_28ch_0p25_6h.zarr` | 1–3 ч (CPU-heavy) | Python-скрипт: открыть Zarr, выбрать каналы по фиксированному порядку, сохранить |
| 1.3 | Conservative remapping 0.25° → 0.5° (360×720 grid) → `era5_28ch_0p5_6h.zarr` | 1–2 ч (CPU) | `xesmf` + `xarray`, однократно |
| 1.4 | Вычислить mean/std по train-периоду (2014–2019) для нормализации | ~30 мин | Поблочное чтение Zarr, накопление статистик |
| 1.5 | Сгенерировать индексы train/val/test с сезонной балансировкой для N ∈ {128, 256, 512, 1024, 2048, 4096, 8192} | ~15 мин | Случайная стратифицированная выборка по месяцам |
| 1.6 | Создать SHA256-манифесты всех локальных датасетов | ~10 мин | `sha256sum` по чанкам Zarr |

**Итого этап данных:** ~4–12 часов (зависит от сети и дискового I/O).

---

## 2. Обучение автоэнкодера (Задача 1 — Compression)

Целевой диапазон сжатия: **32–64×**. Модель ≤ 20M параметров, ≤ 50k шагов.

### 2.1 Основные запуски (0.5° сетка, быстрее)

| # | Эксперимент | Dataset N | CR target | Шаги | GPU-часы (оценка) |
|---|-------------|-----------|-----------|------|-------------------|
| 2.1.1 | Baseline VAE с квантованным bottleneck | 1024 | 32× | 50k | 6–10 ч |
| 2.1.2 | Baseline VAE с квантованным bottleneck | 1024 | 64× | 50k | 6–10 ч |
| 2.1.3 | Baseline VAE с квантованным bottleneck | 4096 | 32× | 50k | 8–12 ч |
| 2.1.4 | Baseline VAE с квантованным bottleneck | 4096 | 64× | 50k | 8–12 ч |
| 2.1.5 | Baseline VAE с квантованным bottleneck | 8192 | 32× | 50k | 10–14 ч |
| 2.1.6 | Baseline VAE с квантованным bottleneck | 8192 | 64× | 50k | 10–14 ч |

### 2.2 Архитектурные варианты (0.5°)

| # | Эксперимент | Описание |
|---|-------------|----------|
| 2.2.1 | CNN-AE + FSQ (Finite Scalar Quantization) | Простая CNN-архитектура с FSQ bottleneck |
| 2.2.2 | ViT-AE + VQ (Vector Quantization) | Vision Transformer based autoencoder |
| 2.2.3 | Perceiver-IO VAE | Perceiver-based encoder/decoder как в концепции AIRI |
| 2.2.4 | Swin-UNet + quantization | Swin Transformer backbone |

### 2.3 0.25° сетка (patch-based training)

| # | Эксперимент | Описание | GPU-часы |
|---|-------------|----------|----------|
| 2.3.1 | Patch-based AE (256×256 patches) | Случайные патчи из полного 721×1440 | 12–24 ч |
| 2.3.2 | Full-tile inference validation | Проверка бесшовности на всём глобусе | ~2 ч (только инференс) |

**Итого обучение:** запустить параллельно 2–4 конфигурации на одной GPU (если VRAM позволяет) или последовательно.

---

## 3. Обучение latent-предиктора (Задача 2 — Downstream/Probe)

После обучения и заморозки энкодера+декодера:

| # | Задача | Параметры | GPU-часы |
|---|--------|-----------|----------|
| 3.1 | latent→latent прогноз на 6ч (все конфиги энкодеров) | ≤2M параметров, 1024 пары, 5000 шагов | 1–2 ч каждый |
| 3.2 | Masked reconstruction probe (W-MAE style) | Маскирование латентных токенов, восстановление | 2–4 ч |
| 3.3 | Абляция: persistence baseline через латенты | Сравнение с простой персистентностью | ~30 мин |

---

## 4. Полный inference + evaluation

| # | Задача | Описание | Время |
|---|--------|----------|-------|
| 4.1 | Прогнать все тестовые сэмплы (2021) через encoder→quantize→decoder | Exact roundtrip, запись bitstream | 2–4 ч |
| 4.2 | Вычислить все метрики: RMSE, NRMSE, PSNR, спектры | По каждому каналу и составные score | 1–2 ч |
| 4.3 | Bootstrap CI (2000 повторений, 7-дневные блоки) | Статистическая оценка non-inferiority | 1–3 ч (CPU, можно параллельно) |
| 4.4 | Метрики экстремальных осадков | Специальные пороговые метрики | ~1 ч |
| 4.5 | Построить графики: качество–данные, качество–битрейт | Визуализация всех кривых | ~30 мин |

---

## 5. Гиперпараметрические сетки (опционально, если есть время)

| # | Сетка | Значения | Запусков |
|---|-------|----------|----------|
| 5.1 | LR sweep | {1e-4, 3e-4, 5e-4, 1e-3} | 4 |
| 5.2 | Bottleneck dim sweep | {64, 128, 256, 512} | 4 |
| 5.3 | Количество VQ codebook векторов | {256, 512, 1024, 2048} | 4 |
| 5.4 | Commitment loss weight (β) | {0.1, 0.25, 0.5, 1.0} | 4 |

---

## 6. Рекомендуемый порядок запуска на одну ночь

```
Приоритет 1 (обязательно):
  ├── 1.1–1.6: Все этапы препроцессинга данных
  ├── 2.1.1 + 2.1.2: Baseline на 1024 кадрах (32× и 64×)
  └── 2.1.3 + 2.1.4: Baseline на 4096 кадрах (32× и 64×)

Приоритет 2 (если уложились по времени):
  ├── 2.3.1: Patch-based AE на 0.25°
  └── 3.1: Latent-предикторы для всех обученных энкодеров

Приоритет 3 (под утро):
  ├── 4.1–4.5: Полный evaluation всех чекпоинтов
  └── 5.x: Дополнительные hyperparameter sweeps (если осталась GPU)
```

---

## 7. Скрипт-оркестратор (псевдокод)

```bash
#!/bin/bash
# night_run.sh — запустить перед уходом

set -e

# Активировать окружение
conda activate mifihack || source venv/bin/activate

# Этап 1: Данные
python scripts/download_era5.py                    # 1.1
python scripts/extract_28ch.py                     # 1.2
python scripts/regrid_0p5.py                       # 1.3
python scripts/compute_stats.py                    # 1.4
python scripts/generate_splits.py --all-n          # 1.5
python scripts/create_manifests.py                 # 1.6

# Этап 2: Обучение (запустить в tmux/screen, мониторить wandb/tensorboard)
for N in 1024 4096 8192; do
    for CR in 32 64; do
        python train.py \
            --dataset_n $N \
            --compression_target $CR \
            --max_steps 50000 \
            --grid 0p5 \
            --output_dir ./checkpoints/N${N}_CR${CR} \
            --wandb_project mifihack &
    done
done
wait

# Этап 3: Probe
for ckpt in ./checkpoints/*/best.pt; do
    python train_probe.py --encoder_ckpt $ckpt --max_steps 5000 &
done
wait

# Этап 4: Evaluation
python evaluate.py --checkpoints_dir ./checkpoints --output ./results
python bootstrap_ci.py --results_dir ./results
python plot_curves.py --results_dir ./results

echo "All night tasks completed at $(date)"
```

---

## 8. Мониторинг

- **Weights & Biases / TensorBoard** — логировать loss, CR, NRMSE онлайн
- **tmux/screen** — держать сессию живой при обрыве SSH
- **nvidia-smi** — мониторинг GPU-памяти и утилизации: `watch -n 10 nvidia-smi`
- **Логи:** `python train.py ... 2>&1 | tee logs/$(date +%Y%m%d_%H%M%S).log`