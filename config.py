"""
MifiHack — единый конфигурационный файл.
Все пути, гиперпараметры и настройки в одном месте.
NVIDIA Tesla P100 16GB VRAM, CUDA 12.x.
"""
import os
from pathlib import Path

# ── Пути ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZARR_DIR = DATA_DIR / "zarr"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"

# Автосоздание директорий
for d in [RAW_DIR, ZARR_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Данные ────────────────────────────────────────────
# WeatherBench2 Zarr (анонимный доступ)
WB2_ERA5_ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

# 28-канальный набор (фиксированный порядок):
#   surface:   t2m, mslp, u10, v10, tp6h, sst, tcwv, tcc  (8 каналов)
#   pressure:  T, U, V, Z, Q × уровни [1000, 925, 850, 700] hPa  (20 каналов)
SURFACE_VARS = [
    "2m_temperature",           # t2m
    "mean_sea_level_pressure",  # mslp
    "10m_u_component_of_wind",  # u10
    "10m_v_component_of_wind",  # v10
    "total_precipitation_6hr",  # tp6h
    "sea_surface_temperature",  # sst
    "total_column_water_vapour",# tcwv
    "total_cloud_cover",        # tcc
]

PRESSURE_VARS = [
    "temperature",              # T
    "u_component_of_wind",      # U
    "v_component_of_wind",      # V
    "geopotential",             # Z
    "specific_humidity",        # Q
]

PRESSURE_LEVELS = [1000, 925, 850, 700]

# Порядок каналов для модели (28 + статика)
CHANNEL_NAMES = SURFACE_VARS + [
    f"{v}_{l}" for v in PRESSURE_VARS for l in PRESSURE_LEVELS
]
N_CHANNELS = len(CHANNEL_NAMES)  # 28

# Статические поля (подаются как conditioning, не восстанавливаются)
STATIC_FIELDS = ["land_sea_mask", "orography", "sin_lat", "cos_lat"]

# Временные диапазоны
TIME_TRAIN = ("2014", "2019")
TIME_VAL   = ("2020", "2020")
TIME_TEST  = ("2021", "2021")

# Размеры вложенных обучающих наборов
DATASET_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]

# ── Сетки ─────────────────────────────────────────────
# 0.25°  (721 × 1440)  — нативная
# 0.5°   (360 × 720)   — после conservative remapping
GRID_025 = {"lat": 721, "lon": 1440}
GRID_05  = {"lat": 360, "lon": 720}

# ── Предобработка ─────────────────────────────────────
NORMALIZATION = "standard"      # StandardScaler (mean/std)
USE_OCEAN_MASK = True           # SST только над океаном

# ── Модель (SwinV2 + BiFPN + UNet) ────────────────────
# Encoder: Swin Transformer v2
SWIN = {
    "patch_size": 2,
    "window_size": 8,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "mlp_ratio": 4.0,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
}

# Neck: BiFPN
BIFPN = {
    "num_levels": 4,
    "feature_dim": 256,
}

# Decoder: UNet
UNET_DECODER = {
    "num_stages": 4,
    "activation": "silu",
}

# Bottleneck quantization
BOTTLENECK = {
    "latent_dim": 256,           # размерность латента
    "num_embeddings": 1024,      # размер codebook для VQ
    "compression_target": 32,    # целевой CR (32× или 64×)
}

# ── Тренировка ────────────────────────────────────────
TRAIN = {
    "batch_size": 2,
    "gradient_accumulation_steps": 8,   # эффективный batch = 16
    "epochs": 50,
    "max_steps": 50_000,
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "lr_schedule": "cosine",
    "warmup_steps": 1000,
    "early_stopping_patience": 10,
    "max_grad_norm": 1.0,
    "use_amp": True,                     # Automatic Mixed Precision
    "use_gradient_checkpointing": True,  # экономия VRAM
    "loss": "smooth_l1",                 # SmoothL1Loss
    "optimizer": "adamw",
    "num_workers": 4,
    "pin_memory": True,
}

# ── Инференс ──────────────────────────────────────────
INFERENCE = {
    "forecast_horizon_hours": 120,   # 5 суток
    "time_step_hours": 1,
    "num_steps": 120,                # 120 шагов по 1 часу
    "batch_size": 1,
    "modes": ["direct", "ar", "continuous", "const_correction"],
}

# ── Probe / Latent Predictor (Задача 2) ────────────────
PROBE = {
    "max_params": 2_000_000,
    "training_pairs": 1024,
    "max_steps": 5000,
    "forecast_step": 6,  # часы
    "learning_rate": 1e-3,
}

# ── Метрики ───────────────────────────────────────────
METRICS_LIST = ["rmse", "mae", "bias", "acc", "nrmse"]
BOOTSTRAP = {
    "n_iterations": 2000,
    "block_days": 7,
    "confidence": 0.95,
}

# ── GPU ────────────────────────────────────────────────
GPU = {
    "required_vram_gb": 14,      # минимум
    "cuda_version": "12.1",
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu",
}

# ── Логи ──────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"

# ── Submission ─────────────────────────────────────────
SUBMISSION_FORMAT = "csv"
SUBMISSION_COLUMNS = ["forecast_dt", "latitude", "longitude", "t2m_pred"]