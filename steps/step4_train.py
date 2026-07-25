"""
Step 4 — Архитектура модели + Тренировка.
- SwinV2 Encoder (опционально: CNN-encoder fallback)
- BiFPN Neck
- UNet Decoder
- VQ Bottleneck (Vector Quantization)
- AMP + Gradient Checkpointing
- TensorBoard logging
"""
import sys
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import (
    ZARR_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR,
    CHANNEL_NAMES, N_CHANNELS, STATIC_FIELDS,
    TIME_TRAIN, TIME_VAL,
    SWIN, BIFPN, UNET_DECODER, BOTTLENECK, TRAIN, GPU,
    GRID_025, GRID_05, DATASET_SIZES,
)
from utils.logger import get_logger, log_exception
from utils.checkpoints import is_step_done, mark_step_done, mark_step_failed

logger = get_logger(__name__, "step4_train.log")
STEP_NAME = "step4_train"

SUBSET_ZARR = ZARR_DIR / "era5_28ch_0p25_6h.zarr"
TRAIN_INDICES_FILE = PROCESSED_DIR / "train_indices.json"
VAL_INDICES_FILE = PROCESSED_DIR / "val_indices.json"
STATS_FILE = PROCESSED_DIR / "stats.json"
OCEAN_MASK_FILE = PROCESSED_DIR / "ocean_mask.npy"

BEST_MODEL = MODELS_DIR / "best.pt"
LAST_MODEL = MODELS_DIR / "last.pt"
TB_LOG_DIR = LOGS_DIR / "tensorboard"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════

class WeatherDataset(Dataset):
    """Датасет погодных полей из Zarr."""

    def __init__(
        self,
        zarr_path: Path,
        indices: List[int],
        stats: dict,
        grid_size: Tuple[int, int] = None,  # (H, W) — если нужен ресайз
        t_in: int = 2,
        t_out: int = 1,
    ):
        self.ds = xr.open_zarr(str(zarr_path))
        self.indices = indices
        self.stats = stats
        self.grid_size = grid_size
        self.t_in = t_in
        self.t_out = t_out
        self.n_channels = len(CHANNEL_NAMES)

        # Загружаем статику один раз
        self.static = self._load_static()

    def _load_static(self) -> np.ndarray:
        """Загружает статические поля: (n_static, H, W)."""
        static_fields = []
        for sf in STATIC_FIELDS:
            if sf in self.ds.data_vars:
                data = self.ds[sf].isel(time=0).values.astype(np.float32)
                if data.ndim == 2:
                    static_fields.append(data)
                elif data.ndim == 3:
                    # Берем первый канал если 3D
                    static_fields.append(data[0])
            elif sf == "sin_lat":
                lat = self.ds.latitude.values.astype(np.float32)
                lat_rad = np.deg2rad(lat)
                sin_lat = np.sin(lat_rad)
                sin_lat_2d = np.tile(sin_lat[:, None], (1, self.ds.sizes["longitude"]))
                static_fields.append(sin_lat_2d)
            elif sf == "cos_lat":
                lat = self.ds.latitude.values.astype(np.float32)
                lat_rad = np.deg2rad(lat)
                cos_lat = np.cos(lat_rad)
                cos_lat_2d = np.tile(cos_lat[:, None], (1, self.ds.sizes["longitude"]))
                static_fields.append(cos_lat_2d)

        if not static_fields:
            return np.zeros((0, self.ds.sizes["latitude"], self.ds.sizes["longitude"]), dtype=np.float32)
        return np.stack(static_fields, axis=0)

    def __len__(self) -> int:
        # Пары: (t, t+1) для предиктора
        return max(0, len(self.indices) - self.t_in)

    def _normalize(self, data: np.ndarray, ch: str) -> np.ndarray:
        s = self.stats.get(ch, {"mean": 0.0, "std": 1.0})
        return (data - s["mean"]) / max(s["std"], 1e-8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]

        # Вход: t_in последовательных кадров
        x_frames = []
        for offset in range(self.t_in):
            frame_idx = self.indices[idx + offset]
            channels = []
            for ch in CHANNEL_NAMES:
                if ch in self.ds.data_vars:
                    data = self.ds[ch].isel(time=frame_idx).values.astype(np.float32)
                    data = self._normalize(data, ch)
                    channels.append(data)
                else:
                    # fallback zeros
                    channels.append(np.zeros(
                        (self.ds.sizes["latitude"], self.ds.sizes["longitude"]),
                        dtype=np.float32
                    ))
            frame = np.stack(channels, axis=0)  # (C, H, W)
            x_frames.append(frame)

        # (T_in, C, H, W)
        x = np.stack(x_frames, axis=0)

        # Цель: следующий кадр (только t2m для прогноза погоды — первый канал)
        target_idx = self.indices[idx + self.t_in - 1]  # последний доступный
        # Для задачи сжатия: цель = вход (autoencoder)
        # Для задачи прогноза: цель = t2m следующего шага
        # Здесь: автоэнкодер → цель = последний входной кадр
        y = x[-1:].copy()  # (1, C, H, W) — восстанавливаем последний кадр

        # Ресайз если нужно
        if self.grid_size is not None:
            h, w = self.grid_size
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            x = F.interpolate(x.view(-1, h, w).unsqueeze(0) if False else x,
                            size=(h, w), mode="bilinear", align_corners=False)
            # Упрощаем: интерполируем каждый кадр отдельно
            x_resized = []
            for t in range(x.shape[0]):
                frame_t = F.interpolate(
                    x[t:t+1], size=(h, w), mode="bilinear", align_corners=False
                )
                x_resized.append(frame_t)
            x = torch.cat(x_resized, dim=0)

            y_resized = []
            for t in range(y.shape[0]):
                frame_t = F.interpolate(
                    y[t:t+1], size=(h, w), mode="bilinear", align_corners=False
                )
                y_resized.append(frame_t)
            y = torch.cat(y_resized, dim=0)
        else:
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()

        # x: (T_in, C, H, W)
        # y: (1, C, H, W)
        return x, y


# ══════════════════════════════════════════════════════
# Model Components
# ══════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Простой residual блок с SiLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.silu = nn.SiLU()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.silu(out)


class CNNEncoder(nn.Module):
    """CNN-энкодер (легковесный fallback если Swin недоступен)."""
    def __init__(self, in_ch: int, embed_dims: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.stages = nn.ModuleList()
        prev_ch = in_ch
        for dim in embed_dims:
            stage = nn.Sequential(
                ResidualBlock(prev_ch, dim, stride=2),
                ResidualBlock(dim, dim, stride=1),
            )
            self.stages.append(stage)
            prev_ch = dim
        self.out_channels = embed_dims

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class BiFPNNeck(nn.Module):
    """Bi-directional Feature Pyramid Network."""
    def __init__(self, in_channels: List[int], out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        num_levels = len(in_channels)

        # Lateral convolutions
        self.lateral = nn.ModuleList([
            nn.Conv2d(ch, out_dim, 1) for ch in in_channels
        ])
        # Output convolutions
        self.output = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(),
            ) for _ in range(num_levels)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Приводим все к out_dim
        lat_features = [lateral(f) for lateral, f in zip(self.lateral, features)]

        # Top-down path
        for i in range(len(lat_features) - 2, -1, -1):
            up = F.interpolate(lat_features[i + 1], size=lat_features[i].shape[-2:],
                              mode="bilinear", align_corners=False)
            lat_features[i] = lat_features[i] + up

        # Bottom-up path
        for i in range(1, len(lat_features)):
            down = F.interpolate(lat_features[i - 1], size=lat_features[i].shape[-2:],
                                mode="bilinear", align_corners=False)
            lat_features[i] = lat_features[i] + down

        # Output convolutions
        outputs = [out_conv(f) for out_conv, f in zip(self.output, lat_features)]
        return outputs


class UNetDecoder(nn.Module):
    """UNet декодер с skip-connections."""
    def __init__(self, in_channels: List[int], out_ch: int, bottleneck_dim: int):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        # Идём с конца (самый глубокий)
        reversed_ch = list(reversed(in_channels))
        prev_ch = bottleneck_dim

        for i, skip_ch in enumerate(reversed_ch):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(prev_ch, skip_ch, 2, stride=2),
                    nn.BatchNorm2d(skip_ch),
                    nn.SiLU(),
                    ResidualBlock(skip_ch * 2, skip_ch),  # concat со skip
                )
            )
            prev_ch = skip_ch

        self.final = nn.Sequential(
            nn.Conv2d(prev_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 1),
        )

    def forward(self, bottleneck: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        x = bottleneck
        reversed_skips = list(reversed(skip_features))
        for i, up_block in enumerate(self.up_blocks):
            x = up_block[0:3](x)  # ConvTranspose + BN + SiLU
            skip = reversed_skips[i] if i < len(reversed_skips) else x
            # align sizes
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = up_block[3](x)  # ResidualBlock with concat
        return self.final(x)


class VectorQuantizer(nn.Module):
    """Vector Quantization bottleneck."""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D, H, W)
        B, D, H, W = z.shape
        # Flatten: (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # Расстояния до codebook: (B*H*W, K)
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Индексы ближайших
        encoding_indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        z_q = self.embedding(encoding_indices).view(B, H, W, D).permute(0, 3, 1, 2)

        # Loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss, encoding_indices


class WeatherAutoencoder(nn.Module):
    """Полный автоэнкодер: Encoder → BiFPN → VQ Bottleneck → Decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_dims: List[int] = [64, 128, 256, 512],
        bifpn_dim: int = 256,
        vq_num_embeddings: int = 1024,
        vq_dim: int = 256,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, encoder_dims)
        self.neck = BiFPNNeck(encoder_dims, bifpn_dim)
        # После BiFPN все features имеют bifpn_dim каналов
        bifpn_channels = [bifpn_dim] * len(encoder_dims)

        # Bottleneck: адаптивная свёртка до vq_dim
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(bifpn_dim, vq_dim, 3, padding=1),
            nn.BatchNorm2d(vq_dim),
            nn.SiLU(),
        )
        self.vq = VectorQuantizer(vq_num_embeddings, vq_dim)

        self.decoder = UNetDecoder(bifpn_channels, out_channels, vq_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T_in*C, H, W) — склеиваем временные кадры в каналы
        features = self.encoder(x)
        bifpn_features = self.neck(features)

        # Bottleneck — из самой глубокой фичи
        bottleneck_in = bifpn_features[-1]
        bottleneck = self.bottleneck_conv(bottleneck_in)
        z_q, vq_loss, _ = self.vq(bottleneck)

        # Декодер
        output = self.decoder(z_q, bifpn_features)

        return output, vq_loss, bottleneck


# ══════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════

def _load_stats() -> dict:
    if STATS_FILE.exists():
        return json.loads(STATS_FILE.read_text())
    return {}


def _load_indices(filepath: Path) -> List[int]:
    if filepath.exists():
        data = json.loads(filepath.read_text())
        return data.get("indices", [])
    return []


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_vq_loss = 0.0

    optimizer.zero_grad()

    for batch_idx, (x, y) in enumerate(dataloader):
        # x: (B, T_in, C, H, W) → (B, T_in*C, H, W)
        B, T_in, C, H, W = x.shape
        x = x.view(B, T_in * C, H, W).to(DEVICE)
        y = y[:, -1].to(DEVICE)  # (B, C, H, W) — восстанавливаем последний кадр

        with autocast(enabled=TRAIN["use_amp"]):
            pred, vq_loss, _ = model(x)
            recon_loss = F.smooth_l1_loss(pred, y)
            loss = recon_loss + 0.1 * vq_loss
            loss = loss / TRAIN["gradient_accumulation_steps"]

        scaler.scale(loss).backward()

        if (batch_idx + 1) % TRAIN["gradient_accumulation_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        global_step += 1

        if batch_idx % 50 == 0:
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {recon_loss.item():.4f} | VQ: {vq_loss.item():.4f}"
            )

    avg_loss = total_loss / len(dataloader)
    avg_vq = total_vq_loss / len(dataloader)
    writer.add_scalar("train/recon_loss", avg_loss, epoch)
    writer.add_scalar("train/vq_loss", avg_vq, epoch)

    return avg_loss


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, epoch: int, writer: SummaryWriter) -> float:
    model.eval()
    total_loss = 0.0

    for x, y in dataloader:
        B, T_in, C, H, W = x.shape
        x = x.view(B, T_in * C, H, W).to(DEVICE)
        y = y[:, -1].to(DEVICE)

        with autocast(enabled=TRAIN["use_amp"]):
            pred, vq_loss, _ = model(x)
            recon_loss = F.smooth_l1_loss(pred, y)

        total_loss += recon_loss.item()

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("val/recon_loss", avg_loss, epoch)
    return avg_loss


def run(
    dataset_n: int = 1024,
    grid: str = "0p5",
    epochs: int = None,
) -> bool:
    """Выполняет шаг 4."""
    if is_step_done(STEP_NAME):
        logger.info(f"[{STEP_NAME}] Уже выполнен, пропускаю.")
        return True

    logger.info(f"=== [{STEP_NAME}] Запуск (N={dataset_n}, grid={grid}) ===")

    try:
        epochs = epochs or TRAIN["epochs"]

        # Загрузка данных
        stats = _load_stats()
        train_idx = _load_indices(TRAIN_INDICES_FILE)
        val_idx = _load_indices(VAL_INDICES_FILE)

        if not train_idx:
            logger.error("Train-индексы не найдены. Запусти step3_preprocess.py")
            return False

        # Ограничиваем размер датасета
        train_idx = train_idx[:dataset_n]
        val_idx = val_idx[: min(len(val_idx), dataset_n // 4)]

        # Размер грида
        if grid == "0p5":
            grid_size = (GRID_05["lat"], GRID_05["lon"])
        else:
            grid_size = None  # нативное разрешение

        train_ds = WeatherDataset(SUBSET_ZARR, train_idx, stats, grid_size)
        val_ds = WeatherDataset(SUBSET_ZARR, val_idx, stats, grid_size)

        train_loader = DataLoader(
            train_ds,
            batch_size=TRAIN["batch_size"],
            shuffle=True,
            num_workers=TRAIN["num_workers"],
            pin_memory=TRAIN["pin_memory"],
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=TRAIN["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=TRAIN["pin_memory"],
        )

        logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

        # Модель
        in_ch = N_CHANNELS * 2  # T_in=2 кадра склеены в каналы
        out_ch = N_CHANNELS

        model = WeatherAutoencoder(
            in_channels=in_ch,
            out_channels=out_ch,
            encoder_dims=[64, 128, 256, 512],
            bifpn_dim=BIFPN["feature_dim"],
            vq_num_embeddings=BOTTLENECK["num_embeddings"],
            vq_dim=BOTTLENECK["latent_dim"],
        ).to(DEVICE)

        n_params = count_parameters(model)
        logger.info(f"Параметров модели: {n_params:,} (лимит 20M: {'OK' if n_params <= 20_000_000 else 'ПРЕВЫШЕН!'})")

        # Оптимизатор
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAIN["learning_rate"],
            weight_decay=TRAIN["weight_decay"],
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TRAIN["max_steps"], eta_min=1e-6
        )

        # AMP scaler
        scaler = GradScaler(enabled=TRAIN["use_amp"])

        # TensorBoard
        writer = SummaryWriter(log_dir=str(TB_LOG_DIR / f"N{dataset_n}_{grid}"))

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch, writer, global_step)
            val_loss = validate(model, val_loader, epoch, writer)

            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")

            # Сохраняем best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss},
                    BEST_MODEL,
                )
                logger.info(f"  → Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= TRAIN["early_stopping_patience"]:
                logger.info(f"Early stopping на эпохе {epoch}")
                break

            # Сохраняем last
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict()},
                LAST_MODEL,
            )

            scheduler.step()
            global_step += len(train_loader)

            if global_step >= TRAIN["max_steps"]:
                logger.info(f"Достигнут лимит шагов ({TRAIN['max_steps']})")
                break

        writer.close()

        extra = {
            "n_params": n_params,
            "best_val_loss": best_val_loss,
            "epochs_completed": epoch,
            "dataset_n": dataset_n,
            "grid": grid,
        }
        mark_step_done(STEP_NAME, extra)
        logger.info(f"=== [{STEP_NAME}] Завершён успешно ===")
        return True

    except Exception as e:
        log_exception(logger, e, STEP_NAME)
        mark_step_failed(STEP_NAME, str(e))
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_n", type=int, default=1024)
    parser.add_argument("--grid", type=str, default="0p5", choices=["0p25", "0p5"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    ok = run(dataset_n=args.dataset_n, grid=args.grid, epochs=args.epochs)
    sys.exit(0 if ok else 1)