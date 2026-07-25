"""MifiHack — метрики качества: RMSE, MAE, Bias, ACC, NRMSE.
Все метрики — latitude-weighted (взвешенные по косинусу широты).
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


def _latitude_weights(
    lat: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """
    Веса по широте: cos(φ).

    Args:
        lat: массив широт в градусах [-90, 90]
        normalize: нормировать ли сумму весов на 1

    Returns:
        массив весов формы (H,)
    """
    lat_rad = np.deg2rad(lat)
    w = np.cos(lat_rad).astype(np.float32)
    w = np.clip(w, 0, None)  # убираем отрицательные (если есть за полюсами)
    if normalize:
        w = w / w.sum()
    return w


def _apply_weights(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Применяет веса к предсказаниям и цели."""
    if weights is not None:
        # weights: (H,) → (1, 1, H, 1) для (B, C, H, W)
        w = weights.to(pred.device).view(1, 1, -1, 1)
        pred = pred * w
        target = target * w
    return pred, target


def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
) -> float:
    """Latitude-weighted RMSE."""
    pred, target = _apply_weights(pred, target, lat_weights)
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
) -> float:
    """Latitude-weighted MAE."""
    pred, target = _apply_weights(pred, target, lat_weights)
    return torch.mean(torch.abs(pred - target)).item()


def bias(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
) -> float:
    """Latitude-weighted Bias (средняя ошибка)."""
    pred, target = _apply_weights(pred, target, lat_weights)
    return torch.mean(pred - target).item()


def acc(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_clim: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
) -> float:
    """
    Anomaly Correlation Coefficient (ACC).

    ACC = 1 - MSE(pred, target) / MSE(clim, target)

    Args:
        pred: предсказания
        target: истинные значения
        target_clim: климатология (среднее по train для target)
        lat_weights: веса широт

    Returns:
        ACC (float)
    """
    pred, target = _apply_weights(pred, target, lat_weights)
    # climatology тоже взвешиваем
    if lat_weights is not None:
        w = lat_weights.to(target_clim.device).view(1, 1, -1, 1)
        target_clim_w = target_clim * w
    else:
        target_clim_w = target_clim

    mse_model = torch.mean((pred - target) ** 2)
    mse_clim = torch.mean((target_clim_w - target) ** 2)

    if mse_clim == 0:
        return 0.0
    return (1.0 - mse_model / mse_clim).item()


def nrmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    std_train: float,
    lat_weights: Optional[torch.Tensor] = None,
) -> float:
    """
    Normalized RMSE: NRMSE = RMSE / σ_train

    Args:
        pred: предсказания
        target: истинные значения
        std_train: стандартное отклонение поля по обучающей выборке
        lat_weights: веса широт

    Returns:
        NRMSE (float)
    """
    r = rmse(pred, target, lat_weights)
    if std_train == 0:
        return 0.0
    return r / std_train


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_clim: Optional[torch.Tensor] = None,
    std_train: Optional[float] = None,
    lat: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Вычисляет все метрики за один проход.

    Args:
        pred: предсказания (B, C, H, W) numpy или torch
        target: истина (B, C, H, W)
        target_clim: климатология для ACC
        std_train: σ_train для NRMSE
        lat: массив широт для взвешивания

    Returns:
        словарь {metric_name: value}
    """
    # Конвертация numpy → torch
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    if isinstance(target_clim, np.ndarray):
        target_clim = torch.from_numpy(target_clim)

    pred = pred.float()
    target = target.float()

    lat_weights = None
    if lat is not None:
        w = _latitude_weights(lat, normalize=False)
        lat_weights = torch.from_numpy(w).float()

    results = {
        "rmse": rmse(pred, target, lat_weights),
        "mae": mae(pred, target, lat_weights),
        "bias": bias(pred, target, lat_weights),
    }

    if target_clim is not None:
        target_clim = target_clim.float()
        results["acc"] = acc(pred, target, target_clim, lat_weights)

    if std_train is not None:
        results["nrmse"] = nrmse(pred, target, std_train, lat_weights)

    return results


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
) -> float:
    """PSNR с заданным data_range."""
    mse_val = torch.mean((pred - target) ** 2)
    if mse_val == 0:
        return float("inf")
    return (20.0 * np.log10(data_range) - 10.0 * np.log10(mse_val.item()))