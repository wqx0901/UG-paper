"""
第八周消融实验数据模块。
   - A: 纯 CNN，直接使用原始序列；
   - B: 分解 + CNN，使用趋势项与周期项作为双通道输入。
"""

from __future__ import annotations

import os
import sys

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CURRENT_DIR)
_WEEK6_DIR = os.path.join(_ROOT_DIR, "week6")
_WEEK7_DIR = os.path.join(_ROOT_DIR, "week7")

for _path in (_WEEK6_DIR, _WEEK7_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from decomposition import moving_average_trend, seasonal_pattern_from_detrended
from transforms import reshape_sequences_to_2d


def decompose_window(
    window: np.ndarray,
    period: int = 24,
    trend_window: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对单个滑窗序列做趋势-周期分解。

    Args:
        window: 单个样本窗口，形状为 [T]。
        period: 周期长度，小时级负荷一般取 24。
        trend_window: 移动平均窗口。

    Returns:
        trend: 趋势项，形状为 [T]
        periodic: 周期项，形状为 [T]
    """
    window = np.asarray(window, dtype=np.float32)
    trend = moving_average_trend(window, window=trend_window).astype(np.float32)
    detrended = window - trend
    pattern = seasonal_pattern_from_detrended(detrended, period=period).astype(np.float32)
    periodic = pattern[np.arange(len(window)) % period].astype(np.float32)
    return trend, periodic


def make_decomposed_channels(
    x: np.ndarray,
    height: int,
    width: int,
    period: int = 24,
    trend_window: int = 25,
) -> np.ndarray:
    """
    将原始序列批量分解为双通道二维输入。

    输入：
        x: [Batch, T]

    输出：
        out: [Batch, 2, H, W]
        其中：
        - out[:, 0, :, :] 为趋势项；
        - out[:, 1, :, :] 为周期项。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.empty((0, 2, height, width), dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"x 必须是 [Batch, T]，当前形状为 {x.shape}")

    trend_list = []
    periodic_list = []

    for sample in x:
        trend, periodic = decompose_window(
            sample,
            period=period,
            trend_window=trend_window,
        )
        trend_list.append(trend)
        periodic_list.append(periodic)

    trend_array = np.stack(trend_list).astype(np.float32)
    periodic_array = np.stack(periodic_list).astype(np.float32)

    trend_2d = reshape_sequences_to_2d(trend_array, height=height, width=width)
    periodic_2d = reshape_sequences_to_2d(periodic_array, height=height, width=width)
    return np.concatenate([trend_2d, periodic_2d], axis=1)


def make_ablation_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    height: int,
    width: int,
    period: int = 24,
    trend_window: int = 25,
) -> dict[str, np.ndarray]:
    """
    同时构造消融实验所需的两类输入。

    Returns:
        一个字典，包含：
        - baseline_train_x: [N_train, 1, H, W]
        - baseline_test_x: [N_test, 1, H, W]
        - decomposed_train_x: [N_train, 2, H, W]
        - decomposed_test_x: [N_test, 2, H, W]
        - train_y: [N_train, H_out]
        - test_y: [N_test, H_out]
    """
    baseline_train_x = reshape_sequences_to_2d(x_train, height=height, width=width)
    baseline_test_x = reshape_sequences_to_2d(x_test, height=height, width=width)

    decomposed_train_x = make_decomposed_channels(
        x_train,
        height=height,
        width=width,
        period=period,
        trend_window=trend_window,
    )
    decomposed_test_x = make_decomposed_channels(
        x_test,
        height=height,
        width=width,
        period=period,
        trend_window=trend_window,
    )

    return {
        "baseline_train_x": baseline_train_x,
        "baseline_test_x": baseline_test_x,
        "decomposed_train_x": decomposed_train_x,
        "decomposed_test_x": decomposed_test_x,
        "train_y": np.asarray(y_train, dtype=np.float32),
        "test_y": np.asarray(y_test, dtype=np.float32),
    }
