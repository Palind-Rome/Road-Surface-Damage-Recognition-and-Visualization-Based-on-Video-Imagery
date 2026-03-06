# -*- coding: utf-8 -*-
"""
绘图工具模块 —— 在图像上绘制检测结果（分割掩码、边界框、中文标签等）。
将之前分散在 web.py / demo_test_*.py 中的绘图函数统一到此处。
"""

from __future__ import annotations

from hashlib import md5

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────── 颜色生成 ────────────────────────


def generate_color_based_on_name(name: str) -> tuple[int, int, int]:
    """根据类别名称的哈希值生成稳定的 BGR 颜色。"""
    hex_color = md5(name.encode()).hexdigest()[:6]
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV BGR


# ──────────────────────── 中文文本绘制 ────────────────────────


def draw_with_chinese(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 20,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    在 OpenCV 图像上绘制中文文本（通过 Pillow）。
    ``color`` 为 RGB 格式。
    """
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("simsun.ttc", font_size, encoding="unic")
    except OSError:
        # 若系统没有 simsun，回退到默认字体
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


# ──────────────────────── 自适应参数 ────────────────────────


def adjust_parameter(image_size: tuple[int, ...], base_size: int = 1000) -> float:
    """根据图片最大尺寸计算自适应缩放因子。"""
    return max(image_size) / base_size


# ──────────────────────── 多边形面积 ────────────────────────


def calculate_polygon_area(points: np.ndarray) -> float:
    """计算多边形面积。输入为 Nx2 numpy 数组。"""
    if len(points) < 3:
        return 0.0
    return float(cv2.contourArea(points.astype(np.float32)))


# ──────────────────────── 核心：绘制检测结果 ────────────────────────


def draw_detections(
    image: np.ndarray,
    info: dict,
    alpha: float = 0.2,
) -> tuple[np.ndarray, float]:
    """
    在图像上绘制单个检测目标的可视化结果。

    Parameters
    ----------
    image : np.ndarray
        BGR 格式图像。
    info : dict
        必须包含 ``class_name``, ``bbox``, ``score``, ``class_id``, ``mask``。
    alpha : float
        掩码覆盖透明度。

    Returns
    -------
    (image, aim_frame_area)
        绘制后的图像 & 目标区域面积。
    """
    name = info["class_name"]
    bbox = info["bbox"]
    mask = info["mask"]

    adjust_param = adjust_parameter(image.shape[:2])
    spacing = int(20 * adjust_param)

    if mask is None:
        # ── 仅边界框 ──
        x1, y1, x2, y2 = bbox
        aim_frame_area = float((x2 - x1) * (y2 - y1))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), int(3 * adjust_param))
        image = draw_with_chinese(
            image, name, (x1, y1 - int(30 * adjust_param)), font_size=int(35 * adjust_param)
        )
    else:
        # ── 实例分割掩码 ──
        mask_points = np.concatenate(mask)
        aim_frame_area = calculate_polygon_area(mask_points)
        mask_color = generate_color_based_on_name(name)

        try:
            overlay = image.copy()
            cv2.fillPoly(overlay, [mask_points.astype(np.int32)], mask_color)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            cv2.drawContours(
                image, [mask_points.astype(np.int32)], -1, (0, 0, 255),
                thickness=int(8 * adjust_param),
            )

            # 几何指标
            area = cv2.contourArea(mask_points.astype(np.int32))
            perimeter = cv2.arcLength(mask_points.astype(np.int32), True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # 颜色采样
            binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(binary_mask, [mask_points.astype(np.int32)], -1, 255, -1)
            color_points = cv2.findNonZero(binary_mask)
            if color_points is not None and len(color_points) >= 5:
                selected = color_points[np.random.choice(len(color_points), 5, replace=False)]
                colors = np.mean([image[y, x] for x, y in selected[:, 0]], axis=0)
                color_str = f"({colors[0]:.1f}, {colors[1]:.1f}, {colors[2]:.1f})"
            else:
                color_str = "(N/A)"

            # 绘制类别名称
            x, y = np.min(mask_points, axis=0).astype(int)
            image = draw_with_chinese(
                image, name, (x, y - int(30 * adjust_param)), font_size=int(35 * adjust_param)
            )

        except Exception as exc:
            print(f"绘制检测结果时发生错误: {exc}")

    return image, aim_frame_area


def draw_rect_box(
    image: np.ndarray,
    bbox: list[int],
    alpha: float = 0.2,
    add_text: str = "",
    color: list[int] | tuple[int, ...] = (0, 255, 0),
) -> np.ndarray:
    """
    绘制半透明矩形框 + 标签文本。
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if add_text:
        image = draw_with_chinese(image, add_text, (x1, y1 - 25), font_size=20, color=(255, 255, 255))
    return image
