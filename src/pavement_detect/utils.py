# -*- coding: utf-8 -*-
"""
路径与文件工具 —— 替代原项目中对 QtFusion.path 等的依赖。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ──────────────────────── 路径工具 ────────────────────────

# 项目根目录（SITPproject/）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def abs_path(relative: str) -> str:
    """
    将相对路径转换为基于项目根目录的绝对路径。
    功能等同于原项目 ``QtFusion.path.abs_path``。
    """
    return str(PROJECT_ROOT / relative)


# ──────────────────────── 文件操作 ────────────────────────


def save_uploaded_file(uploaded_file) -> str | None:
    """
    将 Streamlit 上传的文件保存到 tempDir/ 目录，返回保存路径。
    """
    if uploaded_file is None:
        return None

    base_path = abs_path("tempDir")
    os.makedirs(base_path, exist_ok=True)

    file_path = os.path.join(base_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def save_chinese_image(file_path: str, image_array: np.ndarray) -> None:
    """
    保存图片（支持中文路径），使用 Pillow 绕开 OpenCV 中文路径问题。
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        image.save(file_path)
        print(f"成功保存图像到: {file_path}")
    except Exception as exc:
        print(f"保存图像失败: {exc}")


# ──────────────────────── UI 辅助 ────────────────────────


def concat_results(
    result: str, location: str, confidence: str, time_str: str
) -> pd.DataFrame:
    """返回一行检测结果的 DataFrame，用于 Streamlit 表格展示。"""
    return pd.DataFrame(
        {
            "识别结果": [result],
            "位置": [location],
            "面积": [confidence],
            "时间": [time_str],
        }
    )


def load_default_image() -> Image.Image:
    """加载默认占位图片。"""
    ini_image = abs_path("icon/ini-image.png")
    if os.path.exists(ini_image):
        return Image.open(ini_image)
    # 若占位图不存在，返回一个纯灰色图
    return Image.fromarray(np.full((360, 640, 3), 200, dtype=np.uint8))


def get_camera_names() -> list[str]:
    """
    枚举可用摄像头，返回名称列表。
    首项为 ``"摄像头检测关闭"``，其后为可用摄像头索引号字符串。
    """
    camera_names = ["摄像头检测关闭"]
    max_test = 10
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            if str(i) not in camera_names:
                camera_names.append(str(i))
            cap.release()
    return camera_names


def format_time(seconds: float) -> str:
    """将秒数格式化为 ``HH:MM:SS``。"""
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"
