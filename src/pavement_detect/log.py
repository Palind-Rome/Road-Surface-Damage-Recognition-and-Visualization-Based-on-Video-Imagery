# -*- coding: utf-8 -*-
"""
日志与检测结果记录模块。
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from pavement_detect.utils import abs_path, save_chinese_image

# ──────────────────────── ResultLogger ────────────────────────


class ResultLogger:
    """单帧检测结果的表格记录器。"""

    COLUMNS = ["识别结果", "位置", "面积", "时间"]

    def __init__(self) -> None:
        self.results_df = pd.DataFrame(columns=self.COLUMNS)

    def concat_results(
        self,
        result: str,
        location,
        confidence: str,
        time_str: str,
    ) -> pd.DataFrame:
        new_row = pd.DataFrame(
            {
                "识别结果": [result],
                "位置": [str(location)],
                "面积": [confidence],
                "时间": [time_str],
            }
        )
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        return self.results_df


# ──────────────────────── LogTable ────────────────────────


class LogTable:
    """
    跨帧的检测日志表，支持追加、保存 CSV、导出视频/图片。
    """

    COLUMNS = ["文件路径", "识别结果", "位置", "面积", "时间"]

    def __init__(self, csv_file_path: str | None = None) -> None:
        self.csv_file_path = csv_file_path
        self.saved_images: list[np.ndarray] = []
        self.saved_target_images: list[np.ndarray] = []
        self.saved_images_ini: list[np.ndarray] = []
        self.saved_results: list = []

        # 初始化 / 加载 CSV
        try:
            if csv_file_path and not os.path.exists(csv_file_path):
                os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
                pd.DataFrame(columns=self.COLUMNS).to_csv(
                    csv_file_path, index=False, header=True
                )
            self.data = pd.DataFrame(columns=self.COLUMNS)
        except Exception:
            self.data = pd.DataFrame(columns=self.COLUMNS)

    # ──── 帧管理 ────

    def add_frames(
        self, image: np.ndarray, det_info, img_ini: np.ndarray
    ) -> None:
        self.saved_images.append(image)
        self.saved_images_ini.append(img_ini)
        self.saved_results = det_info
        if det_info:
            self.saved_target_images.append(image)

    def clear_frames(self) -> None:
        self.saved_images.clear()
        self.saved_images_ini.clear()
        self.saved_results.clear()
        self.saved_target_images.clear()

    # ──── 导出 ────

    def save_frames_file(
        self, fps: float = 30, video_name: str | None = "save"
    ) -> str | bool:
        """将 saved_images 导出为图片（单帧）或 AVI 视频（多帧）。"""
        if not self.saved_images:
            return False

        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        if len(self.saved_images) == 1:
            file_name = abs_path(f"tempDir/pic_{now_time}.png")
            cv2.imwrite(file_name, self.saved_images[0])
            return file_name

        # 多帧 → 视频
        h, w, _ = self.saved_images[0].shape
        save_name = video_name or "camera"
        file_name = abs_path(f"tempDir/{save_name}.avi")
        out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h))
        for img in self.saved_images:
            out.write(img)
        out.release()
        return file_name

    # ──── 日志条目 ────

    def add_log_entry(
        self,
        file_path: str,
        recognition_result: str,
        position,
        confidence,
        time_spent: str,
    ) -> pd.DataFrame:
        new_entry = pd.DataFrame(
            [[str(file_path), recognition_result, str(position), confidence, time_spent]],
            columns=self.COLUMNS,
        )
        self.data = pd.concat([new_entry, self.data]).reset_index(drop=True)
        return self.data

    def clear_data(self) -> None:
        self.data = pd.DataFrame(columns=self.COLUMNS)

    def save_to_csv(self) -> None:
        if self.csv_file_path:
            self.data.to_csv(
                self.csv_file_path, index=False, encoding="utf-8", mode="a", header=False
            )

    def update_table(self, log_table_placeholder) -> None:
        """在 Streamlit 占位符中刷新显示（最新 500 条）。使用 dataframe 而不是 table 避免页面无限变长。"""
        display_data = self.data.head(500) if len(self.data) > 500 else self.data
        # dataframe 会自带滚动条，比原生的 table 在行数过多时排版好得多
        log_table_placeholder.dataframe(display_data, use_container_width=True, hide_index=True)
