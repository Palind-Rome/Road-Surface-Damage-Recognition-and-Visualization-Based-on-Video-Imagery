# -*- coding: utf-8 -*-
"""
YOLOv8 检测器封装 —— 替代原项目对 QtFusion.models.Detector 的继承。
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import torch

from pavement_detect.config import Chinese_name

# ──────────────── 延迟导入 ultralytics（项目本地副本） ────────────────
# ultralytics 放在项目根目录，运行时会被 sys.path 覆盖到
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DEFAULT_PARAMS: dict[str, Any] = {
    "device": DEVICE,
    "conf": 0.3,
    "iou": 0.05,
    "classes": None,
    "verbose": False,
}


def count_classes(det_info: list[dict], class_names: list[str]) -> list[int]:
    """统计各类别检出数量。"""
    count_dict = {name: 0 for name in class_names}
    for info in det_info:
        cname = info["class_name"]
        if cname in count_dict:
            count_dict[cname] += 1
    return [count_dict[n] for n in class_names]


class WebDetector:
    """
    YOLOv8 实例分割检测器。

    对外暴露与原 ``Web_Detector`` 相同的接口：
    ``load_model``, ``preprocess``, ``predict``, ``postprocess``, ``set_param``。
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.model: YOLO | None = None
        self.img: np.ndarray | None = None
        self.imgsz: int = 640
        self.names: list[str] = list(Chinese_name.values())
        self.params: dict[str, Any] = params if params else dict(DEFAULT_PARAMS)

    # ────────────────────────────────────────────────────

    def load_model(self, model_path: str) -> None:
        """加载 YOLO 权重文件并预热。"""
        self.device = select_device(self.params["device"])
        task = "segment"  # 本项目统一使用实例分割
        self.model = YOLO(model_path, task=task)

        names_dict: dict[int, str] = self.model.names  # type: ignore[assignment]
        self.names = [
            Chinese_name[v] if v in Chinese_name else v for v in names_dict.values()
        ]

        # 预热
        dummy = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device)
        dummy = dummy.type_as(next(self.model.model.parameters()))
        self.model(dummy)

    # ────────────────────────────────────────────────────

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        self.img = img
        return img

    def predict(self, img: np.ndarray):
        """运行推理，返回 YOLO Results 列表。"""
        return self.model(img, **self.params)

    def postprocess(self, pred) -> list[dict]:
        """
        将 YOLO 推理结果解析为统一的字典列表。

        每个字典包含:
        - ``class_name``  (str)
        - ``bbox``        (list[int])
        - ``score``       (float)
        - ``class_id``    (int)
        - ``mask``        (list | None)
        """
        results: list[dict] = []

        for res in pred[0].boxes:
            aim_id = 0
            for box in res:
                class_id = int(box.cls.cpu())
                bbox = [int(c) for c in box.xyxy.cpu().squeeze().tolist()]
                result = {
                    "class_name": self.names[class_id],
                    "bbox": bbox,
                    "score": float(box.conf.cpu().squeeze().item()),
                    "class_id": class_id,
                    "mask": (
                        pred[0].masks[aim_id].xy
                        if pred[0].masks is not None
                        else None
                    ),
                }
                results.append(result)
                aim_id += 1

        return results

    # ────────────────────────────────────────────────────

    def set_param(self, params: dict[str, Any]) -> None:
        self.params.update(params)
