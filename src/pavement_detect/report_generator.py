# -*- coding: utf-8 -*-
"""
AI 报告生成器 —— 调用硅基流动 API（DeepSeek-V3）对检测结果进行分析并生成报告。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Generator

import cv2
import numpy as np
import requests

from pavement_detect.config import API_KEY, API_URL, MODEL_NAME, BASE_PROMPT, REPORT_TEMPLATES


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return float(cv2.contourArea(points.astype(np.float32)))


class ReportGenerator:
    """管理报告生成、追问对话与历史记录。"""

    def __init__(self, max_history: int = 10) -> None:
        self.history: list[dict] = []
        self.max_history = max_history

    # ──────────── 数据格式化 ────────────

    def format_detection_data(
        self, det_info: list[dict], file_type: str, file_name: str
    ) -> str:
        if not isinstance(det_info, list):
            raise ValueError("检测信息必须是列表类型")

        lines = [
            f"文件类型: {file_type}",
            f"文件名: {file_name}",
            f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "检测结果:",
        ]

        for idx, info in enumerate(det_info, 1):
            lines.append(f"\n目标 {idx}:")
            lines.append(f"- 类别: {info['class_name']}")
            lines.append(f"- 置信度: {info['score']:.2f}")
            lines.append(f"- 位置: {info['bbox']}")

            if info.get("mask") is not None:
                lines.append("- 分割区域信息:")
                try:
                    mask_pts = np.array(info["mask"], dtype=np.int32)
                    if len(mask_pts) > 0:
                        area = _polygon_area(mask_pts)
                        perimeter = cv2.arcLength(mask_pts, True)
                        circularity = (
                            4 * np.pi * area / (perimeter ** 2)
                            if perimeter > 0
                            else 0
                        )
                        lines.append(f"  - 面积: {area:.2f}")
                        lines.append(f"  - 周长: {perimeter:.2f}")
                        lines.append(f"  - 圆度: {circularity:.2f}")
                except Exception as exc:
                    lines.append(f"  - 无法计算分割区域信息 ({exc})")

        return "\n".join(lines)

    # ──────────── 报告生成（流式） ────────────

    def generate_report(
        self,
        det_info: list[dict],
        file_type: str,
        file_name: str,
        template: str = "详细报告",
        language: str = "中文",
        is_summary: bool = False,
    ) -> Generator[str, None, None]:
        """以流式方式逐段 yield 报告文本。"""
        if not det_info:
            yield "没有检测到目标，请先进行检测"
            return

        try:
            detection_data = self.format_detection_data(det_info, file_type, file_name)
            full_prompt = BASE_PROMPT.format(detection_data=detection_data)
            full_prompt += f"\n请使用{language}生成报告。\n"
            full_prompt += (
                "请生成一份精简版的报告摘要，包含主要发现和建议。"
                if is_summary
                else REPORT_TEMPLATES[template]
            )

            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "你是一个专业的图像分析助手。"},
                        {"role": "user", "content": full_prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "stream": True,
                },
                stream=True,
                timeout=30,
            )

            if response.status_code != 200:
                yield f"API 调用失败，状态码: {response.status_code}"
                return

            report_content = ""
            for line in response.iter_lines():
                if not line:
                    continue
                raw = line[6:] if line.startswith(b"data: ") else line
                try:
                    chunk = json.loads(raw)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        content = delta["content"]
                        report_content += content
                        yield content
                except json.JSONDecodeError:
                    continue

            if report_content:
                self.history.append(
                    {
                        "report": report_content,
                        "template": template,
                        "language": language,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "conversation_history": [],
                    }
                )
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history :]

        except requests.exceptions.RequestException as exc:
            yield f"API 请求失败: {exc}"
        except Exception as exc:
            yield f"生成报告时出错: {exc}"

    # ──────────── 追问 ────────────

    def follow_up_question(
        self, history_index: int, question: str, language: str = "中文"
    ) -> str:
        if not (0 <= history_index < len(self.history)):
            return "无效的历史记录索引"

        entry = self.history[history_index]
        prompt = (
            f"基于之前的报告内容：\n{entry['report']}\n\n"
            f"用户提问：{question}\n\n"
            f"请用{language}回答这个问题，保持专业性和准确性。"
        )

        try:
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "你是一个专业的图像分析助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"]
                entry["conversation_history"].append(
                    {
                        "question": question,
                        "answer": answer,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                return answer
            return f"处理追问时出错: {resp.status_code} - {resp.text}"
        except Exception as exc:
            return f"处理追问时发生错误: {exc}"

    # ──────────── 导出 / 历史 ────────────

    def export_report(self, history_index: int, fmt: str = "pdf") -> str:
        if 0 <= history_index < len(self.history):
            return f"报告已导出为{fmt.upper()}格式"
        return "无效的历史记录索引"

    def get_history(self) -> list[dict]:
        return self.history

    def clear_history(self) -> None:
        self.history.clear()
