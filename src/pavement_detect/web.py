# -*- coding: utf-8 -*-
"""
Streamlit Web 检测主界面。
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pavement_detect.config import REPORT_TEMPLATES, Label_list
from pavement_detect.drawing import draw_detections, draw_rect_box
from pavement_detect.log import LogTable, ResultLogger
from pavement_detect.model import WebDetector
from pavement_detect.report_generator import ReportGenerator
from pavement_detect.ui_style import apply_all_styles, def_css_hitml
from pavement_detect.utils import (
    abs_path,
    concat_results,
    format_time,
    get_camera_names,
    load_default_image,
    save_chinese_image,
    save_uploaded_file,
)


class DetectionUI:
    def __init__(self) -> None:
        self.cls_name = Label_list
        self.colors = [
            [random.randint(0, 255) for _ in range(3)] for _ in range(len(self.cls_name))
        ]
        self.title = "基于视频图像的路面病害识别与展示"

        st.set_page_config(
            page_title=self.title,
            page_icon="REC",
            initial_sidebar_state="expanded",
            layout="wide",
        )
        st.markdown(
            f'<h1 style="text-align: center;">{self.title}</h1>', unsafe_allow_html=True
        )
        def_css_hitml()

        # 模型及阈值
        self.conf_threshold: float = 0.15
        self.iou_threshold: float = 0.5
        self.model_type: str = "默认任务"

        # 输入源
        self.selected_camera: str = "摄像头检测关闭"
        self.file_type: str = "图片文件"
        self.uploaded_file = None
        self.uploaded_video = None
        self.custom_model_file: str | None = None

        # 状态控制
        self.current_det_info: list[dict] = []
        self.FPS = 30
        self.display_mode: str = "叠加显示"
        self.close_flag = False

        # Session State 初始化
        self.saved_log_data = abs_path("tempDir/log_table_data.csv")
        if "logTable" not in st.session_state:
            st.session_state["logTable"] = LogTable(self.saved_log_data)
        self.logTable: LogTable = st.session_state["logTable"]

        if "available_cameras" not in st.session_state:
            st.session_state["available_cameras"] = get_camera_names()
        self.available_cameras = st.session_state["available_cameras"]

        if "model" not in st.session_state:
            st.session_state["model"] = WebDetector()
            st.session_state["loaded_model_path"] = ""
        self.model: WebDetector = st.session_state["model"]

        if "report_generator" not in st.session_state:
            st.session_state["report_generator"] = ReportGenerator()
        self.report_generator: ReportGenerator = st.session_state["report_generator"]

    # ──────────────────────────────────────────────

    def setup_main_window(self) -> None:
        self.setup_sidebar()

        st.write("--------")
        st.write("同济大学SITP项目：基于视频图像的路面病害识别与展示")
        st.write("--------")

        col1, col2, col3 = st.columns([4, 1, 3])

        # 左侧 - 画面区
        with col1:
            self.display_mode = st.radio(
                "显示模式",
                ["叠加显示", "对比显示"],
                horizontal=True,
                key="radio_display_mode",
            )
            if "current_image" not in st.session_state:
                st.session_state["current_image"] = load_default_image()

            self.image_placeholder = st.empty()
            self.image_placeholder_res = st.empty() if self.display_mode == "对比显示" else None
            self.progress_bar = st.progress(0)

            # 初始占位显示
            if not self.logTable.saved_images_ini:
                curr_img = st.session_state["current_image"]
                self.image_placeholder.image(curr_img, caption="原始画面")
                if self.image_placeholder_res is not None:
                    self.image_placeholder_res.image(curr_img, caption="识别画面")

        # 右侧 - 结果表格
        with col3:
            self.table_placeholder = st.empty()
            # 填入初值（此处也改为 dataframe 展示体验更好）
            res = concat_results("None", "[0, 0, 0, 0]", "0.00", "0.00s")
            self.table_placeholder.dataframe(res, use_container_width=True, hide_index=True)

            st.write("---")
            if st.button("导出近期结果为视频/图片", use_container_width=True):
                self.logTable.save_to_csv()
                req_name = self.uploaded_video.name if self.uploaded_video else None
                saved_path = self.logTable.save_frames_file(fps=self.FPS, video_name=req_name)
                if saved_path:
                    st.success(f"已导出文件：{saved_path}")
                st.success(f"CSV 数据已保存：{self.saved_log_data}")
                self.logTable.clear_data()

            self.log_table_placeholder = st.empty()
            self.logTable.update_table(self.log_table_placeholder)

        # 中间 - 控制按钮
        with col2:
            st.write("")
            st.write("")
            self.close_placeholder = st.empty()
            run_btn = st.button("▶ 开始检测", use_container_width=True)
            if run_btn:
                self.process_camera_or_file()

        # AI 报告生成区
        st.write("--------")
        st.subheader("🤖 AI 数据分析与报告生成")
        self.setup_report_section()

    def setup_sidebar(self) -> None:
        st.sidebar.header("参数配置")
        self.conf_threshold = st.sidebar.slider("置信度", 0.0, 1.0, 0.25)
        self.iou_threshold = st.sidebar.slider("IOU设定", 0.0, 1.0, 0.45)

        st.sidebar.header("模型选择")
        opt = st.sidebar.radio("模型来源", ["默认提供权重", "自定义权重(.pt)"])
        if opt == "自定义权重(.pt)":
            m_file = st.sidebar.file_uploader("上传 YOLOv8 .pt 文件", type=["pt"])
            if m_file:
                path = save_uploaded_file(m_file)
                if path and st.session_state.get("loaded_model_path") != path:
                    self.model.load_model(path)
                    st.session_state["loaded_model_path"] = path
                    # 重新分配颜色
                    self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.model.names))]
                    st.sidebar.success("自定义模型加载成功！")
        else:
            seg_path = abs_path("weights/yolov8s-seg.pt")
            if os.path.exists(seg_path) and st.session_state.get("loaded_model_path") != seg_path:
                try:
                    self.model.load_model(seg_path)
                    st.session_state["loaded_model_path"] = seg_path
                    self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.model.names))]
                except Exception as exc:
                    st.sidebar.error(f"无法加载默认模型: {exc}")

        st.sidebar.header("输入源设置")
        # Camera
        self.selected_camera = st.sidebar.selectbox("实时摄像头", self.available_cameras)
        # Media
        self.file_type = st.sidebar.selectbox("本地文件", ["图片文件", "视频文件"])
        if self.file_type == "图片文件":
            self.uploaded_file = st.sidebar.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
        else:
            self.uploaded_video = st.sidebar.file_uploader("上传视频", type=["mp4", "avi"])

    # ──────────────────────────────────────────────

    def _update_ui_frames(self, image: np.ndarray, frame_copy: np.ndarray) -> None:
        h, w = image.shape[:2]
        new_w = 800
        new_h = int(new_w * (h / w))
        im_res = cv2.resize(image, (new_w, new_h))
        im_raw = cv2.resize(frame_copy, (new_w, new_h))

        if self.display_mode == "叠加显示":
            self.image_placeholder.image(im_res, channels="BGR", caption="识别结果")
        else:
            self.image_placeholder.image(im_raw, channels="BGR", caption="原始画面")
            if self.image_placeholder_res:
                self.image_placeholder_res.image(im_res, channels="BGR", caption="识别结果")

    def process_camera_or_file(self) -> None:
        self.logTable.clear_frames()

        # 1. 摄像头
        if self.selected_camera != "摄像头检测关闭":
            self.close_flag = self.close_placeholder.button("⏹ 停止捕获")
            cam_idx = 0 if self.selected_camera == "0" else self.selected_camera
            try:
                cap = cv2.VideoCapture(int(cam_idx))
            except ValueError:
                cap = cv2.VideoCapture(cam_idx)

            self.FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_cnt = 0
            while cap.isOpened() and not self.close_flag:
                ret, frame = cap.read()
                if not ret:
                    break
                fc = frame.copy()
                image, det_info, _ = self.frame_process(frame, "camera")
                self._update_ui_frames(image, fc)
                self.logTable.add_frames(image, det_info, cv2.resize(frame, (640, 640)))

                frame_cnt += 1
                self.progress_bar.progress((frame_cnt % 100) / 100.0)

            cap.release()
            self._finalize_processing()
            return

        # 2. 图片
        if self.uploaded_file is not None and self.file_type == "图片文件":
            file_bytes = np.frombuffer(self.uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            fc = img.copy()

            image, det_info, _ = self.frame_process(img, self.uploaded_file.name)
            self._update_ui_frames(image, fc)
            self.logTable.add_frames(image, det_info, cv2.resize(img, (640, 640)))
            self.progress_bar.progress(1.0)
            self._finalize_processing()
            return

        # 3. 视频
        if self.uploaded_video is not None and self.file_type == "视频文件":
            self.close_flag = self.close_placeholder.button("⏹ 停止处理")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(self.uploaded_video.read())
                tpath = tfile.name

            cap = cv2.VideoCapture(tpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
            idx = 0

            while cap.isOpened() and not self.close_flag:
                ret, frame = cap.read()
                if not ret:
                    break
                fc = frame.copy()
                curr_t = format_time(idx / self.FPS)
                image, det_info, _ = self.frame_process(
                    frame, self.uploaded_video.name, video_time=curr_t
                )
                self._update_ui_frames(image, fc)
                self.logTable.add_frames(image, det_info, cv2.resize(frame, (640, 640)))

                idx += 1
                if total > 0:
                    self.progress_bar.progress(min(idx / total, 1.0))

            cap.release()
            self._finalize_processing()
            return

        st.warning("⚠️ 请从侧边栏选择监控摄像头或上传有效文件！")

    def _finalize_processing(self) -> None:
        self.logTable.save_to_csv()
        self.logTable.update_table(self.log_table_placeholder)

    # ──────────────────────────────────────────────

    def frame_process(self, image: np.ndarray, fname: str, video_time: str | None = None):
        """单帧前处理 -> 推理 -> 后处理 -> 绘图 -> 塞入UI表格。"""
        pre_img = self.model.preprocess(image)
        self.model.set_param({"conf": self.conf_threshold, "iou": self.iou_threshold})

        start = time.time()
        preds = self.model.predict(pre_img)
        t_used = round(time.time() - start, 3)

        det = preds[0]
        det_info_for_log = []
        sel_info = ["全部目标"]
        gui_res = None

        if det is not None and len(det):
            parsed_info = self.model.postprocess(preds)
            if parsed_info:
                rlog = ResultLogger()
                for i, info in enumerate(parsed_info):
                    name, bbox = info["class_name"], info["bbox"]
                    # 画图
                    image, area = draw_detections(image, info, alpha=0.5)

                    time_val = video_time if video_time else f"{t_used}s"
                    area_str = str(int(area))

                    # Logger
                    gui_res = rlog.concat_results(name, bbox, area_str, time_val)
                    self.logTable.add_log_entry(fname, name, bbox, area_str, time_val)
                    det_info_for_log.append([name, bbox, area_str, time_val, info["class_id"]])
                    sel_info.append(f"{name}-{i}")

                if gui_res is not None:
                    # 使用 dataframe 解决长距离卡顿
                    self.table_placeholder.dataframe(gui_res, use_container_width=True, hide_index=True)

        # 缓存给 AI 报告用
        if det is not None and len(det):
            self.current_det_info = self.model.postprocess(preds)
        else:
            self.current_det_info = []
        st.session_state["current_det_info"] = self.current_det_info

        return image, det_info_for_log, sel_info

    # ──────────────────────────────────────────────

    def setup_report_section(self) -> None:
        apply_all_styles()
        c1, c2 = st.columns([3, 1])

        # 右侧：设置与生成按钮
        with c2:
            st.markdown("##### ⚙️ 报告排版设置")
            tpl = st.selectbox("报告模板", list(REPORT_TEMPLATES.keys()))
            lang = st.selectbox("语言", ["中文", "English"])

            if st.button("📝 一键生成完整报告", use_container_width=True):
                info = st.session_state.get("current_det_info", [])
                if not info:
                    st.warning("⚠️ 暂无最新检测数据，请先执行“开始检测”。")
                else:
                    self._run_report_gen(info, tpl, lang, False, c1)

            if st.button("📋 生成简要摘要", use_container_width=True):
                info = st.session_state.get("current_det_info", [])
                if not info:
                    st.warning("⚠️ 暂无最新检测数据。")
                else:
                    self._run_report_gen(info, tpl, lang, True, c1)

            st.write("---")
            if st.button("🗑️ 清空历史", use_container_width=True):
                self.report_generator.clear_history()
                st.success("历史记录已清空")

        # 左侧：报表展示区
        with c1:
            st.markdown("##### 📄 当前视图")
            if not hasattr(self, "report_placeholder"):
                self.report_placeholder = st.empty()

            hist = self.report_generator.get_history()
            if hist:
                st.markdown("##### 📜 历史报告记录")
                for i, entry in enumerate(reversed(hist)):
                    with st.expander(f"📔 {entry['template']} - {entry['timestamp']}"):
                        st.markdown(
                            f"""
                            <div class="report-card">
                                <div class="report-content">{entry['report']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # 对话追问
                        st.write("**💬 向 AI 追问**")
                        q = st.text_input("输入问题", key=f"q_{i}")
                        if st.button("发送", key=f"btn_{i}"):
                            with st.spinner("AI 思考中..."):
                                ans = self.report_generator.follow_up_question(
                                    len(hist) - 1 - i, q, entry["language"]
                                )
                                st.info(ans)

                        # 显示对话历史
                        if entry.get("conversation_history"):
                            for qa in entry["conversation_history"]:
                                st.caption(f"**问**: {qa['question']}")
                                st.write(f"**答**: {qa['answer']}")
                                st.divider()

    def _run_report_gen(self, info, tpl, lang, is_summ, col_container) -> None:
        source_name = getattr(self.uploaded_file or self.uploaded_video, "name", "摄像头实时")
        source_type = self.file_type if self.selected_camera == "摄像头检测关闭" else "摄像头"
        
        gen = self.report_generator.generate_report(
            info, source_type, source_name, tpl, lang, is_summ
        )
        
        with col_container:
            curr_text = ""
            ph = st.empty()
            with st.spinner("🚀 调用 DeepSeek-V3 中..."):
                for chunk in gen:
                    curr_text += chunk
                    ph.markdown(
                        f"""
                        <div class="report-card">
                            <div class="report-header">
                                <span class="report-title">智能报告生成中</span>
                            </div>
                            <div class="report-content">{curr_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ──────────────────────────────────────────────

def main() -> None:
    app = DetectionUI()
    app.setup_main_window()

if __name__ == "__main__":
    main()
