# -*- coding: utf-8 -*-
"""
Streamlit 自定义 CSS 样式模块。
"""

from __future__ import annotations

import streamlit as st


def def_css_hitml() -> None:
    """注入全局 CSS 样式（表格、按钮、侧边栏等）。"""
    st.markdown(
        """
        <style>
        /* 全局样式：避免使用容易引起 Streamlit 原生组件(如 Radio 按钮内芯)冲突的旧版固定类名 */
        body {
            font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
        }

        /* 按钮样式 */
        .stButton > button {
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 2px 1px;
            cursor: pointer;
            border-radius: 8px;
            background-color: #9896f1;
            box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);
            transition-duration: 0.4s;
        }
        .stButton > button:hover {
            background-color: #5499C7;
            color: white;
            box-shadow: 0 8px 12px 0 rgba(0,0,0,0.24);
        }

        /* 侧边栏样式 */
        [data-testid="stSidebar"] {
            background-color: #f7f9fa;
        }
        
        /* 修复 Streamlit 原生 Radio 的选中态（通过强调内芯主色调解决） */
        div[role="radiogroup"] label[data-selected="true"] div[data-testid="stMarkdownContainer"] p {
            font-weight: bold;
            color: #2E86C1;
        }

        /* 滑块样式 */
        .stSlider .thumb { background-color: #2E86C1; }
        .stSlider .track { background-color: #DDD; }

        /* 表格容器添加滚动条 */
        .table-container {
            max-height: 400px;
            overflow-y: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* 修复 Streamlit 原生 dataframe 滚动 */
        div[data-testid="stDataFrame"] {
            max-height: 500px;
            overflow-y: auto;
        }

        /* 隐藏掉旧有的 table {} 中的过多阴影防止重叠生硬 */
        table {
            border-collapse: collapse;
            margin: 5px 0;
            font-size: 14px;
            min-width: 400px;
            width: 100%;
        }
        thead tr {
            background-color: #3498db !important;
            color: #ffffff;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        th, td { 
            padding: 10px 12px; 
            border: 1px solid #e0e0e0;
        }
        tbody tr { border-bottom: 1px solid #ddd; }
        tbody tr:nth-of-type(even) { background-color: #f8f9fa; }
        tbody tr:last-of-type { border-bottom: 2px solid #5499C7; }
        tbody tr:hover { background-color: #e8f4f8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def report_card_style() -> None:
    """报告卡片 CSS。"""
    st.markdown(
        """
    <style>
    .report-card {
        background: white; border-radius: 10px; padding: 20px;
        margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0; transition: all 0.3s ease;
    }
    .report-card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15); transform: translateY(-2px);
    }
    .report-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;
    }
    .report-title { font-size: 1.2em; font-weight: bold; color: #2c3e50; }
    .report-timestamp { color: #7f8c8d; font-size: 0.9em; }
    .report-content { line-height: 1.6; color: #34495e; }
    .loading-animation {
        display: inline-block; width: 20px; height: 20px;
        border: 3px solid rgba(0,0,0,0.1); border-radius: 50%;
        border-top-color: #3498db; animation: spin 1s ease-in-out infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    </style>
    """,
        unsafe_allow_html=True,
    )


def button_style() -> None:
    """增强按钮 CSS。"""
    st.markdown(
        """
    <style>
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white; border: none; padding: 8px 16px;
        border-radius: 5px; font-weight: bold; transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #2980b9, #3498db);
        transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton>button:active { transform: translateY(0); }
    </style>
    """,
        unsafe_allow_html=True,
    )


def input_style() -> None:
    """输入框 & 选择框 CSS。"""
    st.markdown(
        """
    <style>
    .stTextInput>div>div>input {
        border-radius: 5px; border: 1px solid #e0e0e0;
        padding: 8px 12px; transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #3498db; box-shadow: 0 0 0 2px rgba(52,152,219,0.2);
    }
    .stSelectbox>div>div>div {
        border-radius: 5px; border: 1px solid #e0e0e0; transition: all 0.3s ease;
    }
    .stSelectbox>div>div>div:hover { border-color: #3498db; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def apply_all_styles() -> None:
    """一次调用注入所有增强样式。"""
    report_card_style()
    button_style()
    input_style()
