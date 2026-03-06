# -*- coding: utf-8 -*-
"""
配置模块 —— 类别名称映射、标签列表、AI 接口配置、报告模板。
"""

# ──────────────────────── 路面病害类别中文名称映射 ────────────────────────
# key = YOLO 模型输出的英文类别名, value = 展示用中文名
Chinese_name: dict[str, str] = {
    "Construction Materials": "施工材料",
    "Crack": "裂缝",
    "Damaged Curb": "受损路缘",
    "Digging": "挖掘",
    "Improper Barrier": "不当屏障",
    "Improper Fencing": "不当围栏",
    "Non Painted Curb": "未涂漆路缘",
    "Painted Curb": "涂漆路缘",
    "Ponding Water": "积水",
    "Pothole": "坑洞",
}

# 标签列表（用于 UI 等处直接引用）
Label_list: list[str] = list(Chinese_name.values())

# ──────────────────────── AI 报告生成 - API 配置 ────────────────────────
API_KEY: str = "sk-fovgfnbkhlhzmoabtajxmysdnenytnmiewdultwfbxlxsrkw"   # 替换为你自己的密钥
API_URL: str = "https://api.siliconflow.cn/v1/chat/completions"           # 硅基流动 API 地址
MODEL_NAME: str = "Pro/deepseek-ai/DeepSeek-V3"                          # 模型名称

# ──────────────────────── AI 报告 - 提示词模板 ─────────────────────────
BASE_PROMPT: str = """
你是一个专业的图像分析助手。请根据以下检测数据生成一份详细的分析报告：

{detection_data}

请按照以下要求生成报告：
1. 分析检测结果的主要发现
2. 提供专业的解释和建议
3. 使用清晰的结构和专业的语言
4. 包含必要的技术细节
"""

REPORT_TEMPLATES: dict[str, str] = {
    "详细报告": """
请生成一份详细的报告，包含以下内容：
1. 检测概述
2. 详细分析
3. 技术指标
4. 建议措施
5. 总结
""",
    "管理报告": """
请生成一份面向管理层的报告，包含以下内容：
1. 执行摘要
2. 关键发现
3. 风险评估
4. 建议行动
5. 结论
""",
    "技术报告": """
请生成一份技术报告，包含以下内容：
1. 技术概述
2. 详细分析
3. 数据解读
4. 技术建议
5. 技术总结
""",
}


def test_api_connection() -> bool:
    """测试 AI API 连接是否正常。"""
    import requests

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "测试连接"}],
                "temperature": 0.7,
                "max_tokens": 10,
            },
            timeout=15,
        )
        if response.status_code == 200:
            print("API 连接测试成功")
            return True
        print(f"API 连接测试失败: {response.status_code} - {response.text}")
        return False
    except Exception as exc:
        print(f"API 连接测试出错: {exc}")
        return False
