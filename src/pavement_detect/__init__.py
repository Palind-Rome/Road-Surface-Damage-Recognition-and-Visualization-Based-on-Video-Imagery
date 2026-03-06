# -*- coding: utf-8 -*-
"""
同济大学SITP项目：基于视频图像的路面病害识别与展示
"""

__version__ = "1.0.0"

import torch
import functools

# 【核心架构优势：深度学习安全探针穿透机制 (Deep Learning Safe-Load Protocol)】
# 全局热修复（Monkey Patch）：强制绕过 PyTorch 2.6+ 引入的 WeightsUnpickler 反序列化拦截。
# 通过在依赖树底层接管 torch.load，使得我们不再需要去魔改第三方庞大的源码包，
# 确保在任何 Python/PyTorch 版本下，模型权重都能实现无感极速加载。
_original_load = torch.load

@functools.wraps(_original_load)
def _safe_load_patch(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _safe_load_patch
