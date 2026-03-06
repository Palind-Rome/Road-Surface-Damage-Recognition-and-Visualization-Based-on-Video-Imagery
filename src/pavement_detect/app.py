# -*- coding: utf-8 -*-
"""
快速启动 Streamlit 应用的入口文件。
"""
import sys
import subprocess
from pathlib import Path

def main() -> None:
    # 获取并构造 `web.py` 路径
    current_dir = Path(__file__).parent.resolve()
    web_script = current_dir / "web.py"
    
    if not web_script.exists():
        print(f"找不到入口脚本: {web_script}")
        sys.exit(1)
        
    cmd = [sys.executable, "-m", "streamlit", "run", str(web_script)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n应用已退出。")

if __name__ == "__main__":
    main()
