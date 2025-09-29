
import sys
import os
import tempfile
import subprocess
import shutil
from pathlib import Path

def resource_path(relative_path):
    """获取资源的绝对路径"""
    try:
        # PyInstaller创建的临时文件夹
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def main():
    # 获取web.py的路径
    web_script_path = resource_path("web.py")
    
    # 如果是打包后的exe，需要将资源文件提取到临时目录
    if getattr(sys, 'frozen', False):
        # 创建临时目录存放资源文件
        temp_dir = tempfile.mkdtemp()
        resources_to_copy = [
            "weights",
            "tempDir",
            "chinese_name_list.py",
            "log.py",
            "model.py",
            "ui_style.py",
            "utils.py"
        ]
        
        for resource in resources_to_copy:
            src = resource_path(resource)
            if os.path.exists(src):
                dst = os.path.join(temp_dir, resource)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
        
        # 设置工作目录到临时目录
        os.chdir(temp_dir)
    
    # 运行streamlit应用
    cmd = [sys.executable, "-m", "streamlit", "run", web_script_path, "--server.port", "8501", "--server.address", "0.0.0.0"]
    try:
        subprocess.call(cmd)
    except Exception as e:
        print(f"运行应用时出错: {e}")
        input("按Enter键退出...")

if __name__ == "__main__":
    main()
