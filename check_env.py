# check_env.py
print("开始检查项目环境依赖...\n")

dependencies = {
    "FastAPI": "fastapi",
    "Uvicorn": "uvicorn",
    "Multipart (用于文件上传)": "multipart", 
    "OpenCV": "cv2",
    "Pillow": "PIL",
    "Ultralytics (YOLO)": "ultralytics",
    "Streamlit": "streamlit",
    "PyTorch": "torch"
}

all_passed = True

for name, module in dependencies.items():
    try:
        if module == "multipart":
            # python-multipart 比较特殊，import 时叫 python_multipart 或 multipart
            import multipart
        else:
            __import__(module)
        print(f"✅ {name} 已就绪")
    except ImportError:
        print(f"❌ 缺失依赖: {name} (请运行: pip install {module if module != 'cv2' and module != 'PIL' else 'opencv-python' if module == 'cv2' else 'Pillow'})")
        # 特殊处理 multipart 的提示
        if module == "multipart":
             print("   -> 修复命令: pip install python-multipart")
        all_passed = False

print("\n--- 检查结果 ---")
if all_passed:
    print("🎉 恭喜！所有核心依赖均已成功安装并可以正常导入！")
else:
    print("⚠️ 发现缺失项，请根据上述提示安装缺失的包。")
