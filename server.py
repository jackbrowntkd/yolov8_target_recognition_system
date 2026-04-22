"""
YOLO目标检测后端服务
基于FastAPI框架，提供图像上传和目标检测接口
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
from typing import List, Dict

# 创建FastAPI应用实例
app = FastAPI(
    title="YOLO目标检测API",
    description="基于YOLOv8的目标检测服务，支持图像上传和实时检测",
    version="1.0.0"
)

# ========================================
# 配置CORS跨域（重要！）
# ========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境应配置具体域名）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# ========================================
# 全局变量：加载YOLO模型
# ========================================
# 使用yolov8n（nano版本，速度快，适合CPU）
# 首次运行会自动下载模型文件（约6MB）
MODEL_PATH = "yolov8n.pt"
print(f"正在加载YOLO模型: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("✓ 模型加载完成！")


def read_image_from_upload(upload_file: UploadFile) -> np.ndarray:
    """
    从上传的文件对象中读取图像数据，转换为OpenCV格式

    参数:
        upload_file: FastAPI的上传文件对象

    返回:
        np.ndarray: OpenCV图像格式（BGR）
    """
    # 读取文件内容
    contents = upload_file.file.read()

    # 转换为PIL Image对象
    pil_image = Image.open(io.BytesIO(contents))

    # 转换为OpenCV格式 (RGB -> BGR)
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return opencv_image


def image_to_base64(image: np.ndarray, format: str = ".jpg") -> str:
    """
    将OpenCV图像转换为Base64编码字符串

    参数:
        image: OpenCV图像数组
        format: 图像格式（.jpg 或 .png）

    返回:
        str: Base64编码的图像字符串
    """
    _, buffer = cv2.imencode(format, image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


@app.get("/")
async def root():
    """根路径接口，返回服务状态"""
    return {
        "service": "YOLO目标检测API",
        "status": "running",
        "model": MODEL_PATH,
        "endpoints": [
            {"path": "/", "method": "GET", "description": "服务状态"},
            {"path": "/detect", "method": "POST", "description": "目标检测"}
        ]
    }


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(..., description="上传的图像文件")
):
    """
    目标检测接口

    参数:
        file: 上传的图像文件（支持jpg、png等格式）

    返回:
        JSON响应，包含:
        - success: 是否成功
        - objects: 检测到的目标列表
            - bbox: 边界框 [x1, y1, x2, y2]
            - conf: 置信度 (0-1)
            - class_id: 类别ID
            - class_name: 类别名称
        - image_with_boxes: 带检测框的图像（Base64编码）
        - total_objects: 检测到的目标总数
    """

    try:
        # 1. 读取图像
        image = read_image_from_upload(file)
        original_shape = image.shape  # (height, width, channels)

        # 2. YOLO推理（自动进行预处理：缩放到640×640、归一化）
        results = model(image)

        # 3. 处理检测结果
        detections = []
        result = results[0]  # results是列表，取第一张图的结果

        # 遍历每个检测框
        for box in result.boxes:
            # 获取坐标（xyxy格式：左上角x, 左上角y, 右下角x, 右下角y）
            xyxy = box.xyxy[0].cpu().numpy()  # 转为numpy数组

            # 获取置信度
            conf = float(box.conf[0].cpu().numpy())

            # 获取类别ID和名称
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]

            # 整理检测结果
            detection = {
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "conf": round(conf, 4),
                "class_id": class_id,
                "class_name": class_name
            }
            detections.append(detection)

        # 4. 在图像上绘制检测框和标签
        result_image = result.plot()  # ultralytics提供的绘图方法

        # 5. 将带检测框的图像转换为Base64
        image_base64 = image_to_base64(result_image, format=".jpg")

        # 6. 返回JSON响应
        return {
            "success": True,
            "image_info": {
                "height": int(original_shape[0]),
                "width": int(original_shape[1])
            },
            "total_objects": len(detections),
            "objects": detections,
            "image_with_boxes": f"data:image/jpeg;base64,{image_base64}"
        }

    except Exception as e:
        # 错误处理
        print(f"检测失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"检测失败: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "model": MODEL_PATH}


if __name__ == "__main__":
    # 使用Uvicorn启动ASGI服务器
    # host="0.0.0.0": 允许外部访问
    # port=8000: 服务端口
    # reload=True: 代码修改后自动重启（开发模式）
    import uvicorn
    uvicorn.run(
        "server:app",  # 修改这里：必须加上引号，格式为 "文件名:实例名"
        host="0.0.0.0",
        port=8000,
        reload=True
    )