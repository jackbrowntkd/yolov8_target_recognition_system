# Windows版 YOLO目标检测系统 - 完整实现指南

## 📋 目录

- [系统架构概述](#系统架构概述)
  
- [YOLO模型详细介绍](#yolo模型详细介绍)
  
- [环境准备](#环境准备)
  
- [后端实现（FastAPI）](#后端实现fastapi)
  
- [前端实现（Streamlit）](#前端实现streamlit)
  
- [系统集成与测试](#系统集成与测试)
  
- [常见问题解决](#常见问题解决)
  

---

## 1. 系统架构概述

### 1.1 技术栈选型

| 组件  | 技术选型 | 版本  | 说明  |
| --- | --- | --- | --- |
| **模型** | YOLOv8 (Ultralytics) | 8.x | 业界最先进的目标检测模型，易用性强 |
| **后端框架** | FastAPI | 0.104+ | 高性能异步Web框架，支持自动API文档 |
| **ASGI服务器** | Uvicorn | 0.24+ | 异步ASGI服务器，服务FastAPI |
| **前端框架** | Streamlit | 1.28+ | 无需HTML/CSS，用Python快速构建Web界面 |
| **Python版本** | Python | 3.8+ | 推荐使用3.9或3.10 |

### 1.2 系统工作流程

```plain
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  用户浏览器   │  访问   │  Streamlit  │  HTTP   │  FastAPI    │
│  (客户端)    │ <-----> │  (前端界面)  │ <-----> │  (后端服务)  │
└─────────────┘  8501端口 └─────────────┘  8000端口 └─────────────┘
                                               │
                                               │ 推理
                                               ↓
                                          ┌─────────────┐
                                          │  YOLOv8     │
                                          │  目标检测    │
                                          └─────────────┘
```

**数据流说明：**

1. 用户在浏览器访问前端界面（Streamlit，端口8501）
  
2. 用户上传图片
  
3. 前端通过HTTP POST请求将图片发送到后端FastAPI（端口8000）
  
4. 后端接收图片，调用YOLOv8模型进行目标检测
  
5. 后端将检测结果（边界框、类别、置信度）返回给前端
  
6. 前端在界面上绘制检测框并显示结果
  

---

## 2. YOLO模型详细介绍

### 2.1 YOLO (You Only Look Once) 核心原理

YOLO是一种**单阶段目标检测算法**，与传统的R-CNN系列不同，它将目标检测问题回归为**单一的回归问题**。

#### 2.1.1 核心思想

- **一次性预测**：将整个图像输入网络，一次性输出所有检测框
  
- **网格划分**：将输入图像划分成 S×S 个网格
  
- **边界框预测**：每个网格负责预测中心点落在该网格内的目标
  

#### 2.1.2 YOLOv8 架构详解

```plain
输入图像 (640×640×3)
    │
    ├── Backbone (特征提取网络)
    │   ├── CSPDarknet (跨阶段部分网络)
    │   ├── SPPF (空间金字塔池化快速)
    │   └── 特征图多尺度提取
    │
    ├── Neck (特征融合网络)
    │   ├── PANet (路径聚合网络)
    │   └── FPN (特征金字塔网络)
    │
    └── Head (检测头)
        ├── 解耦头 (分类+回归分离)
        ├── 任务对齐分配器
        └── 输出: [x, y, w, h, conf, cls]
```

### 2.2 YOLOv8 关键创新点

#### 2.2.1 C2f 模块（核心改进）

```python
# 概念图示
┌─────────────────────────────────────┐
│           C2f 模块                   │
│  ┌──► Split ──► Bottleneck1 ──►Concat─┐
│  │             ↓                    │
│  │          Bottleneck2              │
│  │             ↓                    │
│  └───────────────────────────────────┘
```

- **作用**：替代了YOLOv5的C3模块
  
- **优势**：参数量更少，计算量更优，特征融合能力更强
  

#### 2.2.2 解耦检测头（Decoupled Head）

- 传统YOLO使用耦合头（分类和回归共用特征）
  
- YOLOv8将**分类分支**和**回归分支**解耦
  
- 优势：减少任务间的相互干扰，提升检测精度
  

#### 2.2.3 任务对齐学习（Task Aligned Learning）

- 引入TAL（Task Aligned Learning）策略
  
- 动态调整样本权重，让高质量锚点参与训练
  
- 显著提升小目标检测能力
  

### 2.3 模型输出格式详解

YOLOv8对每张图片输出一个张量：

```plain
输出维度: [batch, num_boxes, 84]
         └───84 = 4 (bbox) + 1 (confidence) + 80 (COCO classes)
         └───4 (bbox): [x, y, w, h] (中心点坐标、宽、高)
         └───80 classes: COCO数据集的80个类别概率
```

**COCO数据集类别示例：**

- 0: person (人)
  
- 1: bicycle (自行车)
  
- 2: car (汽车)
  
- 3: motorcycle (摩托车)
  
- ...
  

### 2.4 推理流程

```plain
输入图像 → 预处理(640×640, 归一化) → 
    模型推理 → 
    后处理(NMS非极大值抑制) → 
    输出检测框 [x1, y1, x2, y2, confidence, class_id]
```

---

## 3. 环境准备

### 3.1 系统要求

- **操作系统**：Windows 10/11 (64位)
  
- **Python**：3.8, 3.9 或 3.10（推荐3.10）
  
- **内存**：至少8GB RAM
  
- **硬盘**：至少5GB可用空间
  
- **GPU**：可选（NVIDIA显卡可加速，但CPU也能运行）
  

### 3.2 安装Python

1. 访问 [Python官网](https://www.python.org/downloads/)
  
2. 下载Python 3.10.x Windows installer
  
3. 安装时 **勾选 "Add Python to PATH"**
  
4. 验证安装：
  

```plain
python --version
```

### 3.3 创建虚拟环境

```plain
# 1. 进入项目目录
cd C:\yolo_detection

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
venv\Scripts\activate

# 激活成功后，命令行前面会显示 (venv)
```

### 3.4 安装依赖

创建 `requirements.txt` 文件：

```plain
ultralytics
fastapi
uvicorn
python-multipart
streamlit
requests
pillow
opencv-python
```

安装依赖：

```plain
# 如果下载慢，使用国内镜像源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 3.5 项目目录结构

```plain
C:\yolo_detection\
│
├── venv/                    # 虚拟环境目录
│
├── server.py                # FastAPI后端服务
├── client.py                # Streamlit前端界面
├── requirements.txt         # 依赖包列表
│
├── runs/                    # YOLO模型保存目录（自动生成）
│   └── detect/
│
└── uploads/                 # 临时上传目录（手动创建）
```

---

## 4. 后端实现（FastAPI）

### 4.1 完整代码（server.py）

```python
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
```

### 4.2 代码详细讲解

#### 4.2.1 导入模块说明

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
# FastAPI: 核心框架类
# File: 用于接收上传文件
# UploadFile: 上传文件的数据类型
# HTTPException: 处理异常响应

from fastapi.middleware.cors import CORSMiddleware
# CORSMiddleware: 解决跨域问题

from ultralytics import YOLO
# YOLO: Ultralytics提供的YOLOv8类

import cv2
import numpy as np
from PIL import Image
# 图像处理库

import io
import base64
import json
# 数据序列化和Base64编码
```

#### 4.2.2 CORS配置详解

**为什么需要CORS？**

- 前端（Streamlit默认8501端口）和后端（FastAPI默认8000端口）运行在不同端口
  
- 浏览器的同源策略会阻止跨端口请求
  
- 必须配置CORS允许前端访问后端
  

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许所有域名来源
    allow_credentials=True,     # 允许携带凭证（如cookie）
    allow_methods=["*"],        # 允许所有HTTP方法（GET、POST等）
    allow_headers=["*"],        # 允许所有请求头
)
```

**生产环境建议：**

```python
# 只允许特定域名
allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"]
```

#### 4.2.3 YOLO模型加载

```python
model = YOLO("yolov8n.pt")
```

**YOLO模型版本选择：**

| 模型  | 文件大小 | 速度  | 精度  | 适用场景 |
| --- | --- | --- | --- | --- |
| yolov8n.pt | 约6MB | 最快  | 基准  | CPU设备、实时检测 |
| yolov8s.pt | 约22MB | 快   | 较高  | 边缘设备 |
| yolov8m.pt | 约50MB | 中   | 高   | 平衡选择 |
| yolov8l.pt | 约83MB | 慢   | 很高  | GPU设备 |
| yolov8x.pt | 约130MB | 最慢  | 极高  | 精度优先场景 |

#### 4.2.4 图像处理函数详解

**read_image_from_upload函数：**

```python
def read_image_from_upload(upload_file: UploadFile) -> np.ndarray:
    # 1. 读取原始二进制数据
    contents = upload_file.file.read()

    # 2. PIL解码（处理各种图片格式：jpg、png等）
    pil_image = Image.open(io.BytesIO(contents))

    # 3. 转换为numpy数组（RGB格式）
    rgb_array = np.array(pil_image)

    # 4. RGB -> BGR（OpenCV使用BGR格式）
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    return bgr_array
```

**为什么需要格式转换？**

- PIL和大多数图像库使用RGB格式
  
- OpenCV使用BGR格式（历史原因）
  
- YOLO内部会自动处理格式，但保持统一避免潜在问题
  

#### 4.2.5 检测接口详解

**请求处理流程：**

```python
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # 步骤1: 读取并验证图像
    image = read_image_from_upload(file)

    # 步骤2: 调用YOLO推理
    results = model(image)
    # results是一个列表，即使只检测一张图也是列表格式
    # results[0]包含：
    #   - results[0].boxes: 所有检测框
    #   - results[0].names: 类别名称映射
    #   - results[0].speed: 推理速度统计

    # 步骤3: 解析检测框
    detections = []
    result = results[0]

    for box in result.boxes:
        # box.xyxy: 边界框 [x1, y1, x2, y2]
        # box.conf: 置信度
        # box.cls: 类别ID

        # 重要：数据在GPU上需要转回CPU并转为numpy
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        # 记录检测结果
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "class_id": class_id,
            "class_name": model.names[class_id]
        })

    # 步骤4: 绘制检测框
    result_image = result.plot()
    # plot()方法会自动绘制：
    #   - 彩色边界框（不同类别不同颜色）
    #   - 类别标签
    #   - 置信度分数

    # 步骤5: 图像转Base64（便于网络传输）
    image_base64 = image_to_base64(result_image)

    # 步骤6: 返回JSON响应
    return {...}
```

**响应格式示例：**

```json
{
  "success": true,
  "image_info": {
    "height": 1080,
    "width": 1920
  },
  "total_objects": 3,
  "objects": [
    {
      "bbox": [100, 200, 300, 500],
      "conf": 0.8523,
      "class_id": 0,
      "class_name": "person"
    },
    {
      "bbox": [400, 150, 600, 400],
      "conf": 0.7234,
      "class_id": 2,
      "class_name": "car"
    }
  ],
  "image_with_boxes": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

#### 4.2.6 Uvicorn服务器配置

```python
uvicorn.run(
    app,              # FastAPI应用实例
    host="0.0.0.0",   # 绑定所有网络接口（允许外部访问）
    port=8000,        # 端口号
    reload=True       # 开发模式：代码修改自动重启
)
```

**host参数说明：**

- `127.0.0.1`: 仅本机可访问
  
- `0.0.0.0`: 允许局域网内其他设备访问
  

### 4.3 启动后端服务

**方法1：直接运行Python文件**

```plain
# 确保虚拟环境已激活
venv\Scripts\activate

# 启动服务
python server.py
```

**方法2：使用Uvicorn命令**

```plain
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

**启动成功后的输出：**

```plain
正在加载YOLO模型: yolov8n.pt...
✓ 模型加载完成！
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**访问API文档：**

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
  
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
  

---

## 5. 前端实现（Streamlit）

### 5.1 完整代码（client.py）

```python
"""
YOLO目标检测前端界面
基于Streamlit框架，提供图片上传和检测结果展示
"""

import streamlit as st
import requests
from PIL import Image
import cv2
import numpy as np
import io
import base64

# ========================================
# 配置页面信息
# ========================================
st.set_page_config(
    page_title="YOLO目标检测系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# 配置后端API地址
# ========================================
# 注意：如果是前后端分离部署（不同机器），修改此地址
API_URL = "http://127.0.0.1:8000/detect"

# ========================================
# CSS样式（美化界面）
# ========================================
def load_custom_css():
    st.markdown("""
    <style>
        /* 主标题样式 */
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }

        /* 结果卡片样式 */
        .result-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* 统计数字样式 */
        .stat-number {
            font-size: 3rem;
            font-weight: bold;
            color: #2ecc71;
        }

        /* 类别标签样式 */
        .class-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)

# ========================================
# 辅助函数
# ========================================

def predict_with_yolo(image_file):
    """
    调用后端API进行目标检测

    参数:
        image_file: 上传的图像文件

    返回:
        dict: 后端返回的检测结果
    """
    try:
        # 【核心修复】：把指针重置回开头！
        image_file.seek(0) 

        # 准备请求数据 (这里建议直接传入真实的字节数据)
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}

        # 发送POST请求到后端
        response = requests.post(API_URL, files=files, timeout=30)

        # 解析响应
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"后端错误: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到后端服务！请确保后端服务已启动。")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ 请求超时，请重试。")
        return None
    except Exception as e:
        st.error(f"❌ 发生错误: {str(e)}")
        return None


def base64_to_image(base64_str):
    """
    将Base64编码的图像字符串转换为PIL Image

    参数:
        base64_str: Base64图像字符串

    返回:
        PIL.Image: 图像对象
    """
    # 去掉"data:image/jpeg;base64,"前缀
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    # Base64解码
    img_bytes = base64.b64decode(base64_str)

    # 转换为PIL Image
    image = Image.open(io.BytesIO(img_bytes))

    return image


def get_stats_by_class(objects):
    """
    统计各类别的检测数量

    参数:
        objects: 检测对象列表

    返回:
        dict: {类别名: 数量}
    """
    stats = {}
    for obj in objects:
        class_name = obj["class_name"]
        if class_name not in stats:
            stats[class_name] = 0
        stats[class_name] += 1
    return stats


# ========================================
# 主界面
# ========================================

def main():
    # 加载自定义CSS
    load_custom_css()

    # 侧边栏
    with st.sidebar:
        st.markdown("## ⚙️ 系统设置")

        st.markdown("### 🔗 后端配置")
        api_url_input = st.text_input(
            "API地址",
            value=API_URL,
            help="后端FastAPI服务的地址"
        )

        # 更新全局API地址
        st.session_state["API_URL"] = api_url_input

        st.markdown("---")
        st.markdown("### 📊 检测选项")
        confidence_threshold = st.slider(
            "置信度阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="只显示置信度高于此值的结果"
        )

        st.markdown("---")
        st.markdown("### 💡 使用说明")
        st.info("""
        1. 点击「上传图片」按钮选择图片
        2. 系统自动调用YOLO模型检测
        3. 查看检测结果和统计分析

        支持格式: JPG、PNG、BMP等
        """)

    # 主区域
    st.markdown('<h1 class="main-title">🎯 YOLO目标检测系统</h1>', unsafe_allow_html=True)

    # 图片上传区域
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        uploaded_file = st.file_uploader(
            "📤 上传图片进行检测",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility="collapsed"
        )

    # 如果用户上传了图片
    if uploaded_file is not None:
        st.markdown("---")

        # 显示两列：原图和检测结果
        col_img1, col_img2 = st.columns(2)

        # 显示原图
        with col_img1:
            st.markdown("### 📷 原始图片")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_column_width=True)

        # 调用API检测
        with st.spinner("🔄 正在检测中，请稍候..."):
            result = predict_with_yolo(uploaded_file)

        # 显示检测结果
        if result and result.get("success"):
            with col_img2:
                st.markdown("### ✅ 检测结果")

                # 将Base64图像转换为PIL Image
                result_image = base64_to_image(result["image_with_boxes"])
                st.image(
                    result_image,
                    use_column_width=True,
                    caption=f"检测到 {result['total_objects']} 个目标"
                )

            # 统计信息
            st.markdown("---")
            st.markdown("### 📊 统计分析")

            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### 检测目标总数")
                st.markdown(f'<p class="stat-number">{result["total_objects"]}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_stat2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### 图像尺寸")
                st.markdown(f"**宽度**: {result['image_info']['width']} px<br>**高度**: {result['image_info']['height']} px", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_stat3:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### 最高置信度")
                if result["objects"]:
                    max_conf = max(obj["conf"] for obj in result["objects"])
                    st.markdown(f'<p class="stat-number">{max_conf:.2%}</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="stat-number">-</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 详细检测结果
            if result["objects"]:
                st.markdown("---")
                st.markdown("### 📋 详细检测结果")

                # 按类别统计
                class_stats = get_stats_by_class(result["objects"])

                # 显示类别统计
                st.markdown("#### 类别分布")
                for class_name, count in class_stats.items():
                    st.markdown(
                        f'<span class="class-badge" style="background-color: #3498db; color: white;">'
                        f'{class_name}: {count}'
                        f'</span>',
                        unsafe_allow_html=True
                    )

                # 详细表格
                st.markdown("#### 检测清单")

                # 根据置信度阈值过滤结果
                filtered_objects = [
                    obj for obj in result["objects"]
                    if obj["conf"] >= confidence_threshold
                ]

                if filtered_objects:
                    # 转换为DataFrame格式显示
                    import pandas as pd
                    df_data = []
                    for obj in filtered_objects:
                        x1, y1, x2, y2 = obj["bbox"]
                        df_data.append({
                            "类别": obj["class_name"],
                            "置信度": f"{obj['conf']:.2%}",
                            "位置": f"({x1}, {y1}) → ({x2}, {y2})"
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning(f"根据当前置信度阈值 ({confidence_threshold:.0%})，没有符合条件的检测结果。")

            # 提供下载按钮
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl2:
                st.download_button(
                    label="📥 下载检测结果图片",
                    data=base64.b64decode(result["image_with_boxes"].split(",")[1]),
                    file_name=f"result_{uploaded_file.name}",
                    mime="image/jpeg"
                )
        else:
            st.error("❌ 检测失败，请查看错误信息。")

    # 示例图片区域
    else:
        st.markdown("---")
        st.markdown("### 💡 尚未上传图片")
        st.info("请上传一张图片开始检测，或查看下方示例说明。")

        st.markdown("""
        #### 支持的检测类别（COCO数据集）

        | 类别 | 类别 | 类别 | 类别 |
        |:-----|:-----|:-----|:-----|
        | person 人 | bicycle 自行车 | car 汽车 | motorcycle 摩托车 |
        | airplane 飞机 | bus 巴士 | train 火车 | truck 卡车 |
        | boat 船 | traffic light 红绿灯 | fire hydrant 消防栓 | stop sign 停止标志 |
        | parking meter | bench 长椅 | bird 鸟 | cat 猫 |
        | dog 狗 | horse 马 | sheep 羊 | cow 牛 |
        | elephant 大象 | bear 熊 | zebra 斑马 | giraffe 长颈鹿 |
        | backpack 背包 | umbrella 雨伞 | handbag 手提包 | tie 领带 |
        | suitcase 行李箱 | frisbee 飞盘 | skis 滑雪板 | snowboard 滑雪橇 |
        | sports ball 运动球 | kite 风筝 |棒球棒 | baseball glove |
        | skateboard 滑板 | surfboard 冲浪板 | tennis racket 网球拍 | bottle 瓶子 |
        | wine glass 酒杯 | cup 杯子 | fork 叉子 | knife 刀 |
        | spoon 勺子 | bowl 碗 | banana 香蕉 | apple 苹果 |
        | sandwich 三明治 | orange 橙子 | broccoli 西兰花 | carrot 胡萝卜 |
        | hot dog 热狗 | pizza 披萨 | donut 甜甜圈 | cake 蛋糕 |
        | chair 椅子 | couch 沙发 | potted plant 盆栽 | bed 床 |
        | dining table 餐桌 | toilet 厕所 | tv 电视 | laptop 笔记本 |
        | mouse 鼠标 | remote 遥控器 | keyboard 键盘 | cell phone 手机 |
        | microwave 微波炉 | oven 烤箱 | toaster 烤面包机 | sink 水槽 |
        | refrigerator 冰箱 | book 书 | clock 时钟 | vase 花瓶 |
        | scissors 剪刀 | teddy bear 泰迪熊 | hair drier 吹风机 | toothbrush 牙刷 |
        """)


if __name__ == "__main__":
    main()
```

### 5.2 代码详细讲解

#### 5.2.1 Streamlit页面配置

```python
st.set_page_config(
    page_title="YOLO目标检测系统",   # 浏览器标签页标题
    page_icon="🎯",                  # 图标
    layout="wide",                   # 布局：wide（宽屏）或 centered（居中）
    initial_sidebar_state="expanded" # 侧边栏初始状态
)
```

**可选参数：**

- `layout="wide"`: 使用更宽的布局，适合展示图片和表格
  
- `initial_sidebar_state="collapsed"`: 默认收起侧边栏
  

#### 5.2.2 自定义CSS样式

Streamlit允许注入自定义CSS来美化界面：

```python
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    /* 更多样式... */
</style>
""", unsafe_allow_html=True)
```

**重要注意事项：**

- 必须加上 `unsafe_allow_html=True` 参数
  
- CSS选择器建议使用自定义类名，避免与Streamlit样式冲突
  

#### 5.2.3 调用后端API

```python
def predict_with_yolo(image_file):
    # 准备文件数据
    files = {
        "file": (
            image_file.name,      # 文件名
            image_file,           # 文件对象
            image_file.type       # MIME类型（如"image/jpeg"）
        )
    }

    # 发送POST请求
    response = requests.post(API_URL, files=files, timeout=30)

    # 返回JSON响应
    return response.json()
```

**requests参数说明：**

- `files`: 使用 `multipart/form-data` 格式上传文件
  
- `timeout`: 超时时间（秒），防止长时间等待
  

#### 5.2.4 Base64图像转换

```python
def base64_to_image(base64_str):
    # 步骤1: 去掉前缀
    # 格式: "data:image/jpeg;base64,/9j/4AAQ..."
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    # 步骤2: Base64解码
    img_bytes = base64.b64decode(base64_str)

    # 步骤3: 转为BytesIO对象
    io_bytes = io.BytesIO(img_bytes)

    # 步骤4: PIL解码
    image = Image.open(io_bytes)

    return image
```

**为什么使用Base64？**

- 便于在JSON中传输二进制数据
  
- 可直接在HTML中显示：`<img src="data:image/jpeg;base64,...">`
  

#### 5.2.5 Streamlit布局技巧

**三列布局：**

```python
col1, col2, col3 = st.columns([1, 2, 1])
# 比例为 1:2:1

with col1:
    st.write("左侧内容")

with col2:
    st.write("中间内容（宽度是两侧的2倍）")

with col3:
    st.write("右侧内容")
```

**条件显示区域：**

```python
with st.spinner("加载中..."):
    # 耗时操作
    result = api_call()
    # 显示进度环
```

#### 5.2.6 数据展示组件

**数据表格（DataFrame）：**

```python
import pandas as pd

data = {
    "类别": ["person", "car", "dog"],
    "置信度": [0.95, 0.87, 0.92],
    "位置": [(100,200,300,400), (500,200,700,400), (200,300,400,500)]
}

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True, hide_index=True)
```

**指标卡片：**

```python
col1, col2 = st.columns(2)

with col1:
    st.metric("检测总数", "5", "↑ 2")
    # st.metric(标签, 值, 增量变化)
```

#### 5.2.7 文件下载功能

```python
st.download_button(
    label="📥 下载图片",              # 按钮文本
    data=image_bytes,                # 二进制数据
    file_name="result.jpg",          # 下载后的文件名
    mime="image/jpeg",               # MIME类型
    type="primary",                  # 按钮样式
    use_container_width=True         # 使用容器全宽
)
```

### 5.3 启动前端服务

**方法1：直接运行**

```plain
# 确保虚拟环境已激活
venv\Scripts\activate

# 启动Streamlit
streamlit run client.py --server.port 8501
```

**方法2：指定工作目录（可选）**

```plain
streamlit run client.py --server.port 8501 --server.dir C:\yolo_detection
```

**启动成功后的输出：**

```plain
You can now view your Streamlit app in your browser.

  Local URL: http://127.0.0.1:8501
  Network URL: http://192.168.x.x:8501

For better performance, install the Watchdog module:
  pip install watchdog
```

**访问界面：**

- 本地访问：[http://127.0.0.1:8501](http://127.0.0.1:8501)
  
- 局域网访问：[http://192.168.x.x:8501（其他设备可访问）](http://192.168.x.x:8501（其他设备可访问）)
  

---

## 6. 系统集成与测试

### 6.1 完整启动流程

#### 步骤1：打开两个终端

**终端1 - 启动后端：**

```plain
cd C:\yolo_detection
venv\Scripts\activate
python server.py
```

等到看到输出：

```plain
✓ 模型加载完成！
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**终端2 - 启动前端：**

```plain
cd C:\yolo_detection
venv\Scripts\activate
streamlit run client.py --server.port 8501
```

等到看到输出：

```plain
Local URL: http://127.0.0.1:8501
```

#### 步骤2：浏览器访问

打开浏览器，访问：

```plain
http://127.0.0.1:8501
```

#### 步骤3：测试系统

1. 点击"📤 上传图片进行检测"
  
2. 选择一张包含人物、车辆等物体的图片
  
3. 等待检测结果
  
4. 查看检测框、统计信息和详细表格
  

### 6.2 测试用例

| 测试场景 | 预期结果 |
| --- | --- |
| 上传JPG图片 | 成功检测，显示边界框 |
| 上传PNG图片 | 成功检测，显示边界框 |
| 上传非图片文件 | 后端返回错误，前端显示错误信息 |
| 未启动后端 | 前端显示"无法连接到后端服务" |
| 多人场景图片 | 正确检测多个person |
| 小目标图片 | 能检测到小物体（置信度可能较低） |
| 空场景图片 | 返回"检测到0个目标" |

### 6.3 性能测试

**测试图片大小：**

- 小图片：640×480 (约30KB) - 推理时间约100-200ms
  
- 中图片：1920×1080 (约500KB) - 推理时间约300-500ms
  
- 大图片：4000×3000 (约3MB) - 推理时间约1000-2000ms
  

**优化建议：**

- 对于大图，可在前端先压缩再上传
  
- 使用GPU加速可提升5-10倍速度
  
- 根据需求选择不同大小的YOLO模型（n/s/m/l/x）
  

### 6.4 分离部署（前后端不同机器）

#### 后端部署（服务器）

```plain
# 服务器上运行
python server.py
```

#### 前端部署（客户端）

修改 `client.py` 中的API地址：

```python
# 服务器IP地址
API_URL = "http://192.168.1.100:8000/detect"
```

启动前端：

```plain
streamlit run client.py --server.port 8501
```

**注意事项：**

- 确保服务器防火墙开放8000端口
  
- 前端和后端需要在同一局域网或网络互通
  

---

## 7. 常见问题解决

### 7.1 后端问题

**问题1：启动时提示 "ModuleNotFoundError: No module named 'ultralytics'"**

解决方案：

```plain
# 确认虚拟环境已激活
venv\Scripts\activate

# 重新安装依赖
pip install ultralytics
```

**问题2：模型下载失败**

解决方案：

```python
# 方法1：手动下载模型文件
# 访问 https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
# 下载后放到项目根目录

# 方法2：使用国内镜像（如果支持）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

**问题3：CORS跨域错误**

解决方案：

确认 `server.py` 中已添加：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**问题4：端口8000被占用**

解决方案：

```plain
# 查看占用8000端口的进程
netstat -ano | findstr 8000

# 终止进程（PID是上面查到的进程号）
taskkill /PID <PID> /F

# 或者修改端口
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 7.2 前端问题

**问题1：前端无法访问后端**

解决方案：

1. 确认后端服务已启动
  
2. 检查API地址配置是否正确
  
3. 尝试在浏览器直接访问：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
  

**问题2：图片上传后无响应**

解决方案：

```plain
# 查看前端终端是否有报错
# 查看后端终端是否收到请求

# 检查网络请求（浏览器F12 -> Network）
```

**问题3：检测结果图片显示异常**

解决方案：

确认 `base64_to_image` 函数正确处理了前缀：

```python
if "," in base64_str:
    base64_str = base64_str.split(",")[1]
```

### 7.3 环境问题

**问题1：虚拟环境激活失败**

错误提示：`无法加载venv\Scripts\activate.ps1`

解决方案：

```plain
# PowerShell中需要更改执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 然后重新激活
venv\Scripts\activate
```

**问题2：pip安装速度慢**

解决方案：

```plain
# 使用国内镜像源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 或永久配置
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

### 7.4 性能优化建议

**1. 使用GPU加速**

```plain
# 安装CUDA版本的PyTorch（如果使用NVIDIA显卡）
# 访问 https://pytorch.org/get-started/locally/ 获取合适的安装命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. 图片预处理优化**

```python
# 在前端压缩图片后再上传
def compress_image(image, max_size=1920):
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image
```

**3. 使用缓存**

```python
# Streamlit会自动缓存装饰函数
@st.cache_data
def process_image(image):
    # 结果会被缓存，相同输入不会重复计算
    pass
```

---

## 📚 附录

### A. YOLOv8 模型对比

| 模型  | 参数量 | FLOPs | mAP50 | 推理速度（V100） |
| --- | --- | --- | --- | --- |
| YOLOv8n | 3.2M | 8.7 | 37.3 | 0.99 ms |
| YOLOv8s | 11.2M | 28.6 | 44.9 | 1.64 ms |
| YOLOv8m | 25.9M | 78.9 | 50.2 | 2.60 ms |
| YOLOv8l | 43.7M | 165.7 | 52.9 | 3.74 ms |
| YOLOv8x | 68.2M | 257.8 | 53.9 | 6.27 ms |

### B. COCO数据集80类列表

| No. | 类别  | No. | 类别  | No. | 类别  |
| --- | --- | --- | --- | --- | --- |
| 0   | person | 1   | bicycle | 2   | car |
| 3   | motorcycle | 4   | airplane | 5   | bus |
| 6   | train | 7   | truck | 8   | boat |
| 9   | traffic light | 10  | fire hydrant | 11  | stop sign |
| 12  | parking meter | 13  | bench | 14  | bird |
| 15  | cat | 16  | dog | 17  | horse |
| 18  | sheep | 19  | cow | 20  | elephant |
| 21  | bear | 22  | zebra | 23  | giraffe |
| 24  | backpack | 25  | umbrella | 26  | handbag |
| 27  | tie | 28  | suitcase | 29  | frisbee |
| 30  | skis | 31  | snowboard | 32  | sports ball |
| 33  | kite | 34  | baseball bat | 35  | baseball glove |
| 36  | skateboard | 37  | surfboard | 38  | tennis racket |
| 39  | bottle | 40  | wine glass | 41  | cup |
| 42  | fork | 43  | knife | 44  | spoon |
| 45  | bowl | 46  | banana | 47  | apple |
| 48  | sandwich | 49  | orange | 50  | broccoli |
| 51  | carrot | 52  | hot dog | 53  | pizza |
| 54  | donut | 55  | cake | 56  | chair |
| 57  | couch | 58  | potted plant | 59  | bed |
| 60  | dining table | 61  | toilet | 62  | tv  |
| 63  | laptop | 64  | mouse | 65  | remote |
| 66  | keyboard | 67  | cell phone | 68  | microwave |
| 69  | oven | 70  | toaster | 71  | sink |
| 72  | refrigerator | 73  | book | 74  | clock |
| 75  | vase | 76  | scissors | 77  | teddy bear |
| 78  | hair drier | 79  | toothbrush |     |     |

### C. 常用链接

- **YOLO官方文档**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
  
- **FastAPI官方文档**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
  
- **Streamlit官方文档**: [https://docs.streamlit.io/](https://docs.streamlit.io/)
  
- **COCO数据集**: [https://cocodataset.org/](https://cocodataset.org/)
  

---

## 🎉 总结

本指南完整介绍了在Windows系统上搭建YOLO目标检测系统的方法，包括：

✅ YOLOv8模型的原理和架构详解

✅ FastAPI后端的详细实现和代码讲解

✅ Streamlit前端的完整开发指南

✅ 环境配置、系统集成、性能优化

✅ 常见问题排查和解决方案

现在，你可以：

1. 本地运行完整的检测系统
  
2. 根据需求修改模型或界面
  
3. 部署到生产环境
  

**下一步建议：**

- 尝试训练自定义YOLO模型
  
- 添加视频流检测功能
  
- 部署到云服务器
  
- 优化UI/UX体验
  

祝你开发顺利！🚀
