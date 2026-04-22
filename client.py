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