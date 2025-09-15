import os
import base64
import tempfile
import requests
import json
from PIL import Image
import cv2
import gradio as gr
import leafmap.foliumap as leafmap
import numpy as np
import rasterio
from pyproj import Transformer

# 更新后的geolocate函数
def geolocate(image_path, tile_dir, points):
    with open(image_path, "rb") as f:
        try:
            # 准备表单数据
            data = {
                "tile_dir": tile_dir,
                "top_k": "1"
            }
            
            # 如果有点坐标，转换为JSON字符串
            if points:
                data["points"] = json.dumps(points)
            
            # 准备文件
            files = {"drone_image": (os.path.basename(image_path), f, "image/jpeg")}
            
            # 发送请求
            response = requests.post(
                "http://localhost:7860/geolocate",
                files=files,
                data=data,
                timeout=120
            )
            
            # 检查响应状态
            if response.status_code != 200:
                error_detail = response.json().get("detail", {})
                error_msg = error_detail.get("message", f"服务错误: {response.status_code}")
                return {"error": error_msg}
                
            return response.json()
        except Exception as e:
            return {"error": f"API请求失败: {str(e)}"}

# 其余函数保持不变（保持原样）
def detect_targets(image_path):
    img = Image.open(image_path)
    w, h = img.size
    return [
        (int(w * 0.25), int(h * 0.25)),
        (int(w * 0.75), int(h * 0.25)),
        (int(w * 0.25), int(h * 0.75)),
        (int(w * 0.75), int(h * 0.75)),
        (int(w * 0.5), int(h * 0.5))
    ]

def visualize_targets(image_path, points):
    img = cv2.imread(image_path)
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), radius=12, color=(0, 0, 255), thickness=-1)
        cv2.putText(img, str(i+1), (x-5, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def transform_coords(location_params, points):
    try:
        payload = {
            "location_params": location_params,
            "points": points
        }
        response = requests.post(
            f"http://localhost:7860/transform",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"坐标转换失败: {str(e)}"}

def create_map(coords):
    if not coords or "results" not in coords:
        return "<p>无有效坐标数据</p>"
    
    m = leafmap.Map(center=[30, -95], zoom=4)
    m.add_basemap("SATELLITE")
    
    points = []
    for point, result in coords["results"].items():
        if isinstance(result, dict) and "wgs84" in result:
            lon, lat = result["wgs84"][0]
            points.append((lon, lat))
            m.add_marker(
                location=[lat, lon],
                popup=f"目标点: ({lon:.6f}, {lat:.6f})"
            )
    
    if points:
        lats = [lat for _, lat in points]
        lons = [lon for lon, _ in points]
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        m.fit_bounds(bounds)
        html_str = m.to_html()
        html_base64 = base64.b64encode(html_str.encode("utf-8")).decode("utf-8")
        iframe_code = f'<iframe src="data:text/html;base64,{html_base64}" width="100%" height="600px" frameborder="0"></iframe>'
    return iframe_code

def full_process(drone_image_path, tile_dir):
    os.makedirs("tmp", exist_ok=True)
    points = detect_targets(drone_image_path)
    vis_img = visualize_targets(drone_image_path, points)
    vis_path = "tmp/visualized.jpg"
    cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    # 关键修正：使用新版的geolocate函数
    geolocate_result = geolocate(drone_image_path, tile_dir, points)
    
    if "error" in geolocate_result:
        return vis_path, f"错误: {geolocate_result['error']}", ""
    
    location_params = geolocate_result.get("location_params", [])
    
    # 修正点格式转换
    # formatted_points = [{"x": x, "y": y} for x, y in points]
    
    if points and location_params:
        transform_result = transform_coords(location_params, points)
        map_html = create_map(transform_result)
    else:
        map_html = "<p>无法执行坐标转换（缺少点或位置参数）</p>"
    
    formatted_result = json.dumps(geolocate_result, indent=2)
    return vis_path, formatted_result, map_html

# Gradio界面（保持不变）
with gr.Blocks() as demo:
    gr.Markdown("## 🛰️ 无人机图像地理定位调试工具")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="无人机图像")
            tile_dir = gr.Textbox(
                label="遥感影像目录路径", 
                value="/data/code/120/image-matching-models/datasets/slice/range2"
            )
            submit_btn = gr.Button("开始处理")
        with gr.Column():
            vis_output = gr.Image(label="目标点可视化")
            json_output = gr.JSON(label="定位结果（JSON）")
            map_output = gr.HTML(label="地图结果")
    
    submit_btn.click(
        fn=full_process,
        inputs=[image_input, tile_dir],
        outputs=[vis_output, json_output, map_output]
    )
    
    gr.Markdown("### 使用说明")
    gr.Markdown("""
    1. 上传无人机拍摄的JPG图像
    2. 输入包含参考遥感图像的目录路径
    3. 点击"开始处理"按钮执行地理定位
    4. 查看可视化结果、定位参数和地图
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)