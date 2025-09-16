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
import time  
from typing import Optional, Dict, Any  
from requests.adapters import HTTPAdapter  
from urllib3.util.retry import Retry  
# 更新后的geolocate函数
class RobustGeoLocationClient:  
    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 300):  
        self.base_url = base_url  
        self.timeout = timeout  
        self.session = self._create_session()  
      
    def _create_session(self) -> requests.Session:  
        """创建带重试机制的会话"""  
        session = requests.Session()  
          
        # 配置重试策略  
        retry_strategy = Retry(  
            total=3,  
            backoff_factor=1,  
            status_forcelist=[429, 500, 502, 503, 504],  
            allowed_methods=["POST"]  
        )  
          
        adapter = HTTPAdapter(max_retries=retry_strategy)  
        session.mount("http://", adapter)  
        session.mount("https://", adapter)  
          
        return session  
      
    def geolocate(self, image_path: str, tile_dir: str, points: Optional[list] = None) -> Dict[str, Any]:  
        """增强的地理定位函数"""  
        if not os.path.exists(image_path):  
            return {"error": f"图像文件不存在: {image_path}"}  
          
        if not os.path.exists(tile_dir):  
            return {"error": f"遥感影像目录不存在: {tile_dir}"}  
          
        try:  
            with open(image_path, "rb") as f:  
                # 准备请求数据  
                data = {  
                    "tile_dir": tile_dir,  
                    "top_k": "1"  
                }  
                  
                if points:  
                    # 验证点坐标格式  
                    if not isinstance(points, list):  
                        return {"error": "点坐标必须是列表格式"}  
                      
                    for i, point in enumerate(points):  
                        if not isinstance(point, (list, tuple)) or len(point) != 2:  
                            return {"error": f"第{i+1}个点坐标格式错误"}  
                      
                    data["points"] = json.dumps(points)  
                  
                files = {"drone_image": (os.path.basename(image_path), f, "image/jpeg")}  
                  
                print(f"发送地理定位请求到 {self.base_url}/geolocate")  
                print(f"图像文件: {image_path}")  
                print(f"遥感目录: {tile_dir}")  
                if points:  
                    print(f"点坐标: {len(points)}个")  
                  
                # 发送请求  
                start_time = time.time()  
                response = self.session.post(  
                    f"{self.base_url}/geolocate",  
                    files=files,  
                    data=data,  
                    timeout=self.timeout  
                )  
                  
                elapsed_time = time.time() - start_time  
                print(f"请求完成，耗时: {elapsed_time:.2f}秒")  
                  
                # 处理响应  
                if response.status_code == 200:  
                    result = response.json()  
                    print("地理定位成功")  
                    return result  
                else:  
                    try:  
                        error_detail = response.json().get("detail", {})  
                        error_msg = error_detail.get("message", f"服务错误: {response.status_code}")  
                        error_type = error_detail.get("error_type", "unknown")  
                        request_id = error_detail.get("request_id", "unknown")  
                        print(f"请求失败 [{request_id}]: {error_msg} (类型: {error_type})")  
                        return {"error": error_msg, "error_type": error_type, "request_id": request_id}  
                    except:  
                        return {"error": f"HTTP错误: {response.status_code}"}  
          
        except requests.exceptions.Timeout:  
            return {"error": f"请求超时 (>{self.timeout}秒)"}  
        except requests.exceptions.ConnectionError:  
            return {"error": f"无法连接到服务器: {self.base_url}"}  
        except requests.exceptions.RequestException as e:  
            return {"error": f"网络请求失败: {str(e)}"}  
        except Exception as e:  
            return {"error": f"客户端错误: {str(e)}"}  
# 创建全局客户端实例  
geo_client = RobustGeoLocationClient()
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
    """增强的坐标转换函数"""  
    try:  
        if not location_params:  
            return {"error": "缺少定位参数"}  
          
        if not points:  
            return {"error": "缺少点坐标"}  
          
        # 验证点坐标格式  
        for i, point in enumerate(points):  
            if not isinstance(point, (list, tuple)) or len(point) != 2:  
                return {"error": f"第{i+1}个点坐标格式错误"}  
          
        payload = {  
            "location_params": location_params,  
            "points": points  
        }  
          
        print(f"发送坐标转换请求，包含{len(points)}个点")  
          
        response = requests.post(  
            f"http://localhost:7860/transform",  
            json=payload,  
            timeout=60  
        )  
          
        if response.status_code == 200:  
            result = response.json()  
            print("坐标转换成功")  
            return result  
        else:  
            try:  
                error_detail = response.json().get("detail", {})  
                error_msg = error_detail.get("message", f"转换服务错误: {response.status_code}")  
                return {"error": error_msg}  
            except:  
                return {"error": f"坐标转换HTTP错误: {response.status_code}"}  
                  
    except requests.exceptions.Timeout:  
        return {"error": "坐标转换请求超时"}  
    except requests.exceptions.ConnectionError:  
        return {"error": "无法连接到转换服务"}  
    except Exception as e:  
        return {"error": f"坐标转换失败: {str(e)}"}

def create_map(coords):  
    """增强的地图创建函数"""  
    try:  
        if not coords or "results" not in coords:  
            return "<p>无有效坐标数据</p>"  
          
        results = coords["results"]  
        if not results:  
            return "<p>坐标转换结果为空</p>"  
          
        m = leafmap.Map(center=[30, -95], zoom=4)  
        m.add_basemap("SATELLITE")  
          
        points = []  
        valid_points = 0  
          
        for point_name, result in results.items():  
            try:  
                if isinstance(result, dict) and "wgs84" in result:  
                    wgs84_coords = result["wgs84"]  
                    if isinstance(wgs84_coords, list) and len(wgs84_coords) > 0:  
                        lon, lat = wgs84_coords[0]  
                        if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):  
                            points.append((lon, lat))  
                            m.add_marker(  
                                location=[lat, lon],  
                                popup=f"目标点 {point_name}: ({lon:.6f}, {lat:.6f})"  
                            )  
                            valid_points += 1  
            except Exception as e:  
                print(f"处理点 {point_name} 时出错: {e}")  
                continue  
          
        if valid_points == 0:  
            return "<p>没有有效的坐标点可以显示</p>"  
          
        # 调整地图视野  
        if points:  
            lats = [lat for _, lat in points]  
            lons = [lon for lon, _ in points]  
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]  
            m.fit_bounds(bounds)  
          
        # 生成HTML  
        html_str = m.to_html()  
        html_base64 = base64.b64encode(html_str.encode("utf-8")).decode("utf-8")  
        iframe_code = f'<iframe src="data:text/html;base64,{html_base64}" width="100%" height="600px" frameborder="0"></iframe>'  
          
        print(f"地图创建成功，包含{valid_points}个有效点")  
        return iframe_code  
          
    except Exception as e:  
        error_msg = f"地图创建失败: {str(e)}"  
        print(error_msg)  
        return f"<p>{error_msg}</p>"

def full_process(drone_image_path, tile_dir):  
    """增强的完整处理流程"""  
    if not drone_image_path:  
        return None, "请上传无人机图像", ""  
      
    if not tile_dir or not os.path.exists(tile_dir):  
        return None, f"遥感影像目录不存在: {tile_dir}", ""  
      
    try:  
        # 1. 创建临时目录  
        os.makedirs("tmp", exist_ok=True)  
          
        # 2. 目标检测  
        print("开始目标检测...")  
        points = detect_targets(drone_image_path)  
        print(f"检测到{len(points)}个目标点")  
          
        # 3. 可视化目标点  
        vis_img = visualize_targets(drone_image_path, points)  
        vis_path = "tmp/visualized.jpg"  
        cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))  
          
        # 4. 地理定位（使用增强的客户端）  
        print("开始地理定位...")  
        geolocate_result = geo_client.geolocate(drone_image_path, tile_dir, points)  
          
        if "error" in geolocate_result:  
            error_msg = geolocate_result["error"]  
            error_type = geolocate_result.get("error_type", "unknown")  
            request_id = geolocate_result.get("request_id", "")  
            full_error = f"地理定位失败: {error_msg}"  
            if request_id:  
                full_error += f" [请求ID: {request_id}]"  
            return vis_path, full_error, ""  
          
        location_params = geolocate_result.get("location_params", [])  
        if not location_params:  
            return vis_path, "地理定位成功但未返回定位参数", ""  
          
        # 5. 坐标转换  
        map_html = ""  
        if points and location_params:  
            print("开始坐标转换...")  
            transform_result = transform_coords(location_params, points)  
              
            if "error" in transform_result:  
                print(f"坐标转换失败: {transform_result['error']}")  
                geolocate_result["transform_error"] = transform_result["error"]  
            else:  
                print("坐标转换成功，生成地图...")  
                map_html = create_map(transform_result)  
        else:  
            map_html = "<p>无法执行坐标转换（缺少点或位置参数）</p>"  
          
        # 6. 格式化结果  
        formatted_result = json.dumps(geolocate_result, indent=2, ensure_ascii=False)  
          
        print("处理完成")  
        return vis_path, formatted_result, map_html  
          
    except Exception as e:  
        error_msg = f"处理过程中发生错误: {str(e)}"  
        print(error_msg)  
        return None, error_msg, ""

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