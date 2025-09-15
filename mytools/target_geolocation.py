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
def detect_targets(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    # 模拟返回多个目标，比如 4 个点（四个角中点）
    return [
        (int(w * 0.25), int(h * 0.25)),
        (int(w * 0.75), int(h * 0.25)),
        (int(w * 0.25), int(h * 0.75)),
        (int(w * 0.75), int(h * 0.75)),
        (int(w * 0.5), int(h * 0.5))  # 中心点
    ]


def call_location_service(image_path, remote_tile_dir):
    with open(image_path, "rb") as f:
        files = {"drone_image": (os.path.basename(image_path), f, "image/jpeg")}
        data = {
            "remote_sensing_tile_dir": remote_tile_dir,
            "metadata_option": True,
            "top_k": 1
        }
        response = requests.post("http://localhost:8000/locate", files=files, data=data)
        return response.json()

from pyproj import Transformer


def project_point_to_ortho(eo, R, focal_length, pixel_size, boundary, gsd, ground_height, image_shape, u, v):
    """
    基于小孔成像模型将无人机影像点投影到正射影像坐标
    
    参数:
        eo: 外方位元素 [X0, Y0, Z0]
        R: 旋转矩阵 (3x3)
        focal_length: 焦距 (米)
        pixel_size: 像素大小 (米/像素)
        boundary: 正射影像边界 [min_x, max_x, min_y, max_y]
        gsd: 地面采样距离 (米)
        ground_height: 地面高度 (米)
        image_shape: 无人机影像形状 (height, width)
        u: 无人机影像列坐标
        v: 无人机影像行坐标
        
    返回:
        x_ortho: 正射影像x坐标 (列)
        y_ortho: 正射影像y坐标 (行)
    """
    # 1. 计算内参矩阵 K 的逆
    c_x = image_shape[1] / 2
    c_y = image_shape[0] / 2
    fx = focal_length / pixel_size
    fy = focal_length / pixel_size
    K_inv = np.array([
        [1/fx, 0, -c_x/fx],
        [0, 1/fy, -c_y/fy],
        [0, 0, 1]
    ])
    
    # 2. 计算旋转矩阵的逆
    R_inv = np.linalg.inv(R)
    
    # 3. 将像素坐标转换为归一化相机坐标
    point_uv = np.array([u, v, 1])
    point_cam_norm = K_inv @ point_uv
    
    # 4. 将归一化相机坐标转换到世界坐标系
    # 注意: 这里得到的是从相机中心出发的射线方向
    dir_world = R_inv @ point_cam_norm
    
    # 5. 计算与地面的交点
    # 地面方程: Z = ground_height
    # 相机中心位置: (X0, Y0, Z0)
    # 射线方程: P = eo + lambda * dir_world
    # 解: lambda = (ground_height - eo[2]) / dir_world[2]
    delta_Z = ground_height - eo[2]
    lambda_val = -delta_Z / dir_world[2]  # 注意负号，因为相机Z轴方向
    
    # 6. 计算地面点坐标
    X_ground = eo[0] + lambda_val * dir_world[0]
    Y_ground = eo[1] + lambda_val * dir_world[1]
    
    # 7. 转换为正射影像坐标
    min_x, max_x, min_y, max_y = boundary
    x_ortho = (X_ground - min_x) / gsd
    y_ortho = (max_y - Y_ground) / gsd  # 注意y方向反转
    
    return x_ortho, y_ortho




def pixel_to_geo(pixel_xy, result):
    if result["status"] != "success":
        return None

    geo_coords = []
    for entry in result["results"]:
        H = np.array(entry.get("homography"))
        transform_vals = entry.get("transform")  # [a, b, c, d, e, f]
        if H is None or transform_vals is None:
            geo_coords.append(None)
            continue

        transform = rasterio.Affine(*transform_vals)

        # Step 1: 单应性变换后像素坐标
        if 'info_for_project_point_to_orthoimage' in entry:
            info = entry['info_for_project_point_to_orthoimage']
            R = np.array(info['R']).reshape(3, 3)
            focal_length = info['focal_length']
            pixel_size = info['pixel_size']
            boundary = info['boundary']
            gsd = info['gsd']
            ground_height = info['ground_height']
            image_shape = info['image_shape']
            u, v = pixel_xy
            pixel_xy[0], pixel_xy[1] = project_point_to_ortho(
                entry["eo"], R, focal_length, pixel_size,
                boundary, gsd, ground_height, image_shape, u, v
            )
            
        
        pt = np.array([pixel_xy[0], pixel_xy[1], 1.0]).reshape(3, 1)
        pt_transformed = H @ pt
        pt_transformed /= pt_transformed[2]  # 归一化
        x_proj, y_proj = pt_transformed[0][0], pt_transformed[1][0]

        # Step 2: 像素坐标 → 地图投影坐标系 (米制等)
        map_x, map_y = transform * (x_proj, y_proj)

        # Step 3: 地图坐标系 → 经纬度坐标
        with rasterio.open(entry["ref_image"]) as dataset:
            crs_src = dataset.crs  # 当前影像 CRS
        crs_dst = "EPSG:4326"     # 目标坐标系 WGS84
        transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
        lon, lat = transformer.transform(map_x, map_y)

        geo_coords.append((lon, lat))

    return geo_coords


def create_map_with_points(coords):
    m = leafmap.Map(center=[30, -95], zoom=4)
    m.add_basemap("SATELLITE")

    features = []
    for lon, lat in coords:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "popup": f"Target: ({lon:.6f}, {lat:.6f})"
            }
        })

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    m.add_geojson(geojson_data, layer_name="Detected Targets", style={
        "color": "red",
        "radius": 8,
        "fillColor": "yellow",
        "fillOpacity": 0.7,
        "weight": 2
    })

    lats = [lat for _, lat in coords]
    lons = [lon for lon, _ in coords]
    if lats and lons:
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        m.fit_bounds(bounds)

    html_str = m.to_html()
    html_base64 = base64.b64encode(html_str.encode("utf-8")).decode("utf-8")
    iframe_code = f'<iframe src="data:text/html;base64,{html_base64}" width="100%" height="600px" frameborder="0"></iframe>'
    return iframe_code


def process(drone_image_path, tile_dir):
    # 1. 多目标检测
    targets = detect_targets(drone_image_path)
    
    # 2. 调用定位服务
    location_result = call_location_service(drone_image_path, tile_dir)
    
    # 3. 获取所有目标的地理坐标
    all_geo_coords = []
    for target in targets:
        coords = pixel_to_geo(target, location_result)
        if coords:
            all_geo_coords.extend(coords)

    # 4. 创建地图图层
    map_iframe = create_map_with_points(all_geo_coords)
    
    return json.dumps(location_result, indent=2, ensure_ascii=False), map_iframe


demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="filepath", label="上传无人机图像"),
        gr.Textbox(label="遥感影像目录路径", value="/data/code/120/image-matching-models/datasets/slice/range2")
    ],
    outputs=[
        gr.Textbox(label="定位返回结果 (JSON)"),
        gr.HTML(label="地图展示")
    ],
    title="目标检测与地理定位演示",
    description="上传无人机图像，检测目标，调用定位服务，显示其地理位置"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
