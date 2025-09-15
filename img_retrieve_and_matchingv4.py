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
from poslocation.Coarse_img import SmartImage
from matching.viz import *
from matching import get_matcher
from rasterio.transform import from_origin
from rasterio.warp import transform
from pyproj import Transformer
# # 初始化设备、模型和匹配器
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# similarity_retrieve = Sample4Geo(device=device)
ransac_kwargs = {'ransac_reproj_thresh': 2, 'ransac_conf': 0.99, 'ransac_iters': 2000}
matcher = get_matcher(['minima-splg'], device=device, **ransac_kwargs)
rotation_steps = 9
rotation_angle_step = 360 / rotation_steps

def preprocess_image(image):
    return matcher.load_image(image)
# 图像匹配函数
def image_matching(img0, img1,isCameraImg=False, resize=1024):
    # try:
        if isCameraImg:
            # img0 = source_image.get_coarse_tif()
            # img1 = cv2.imread(target_image)
            # img1_preprocessed = preprocess_image(img1)
            img0 = preprocess_image(img0)
            img1 = preprocess_image(img1)   
            result = matcher(img0, img1)
            num_inliers, H, inlier_kpts0, inlier_kpts1, matched_kpts0, matched_kpts1 = result['num_inliers'], result['H'], result['inlier_kpts0'], result['inlier_kpts1'],  result['matched_kpts0'], result['matched_kpts1']
            # result = stitch_images_optimized(img0, img1, H)

            # # 如果需要更精细的透明度效果，可以按以下方式进行加权操作：
            # # 计算每个像素的透明度加权
            # result[0:height, 0:width] = np.where(result[0:height, 0:width] != 0, result[0:height, 0:width], 
            #                                     cv2.addWeighted(result[0:height, 0:width], alpha, img1_t[0:height, 0:width], beta, 0))

            return H
            
        else:
        # 读取图像
            # img0 = cv2.imread(source_image)
            # img1 = cv2.imread(target_image)

            # # 检查图像是否成功读取
            # if img0 is None or img1 is None:
            #     raise ValueError("Failed to read one or both of the images.")

            # 预处理图像
            # size = 1024
            # img1_t = cv2.resize(img1, (size, size), interpolation=cv2.INTER_LINEAR)
            img1_preprocessed = preprocess_image(img1)
            # img0_resized = cv2.resize(img0, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)

            # 用于存储当前匹配结果
            scale = max(img0.shape[0], img0.shape[1]) / resize
            img0_resized = cv2.resize(img0, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
            all_mkpts0 = []
            all_mkpts1 = []
            all_inlier_kpts0 = []
            all_inlier_kpts1 = []
            
            # 旋转并进行匹配
            for rot in range(rotation_steps):
                # 旋转角度
                angle = rot * rotation_angle_step

                # 使用用户提供的旋转函数
                img0_rotated, M = rotate_image_bound_with_M(img0_resized, angle)
                M = np.vstack((M, np.array([0, 0, 1])))
                M_inv = np.asmatrix(np.linalg.inv(M))

                # 进行匹配
                result = matcher(matcher.load_image(img0_rotated), img1_preprocessed)

                # 提取匹配结果中的内点数和匹配点
                num_inliers, H, inlier_kpts0, inlier_kpts1, matched_kpts0, matched_kpts1 = result['num_inliers'], result['H'], result['inlier_kpts0'], result['inlier_kpts1'],  result['matched_kpts0'], result['matched_kpts1']
                # 还原旋转后的匹配点（将旋转后的点恢复到原图像坐标系）
                inlier_kpts0 = add_ones(inlier_kpts0)
                inlier_kpts0 = (M_inv * inlier_kpts0.T).A.T[:, 0:2]  # 恢复到原图坐标系
                
                matched_kpts0 = add_ones(matched_kpts0)  # 将关键点转换为齐次坐标
                matched_kpts0 = (M_inv * matched_kpts0.T).A.T[:, 0:2]  # 恢复到原图坐标系

                # 将还原的匹配点存储
                all_inlier_kpts0.append(inlier_kpts0)
                all_mkpts0.append(matched_kpts0)
                all_mkpts1.append(matched_kpts1)
                all_inlier_kpts1.append(inlier_kpts1)

            # 将所有还原后的匹配点合并，计算最终的变换矩阵
            all_mkpts0 = np.vstack(all_mkpts0)
            all_mkpts1 = np.vstack(all_mkpts1)
            all_inlier_kpts0 = np.vstack(all_inlier_kpts0)
            all_inlier_kpts1 = np.vstack(all_inlier_kpts1)

            # 计算最终的变换矩阵（使用透视变换）
            H_final, status = cv2.findHomography(all_inlier_kpts0, all_inlier_kpts1, 
                                                method = cv2.USAC_MAGSAC,
                                                ransacReprojThreshold = ransac_kwargs['ransac_reproj_thresh'], 
                                                confidence = ransac_kwargs['ransac_conf'],
                                                maxIters = ransac_kwargs['ransac_iters'])

            # result = stitch_images_optimized(img0, img1, H_final)
            #对 img0 进行透视变换
            
            S1 = np.array([[1/scale, 0, 0],
              [0, 1/scale, 0],
              [0, 0, 1]])  # 3x3标准缩放矩阵
            # S1_inv = np.linalg.inv(S1)
            H = H_final @ S1
            
            return H

# 目标检测函数（新增可视化功能）
def detect_targets(image_path, visualize=False):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    targets = [
        (int(w * 0.25), int(h * 0.25)),
        (int(w * 0.75), int(h * 0.25)),
        (int(w * 0.25), int(h * 0.75)),
        (int(w * 0.75), int(h * 0.75)),
        (int(w * 0.5), int(h * 0.5))
    ]
    
    # 在图像上绘制目标点[3,9](@ref)
    if visualize:
        for i, pt in enumerate(targets):
            # 绘制圆形标记点
            cv2.circle(img, pt, radius=12, color=(0, 0, 255), thickness=-1)
            # 添加序号标签[3](@ref)
            cv2.putText(img, str(i+1), (pt[0]-5, pt[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 转换颜色空间（BGR→RGB）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return targets, img_rgb
    
    return targets

# def call_location_service(image_path, remote_tile_dir):
#     with open(image_path, "rb") as f:
#         files = {"drone_image": (os.path.basename(image_path), f, "image/jpeg")}
#         data = {
#             "remote_sensing_tile_dir": remote_tile_dir,
#             "metadata_option": True,
#             "top_k": 1
#         }
#         response = requests.post("http://localhost:8000/locate", files=files, data=data)
#         return response.json()

def save_as_geotiff(image_array, save_path, ref_tif_path=None, x_offset=0, y_offset=0):
    height, width, channels = image_array.shape

    if ref_tif_path:
        with rasterio.open(ref_tif_path) as ref:
            base_transform = ref.transform
            crs = ref.crs
            xres = base_transform.a
            yres = base_transform.e
            transform = rasterio.Affine(
                xres, 0, base_transform.c + x_offset * xres,
                0, yres, base_transform.f + y_offset * yres
            )
    else:
        crs = rasterio.crs.CRS.from_epsg(4326)
        transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        save_path, 'w', driver='GTiff',
        height=height, width=width,
        count=channels, dtype=image_array.dtype,
        crs=crs, transform=transform
    ) as dst:
        for i in range(channels):
            dst.write(image_array[:, :, i], i + 1)
    return transform, crs

# from pyproj import Transformer
def compute_homography_and_warp_dynamic(src_image, H_final):
    h0, w0 = src_image.shape[:2]
    corners_src = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_src, H_final)
    all_corners = transformed_corners.reshape(-1, 2)
    x_min, y_min = np.int32(all_corners.min(axis=0))
    x_max, y_max = np.int32(all_corners.max(axis=0))
    new_width = x_max - x_min
    new_height = y_max - y_min
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    H_translated = translation_matrix @ H_final
    warped_src = cv2.warpPerspective(src_image, H_translated, (new_width, new_height))
    return warped_src, x_min, y_min, H_translated
def locate(
    drone_image_path,
    remote_tile_dir,
    top_k=1
):
    st = SmartImage(file_path=drone_image_path, search_model='Sample4Geo', device=device, sensor_width=6.3, ground_height=0, epsg=3857, gsd=0.2)
    match_dict, camera_img = st.do_retrieval_(remote_tile_dir, top_k=top_k)
    results = []
    for src_path, ref_paths in match_dict.items():
        for ref_path in ref_paths:
            resize = 1024 if not camera_img else None
            isCameraImg = True if camera_img else False
            # 读取源图像和参考图像
            src_image = camera_img.get_coarse_tif() if camera_img else cv2.imread(src_path)
            ref_image = cv2.imread(ref_path)
            # ref_preprocessed = preprocess_image(ref_image)
            # src_preprocessed = preprocess_image(src_image, resize)
            # match_result = matcher(src_preprocessed, ref_preprocessed)
            H = image_matching(src_image, ref_image, isCameraImg=isCameraImg,resize=resize)
            warped_src, x_min, y_min, H = compute_homography_and_warp_dynamic(src_image, H)
            output_tif_path = os.path.join('tmpdir', f"warped_{os.path.basename(src_path)}.tif")
            transform, crs = save_as_geotiff(warped_src, output_tif_path, ref_path, x_offset=x_min, y_offset=y_min)
            results.append({
                "src_image": src_path,
                "ref_image": ref_path,
                "aligned_image_tif": output_tif_path,
                "homography": H,
                "transform": transform,  # 新增字段
                "crs_src": crs.to_string() if isinstance(crs, rasterio.crs.CRS) else crs,
                "info_for_project_point_to_orthoimage": camera_img.info_for_project_point_to_orthoimage if camera_img else None
            })

    return results

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
    c_x = image_shape[1] / 2
    c_y = image_shape[0] / 2
    # fx = 1 / pixel_size
    # fy = 1 / pixel_size
    
    # === 修正开始 ===
    # 计算去中心化坐标
    u_norm = (u - c_x)*pixel_size 
    v_norm = (v - c_y)*pixel_size  
    
    # 构造正确的归一化方向向量（含焦距因子）
    point_cam_norm = np.array([u_norm, v_norm, -focal_length]) 
    # === 修正结束 ===
    
    # 2. 旋转到世界坐标系
    
    coord_GCS = R.transpose() @ point_cam_norm
    
    # 3. 计算与地面的交点
    scale = (ground_height - eo[2]) / coord_GCS[2]
    plane_coord_GCS = scale * coord_GCS[0:2] 
    
    
    X_ground = plane_coord_GCS[0] + eo[0]
    Y_ground = plane_coord_GCS[1] + eo[1]
    
    # 4. 转换为正射影像坐标
    min_x, max_x, min_y, max_y = boundary
    x_ortho = (X_ground - min_x) / gsd
    y_ortho = (max_y - Y_ground) / gsd  
    
    return x_ortho[0], y_ortho[0]



def pixel_to_geo(pixel_xy, result):
    # if result["status"] != "success":
    #     return None

    geo_coords = []
    for entry in result:
        H = np.array(entry.get("homography"))
        transform_vals = entry.get("transform")  # [a, b, c, d, e, f]
        if H is None or transform_vals is None:
            geo_coords.append(None)
            continue

        transform = rasterio.Affine(*transform_vals)

        # Step 1: 单应性变换后像素坐标
        if entry.get('info_for_project_point_to_orthoimage',None ) is not None: 
            info = entry['info_for_project_point_to_orthoimage']
            R = np.array(info['R']).reshape(3, 3)
            focal_length = info['focal_length']
            pixel_size = info['pixel_size']
            boundary = info['boundary']
            gsd = info['gsd']
            ground_height = info['ground_height']
            image_shape = info['image_shape']
            u, v = pixel_xy
            eo = info['eo'][:3]
            new_x, new_y = project_point_to_ortho(
                eo, R, focal_length, pixel_size,
                boundary, gsd, ground_height, image_shape, u, v
            )
            pixel_xy = [new_x, new_y]
            
        
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
    targets, annotated_img = detect_targets(drone_image_path, visualize=True)
    
    # 2. 调用定位服务
    location_result = locate(drone_image_path, tile_dir)
    
    # 3. 获取所有目标的地理坐标
    all_geo_coords = []
    for target in targets:
        coords = pixel_to_geo(target, location_result)
        if coords:
            all_geo_coords.extend(coords)

    # 4. 创建地图图层
    print(all_geo_coords)
    map_iframe = create_map_with_points(all_geo_coords)
    
    return annotated_img, map_iframe


demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="filepath", label="上传无人机图像"),
        gr.Textbox(label="遥感影像目录路径", value="/data/code/120/image-matching-models/datasets/slice/range2")
    ],
    outputs=[
        # gr.Textbox(label="定位返回结果 (JSON)"),
        gr.Image(label="目标点可视化", type="numpy"),  # 显示标注后的图像
        gr.HTML(label="地图展示")
    ],
    title="目标检测与地理定位演示",
    description="上传无人机图像，检测目标，调用定位服务，显示其地理位置"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
