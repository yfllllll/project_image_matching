import os
import tempfile
import json
import numpy as np
import cv2
import rasterio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pyproj import Transformer
from rasterio.transform import from_origin
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import torch
import logging
import traceback
from poslocation.Coarse_img import SmartImage
from matching.viz import *
from matching import get_matcher
# 初始化服务
app = FastAPI(
    title="地理定位服务 API",
    description="无人机图像地理定位服务，包含定位参数获取和坐标转换功能",
    version="1.3.0"
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化设备、模型和匹配器
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ransac_kwargs = {'ransac_reproj_thresh': 2, 'ransac_conf': 0.99, 'ransac_iters': 2000}
matcher = get_matcher(['minima-splg'], device=device, **ransac_kwargs)
rotation_steps = 9
rotation_angle_step = 360 / rotation_steps

# 定义数据模型
class LocationParam(BaseModel):
    """单个定位参数模型"""
    ref_image: str
    homography: List[List[float]]
    transform: List[float]  # [a, b, c, d, e, f]
    crs_src: str
    info_for_project_point_to_orthoimage: Optional[Dict] = None

class GeoLocationRequest(BaseModel):
    """地理定位请求模型"""
    tile_dir: str
    points: Optional[List[Tuple[int, int]]] = None
    top_k: int = 1

class GeoLocationResponse(BaseModel):
    """地理定位响应模型"""
    status: str
    location_params: List[LocationParam]
    transformed_points: Optional[Dict] = None
    metadata: dict

class TransformRequest(BaseModel):
    location_params: list  # 客户端传递的List[Any]
    points: List[Tuple[int, int]]  # 客户端格式: [[x1,y1], [x2,y2]]
    output_crs: str = "both"

class TransformResponse(BaseModel):
    """坐标转换响应模型"""
    status: str
    results: Dict[Tuple[int, int], Dict[str, Any]]
    metadata: dict

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


def convert_numpy_to_list(obj, special_keys=None):
    """
    递归将包含NumPy类型的数据转换为Python原生类型，确保JSON可序列化。
    :param special_keys: 需要特殊处理的字典键（如["eo", "R"]），默认为None
    """
    # 处理NumPy数组
    if obj is None:
        return None
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # 处理NumPy标量
    elif isinstance(obj, np.generic):
        return obj.item()
    
    # 处理字典
    elif isinstance(obj, dict):
        new_dict = {}
        special_keys = special_keys or []  # 默认空列表
        for key, value in obj.items():
            # 特殊字段优先处理（若指定）
            if key in special_keys and isinstance(value, np.ndarray):
                new_dict[key] = value.tolist()
            else:
                new_dict[key] = convert_numpy_to_list(value, special_keys)
        return new_dict
    
    # 处理列表、元组、集合
    elif isinstance(obj, (list, tuple, set)):
        return type(obj)(convert_numpy_to_list(item, special_keys) for item in obj)
    
    # 其他类型直接返回（可扩展异常处理）
    else:
        return obj
# ====== 服务端点 ======

@app.post("/geolocate", response_model=GeoLocationResponse)
async def geolocate_endpoint(
    drone_image: UploadFile = File(..., description="无人机图像文件"),
    tile_dir: str = Form(..., description="遥感影像目录"),
    points: str = Form(None, description="JSON格式的点坐标列表"),
    top_k: int = Form(1, description="返回前K个匹配结果")
):
    """
    地理定位服务端点
    
    使用场景: 
    用户有无人机影像、参考遥感影像数据集和若干像素坐标点，
    希望一次性完成定位并将像素坐标转换为地理坐标
    
    返回:
    - location_params: 定位参数列表
    - transformed_points: 转换后的坐标点（如果提供了点坐标）
    """
    try:
        # 读取无人机图像
        image_data = await drone_image.read()
        logger.info(f"收到地理定位请求，图像大小: {len(image_data)}字节")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_data)
            drone_image_path = tmp.name
        
        # 解析点坐标
        point_list = []
        if points:
            try:
                point_list = json.loads(points)
                if not isinstance(point_list, list) or not all(len(p) == 2 for p in point_list):
                    raise ValueError("点坐标格式无效")
                logger.info(f"解析到 {len(point_list)} 个点坐标")
            except Exception as e:
                logger.warning(f"点坐标解析失败: {str(e)}")
                point_list = []
        
        # 核心定位逻辑 - 保持您的原始实现
        location_results = locate(drone_image_path, tile_dir, top_k)
        
        # 处理定位结果
        location_params = []
        for result in location_results:
            homography = result["homography"].tolist() if isinstance(result["homography"], np.ndarray) else result["homography"]
            transform = [
                result["transform"].a, result["transform"].b, 
                result["transform"].c, result["transform"].d,
                result["transform"].e, result["transform"].f
            ]
            info_for_project_point_to_orthoimage = convert_numpy_to_list(result.get("info_for_project_point_to_orthoimage"))
            param = LocationParam(
                ref_image=result["ref_image"],
                homography=homography,
                transform=transform,
                crs_src=result["crs_src"],
                info_for_project_point_to_orthoimage=info_for_project_point_to_orthoimage
            )
            location_params.append(param)
        
        # 准备响应数据
        response_data = {
            "status": "success",
            "location_params": location_params,
            "metadata": {
                "filename": drone_image.filename,
                "size": len(image_data),
                "top_k": top_k,
                "tile_dir": tile_dir
            }
        }
        
        # 如果提供了点坐标，直接转换并返回
        if point_list:
            logger.info(f"请求中包含 {len(point_list)} 个点坐标，直接转换")
            transformed = await transform_points(location_params, point_list)
            response_data["transformed_points"] = transformed
        
        # 清理临时文件
        if os.path.exists(drone_image_path):
            os.unlink(drone_image_path)
            
        return response_data
    
    except Exception as e:
        logger.error(f"地理定位失败: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={"status": "error", "message": f"地理定位失败: {str(e)}"}
        )

# 修复 TransformRequest 模型定义
class TransformRequest(BaseModel):
    location_params: List[LocationParam]  # 修正这里缺少逗号的错误
    points: List[Tuple[int, int]]
    output_crs: str = "both"  # "wgs84", "utm", "both"

# 修改 transform_endpoint 函数
@app.post("/transform", response_model=TransformResponse)
async def transform_endpoint(request: TransformRequest):
    try:
        logger.info(f"收到坐标转换请求，包含 {len(request.location_params)} 个定位参数")
        
        # 转换点坐标
        transformed_points = await transform_points(request.location_params, request.points)
        
        logger.info(f"成功转换 {len(request.points)} 个点坐标")
        
        return {
            "status": "success",
            "results": transformed_points,
            "metadata": {
                "points_count": len(request.points),
                "output_crs": request.output_crs,
                "location_params_count": len(request.location_params)
            }
        }
    
    except Exception as e:
        logger.error(f"坐标转换失败: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": f"坐标转换失败: {str(e)}",
                "traceback": traceback.format_exc()
            }
        )

# ====== 辅助函数 ======

async def transform_points(
    location_params: List[LocationParam], 
    points: List[Tuple[int, int]], 
) -> Dict:
    """批量转换点坐标"""
    results = {}
    
    # 使用第一个定位参数作为默认
    main_param = location_params[0] if location_params else None
    
    if not main_param:
        return {}
    
    for point in points:
        point_tuple = tuple(point)
        try:
            # 实际应用中应使用pixel_to_geo函数
            wgs84_coords, projected_points= pixel_to_geo(point, main_param)
            
            # 实际应用中应使用transform_to_utm函数
            result = {
                "wgs84": wgs84_coords,
            }
            
            
            result["projected_points"] = projected_points
            
            results[point_tuple] = result
        except Exception as e:
            results[point_tuple] = {"error": str(e)}
    
    return results

# ====== 核心功能函数 ======

def locate(drone_image_path, remote_tile_dir, top_k=1):
    """定位函数实现"""
    st = SmartImage(file_path=drone_image_path, search_model='Sample4Geo', device=device, sensor_width=6.3, ground_height=0, epsg=3857, gsd=0.2)
    match_dict, camera_img = st.do_retrieval_(remote_tile_dir, top_k=top_k)
    results = []
    for src_path, ref_paths in match_dict.items():
        for ref_path in ref_paths:
            resize = 512 if not camera_img else None
            isCameraImg = True if camera_img else False
            # 读取源图像和参考图像
            src_image = camera_img.get_coarse_tif() if camera_img else cv2.imread(src_path)
            ref_image = cv2.imread(ref_path)
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
    u_norm = (u - c_x)*pixel_size 
    v_norm = (v - c_y)*pixel_size  
    
    point_cam_norm = np.array([u_norm, v_norm, -focal_length]) 
    
    coord_GCS = R.transpose() @ point_cam_norm
    
    scale = (ground_height - eo[2]) / coord_GCS[2]
    plane_coord_GCS = scale * coord_GCS[0:2] 
    
    X_ground = plane_coord_GCS[0] + eo[0]
    Y_ground = plane_coord_GCS[1] + eo[1]
    
    min_x, max_x, min_y, max_y = boundary
    x_ortho = (X_ground - min_x) / gsd
    y_ortho = (max_y - Y_ground) / gsd  
    
    return x_ortho[0], y_ortho[0]

def pixel_to_geo(pixel_xy, entry):
    """坐标转换实现"""
    geo_coords = []
    projected_points = []
    
    H = np.array(entry.homography)
    transform_vals = entry.transform  # [a, b, c, d, e, f]
    if H is None or transform_vals is None:
        geo_coords.append(None)
        projected_points.append(None)
        return geo_coords, projected_points

    transform = rasterio.Affine(*transform_vals)

    # Step 1: 单应性变换后像素坐标
    if entry.info_for_project_point_to_orthoimage is not None: 
        info = entry.info_for_project_point_to_orthoimage
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
    projected_points.append((map_x, map_y))
    # Step 3: 地图坐标系 → 经纬度坐标
    with rasterio.open(entry.ref_image) as dataset:
        crs_src = dataset.crs  # 当前影像 CRS
    crs_dst = "EPSG:4326"     # 目标坐标系 WGS84
    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    lon, lat = transformer.transform(map_x, map_y)

    geo_coords.append((lon, lat))

    return geo_coords, projected_points



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)