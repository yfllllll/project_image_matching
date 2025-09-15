from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
import os
import shutil
import tempfile
import cv2
import numpy as np
import torch
from matching import get_matcher
from retrieve.Desmodel import Sample4Geo
from poslocation.Coarse_img import SmartImage
from retrieve.utils import calculate_overlap, parse_bbox_from_filename, get_tif_metadata
import rasterio
from rasterio.transform import from_origin
from matching.viz import *
app = FastAPI()

# # 初始化设备、模型和匹配器
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# similarity_retrieve = Sample4Geo(device=device)
ransac_kwargs = {'ransac_reproj_thresh': 2, 'ransac_conf': 0.99, 'ransac_iters': 2000}
matcher = get_matcher(['minima-splg'], device=device, **ransac_kwargs)
rotation_steps = 9
rotation_angle_step = 360 / rotation_steps


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
            
            H = np.arrar([scale, 0, 0, 0, scale, 0, 0, 0, 1]) @ H_final

            return H


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
def preprocess_image(image):
    return matcher.load_image(image)
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
    return warped_src, x_min, y_min

def get_topk_overlap(geo_bbox, tile_dir, top_k=2):
    tile_metadata = []
    for tile_file in os.listdir(tile_dir):
        if tile_file.endswith('.tif'):
            tif_path = os.path.join(tile_dir, tile_file)
            tile_bbox = parse_bbox_from_filename(tile_file)
            if tile_bbox is None:
                tile_bbox = get_tif_metadata(tif_path)
            tile_metadata.append((tile_file, tile_bbox))
    overlap_list = []
    for tile_file, tile_bbox in tile_metadata:
        overlap_area = calculate_overlap(geo_bbox, tile_bbox)
        if overlap_area > 0:
            overlap_list.append((tile_file, overlap_area))
    overlap_list.sort(key=lambda x: x[1], reverse=True)
    return [os.path.join(tile_dir, item[0]) for item in overlap_list[:top_k]]

@app.post("/locate")
async def locate(
    drone_image: UploadFile = File(...),
    remote_sensing_tile_dir: str = Form(...),
    metadata_option: bool = Form(True),
    top_k: int = Form(1)
):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_image_path = os.path.join(tmpdir, drone_image.filename)
        with open(src_image_path, "wb") as f:
            shutil.copyfileobj(drone_image.file, f)

        # camera_img = None
        # if metadata_option:
        #     camera_img = CameraImg(src_image_path)
        #     geo_bbox = camera_img.get_boundry()
        #     geo_bbox = [geo_bbox[0], geo_bbox[2], geo_bbox[1], geo_bbox[3]]
        #     ref_list = get_topk_overlap(geo_bbox, remote_sensing_tile_dir, top_k)
        #     match_dict = {src_image_path: ref_list}
        # else:
        #     match_dict = similarity_retrieve(src_image_path, remote_sensing_tile_dir, top_k=top_k)
        st = SmartImage(file_path=src_image_path, search_model='Sample4Geo', device=device, sensor_width=6.3, ground_height=0, epsg=3857, gsd=0.2)
        match_dict, camera_img = st.do_retrieval_(remote_sensing_tile_dir, top_k=1)
        results = []
        for src_path, ref_paths in match_dict.items():
            for ref_path in ref_paths:
                resize = 1024 if camera_img else None
                isCameraImg = True
                src_image = camera_img.get_coarse_tif() if camera_img else cv2.imread(src_path)
                ref_image = cv2.imread(ref_path)
                # ref_preprocessed = preprocess_image(ref_image)
                # src_preprocessed = preprocess_image(src_image, resize)
                # match_result = matcher(src_preprocessed, ref_preprocessed)
                H = image_matching(src_image, ref_image, isCameraImg=isCameraImg,resize=resize)
                warped_src, x_min, y_min = compute_homography_and_warp_dynamic(src_image, H)
                output_tif_path = os.path.join(tmpdir, f"warped_{os.path.basename(src_path)}.tif")
                transform, crs = save_as_geotiff(warped_src, output_tif_path, ref_path, x_offset=x_min, y_offset=y_min)
                results.append({
                    "src_image": src_path,
                    "ref_image": ref_path,
                    "aligned_image_tif": output_tif_path,
                    "homography": H.tolist(),
                    "transform": list(transform),  # 新增字段
                    "crs_src": crs.to_string() if isinstance(crs, rasterio.crs.CRS) else crs,
                    "info_for_project_point_to_orthoimage": camera_img.info_for_project_point_to_orthoimage if camera_img else None
                })

        return JSONResponse(content={"status": "success", "results": results})
