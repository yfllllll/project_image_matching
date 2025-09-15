import os
import re
import rasterio

def calculate_overlap(box1, box2):
    """
    计算两个边界框（bbox）的重叠面积（overlap）
    box1 和 box2 格式：[xmin, xmax, ymin, ymax]
    """
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    overlap_area = inter_width * inter_height
    
    return overlap_area



def parse_bbox_from_filename(filename):
    """
    从文件名解析地理范围（bounding box）。
    处理类似：
    spatial_res_0.4999821350163144_0.5000013895781142_x_13120110.093615696_y_4767525.905654144_x_13120200.0904_y_4767013.904231216.tif
    """
    # 限制数字后面不能跟字母，避免误匹配 `.tif`
    pattern = r'_x_([\d]+\.\d+)_y_([\d]+\.\d+)_x_([\d]+\.\d+)_y_([\d]+\.\d+)(?:\.tif)?$'
    
    match = re.search(pattern, filename)

    if match:
        matched_groups = match.groups()
        print(f"🔍 匹配成功: {matched_groups}")  # 调试输出

        try:
            bbox = [float(coord) for coord in matched_groups]
            print(f"✅ 解析后的 bbox: {bbox}")  # 打印最终解析出的数值
            return bbox
        except ValueError as e:
            print(f"⚠️ 解析错误: {filename}, 错误: {e}")
            return None
    else:
        print(f"⚠️ 无法匹配 bbox 格式: {filename}")
        return None  # 解析失败，返回 None


def get_tif_metadata(tif_path):
    """
    读取 GeoTIFF 影像的地理范围（bounding box）
    返回格式: [xmin, xmax, ymin, ymax]
    """
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # 获取影像的地理范围
        return [bounds.left, bounds.right, bounds.bottom, bounds.top]

def get_topk_overlap(geo_bbox, remote_sensing_tile_dir, top_k=2):
    """
    计算 geo_bbox 与 remote_sensing_tile_dir 中所有 TIF 影像的 overlap，并返回 overlap 最高的 top_k 个影像路径
    """
    tile_metadata = []
    
    # 遍历所有 TIF 影像，获取地理边界信息
    for tile_file in os.listdir(remote_sensing_tile_dir):
        if tile_file.endswith('.tif'):  # 仅处理 TIF 影像
            tif_path = os.path.join(remote_sensing_tile_dir, tile_file)
            
            # 尝试从文件名解析 bbox
            tile_bbox = parse_bbox_from_filename(tile_file)
            
            # 如果文件名无法解析，则读取 TIF 数据
            if tile_bbox is None:
                tile_bbox = get_tif_metadata(tif_path)
            
            tile_metadata.append((tile_file, tile_bbox))
    
    # 计算所有 TIF 影像的 overlap
    overlap_list = []
    for tile_file, tile_bbox in tile_metadata:
        overlap_area = calculate_overlap(geo_bbox, tile_bbox)
        if overlap_area > 0:  # 只考虑有重叠的瓦片
            overlap_list.append((tile_file, overlap_area))
    
    # 根据 overlap 进行排序，取 top_k
    overlap_list.sort(key=lambda x: x[1], reverse=True)
    
    # 返回 top_k 个影像路径
    top_list = [os.path.join(remote_sensing_tile_dir, item[0]) for item in overlap_list[:top_k]]
    
    return top_list
