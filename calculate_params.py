import math
import numpy as np

def calculate_sensor_params(focal_length_mm, fov_degrees, image_width_px, image_height_px, method='diagonal'):
    """
    计算相机传感器参数
    
    参数:
    focal_length_mm: 镜头焦距(毫米)
    fov_degrees: 视场角(度)
    image_width_px: 图像宽度(像素)
    image_height_px: 图像高度(像素)
    method: 计算方法 ('diagonal', 'horizontal', 'vertical')
    
    返回:
    包含传感器参数的字典
    """
    # 将角度转换为弧度
    fov_rad = math.radians(fov_degrees)
    
    # 计算对角线像素数
    diag_px = math.sqrt(image_width_px**2 + image_height_px**2)
    
    # 计算对角线长度
    if method == 'diagonal':
        # 基于对角线视场角计算
        half_fov = fov_rad / 2
        diag_length_mm = 2 * focal_length_mm * math.tan(half_fov)
        pixel_size_mm = diag_length_mm / diag_px
        
        # 计算传感器尺寸
        aspect_ratio = image_width_px / image_height_px
        sensor_height_mm = diag_length_mm / math.sqrt(1 + aspect_ratio**2)
        sensor_width_mm = sensor_height_mm * aspect_ratio
        
    elif method == 'horizontal':
        # 基于水平视场角计算
        half_fov = fov_rad / 2
        sensor_width_mm = 2 * focal_length_mm * math.tan(half_fov)
        pixel_size_mm = sensor_width_mm / image_width_px
        
        # 计算传感器高度
        sensor_height_mm = sensor_width_mm * (image_height_px / image_width_px)
        diag_length_mm = math.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
        
    elif method == 'vertical':
        # 基于垂直视场角计算
        half_fov = fov_rad / 2
        sensor_height_mm = 2 * focal_length_mm * math.tan(half_fov)
        pixel_size_mm = sensor_height_mm / image_height_px
        
        # 计算传感器宽度
        sensor_width_mm = sensor_height_mm * (image_width_px / image_height_px)
        diag_length_mm = math.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
    
    # 计算焦距(像素单位)
    fx = focal_length_mm / pixel_size_mm
    fy = fx  # 假设正方形像素
    
    # 计算主点坐标
    cx = image_width_px / 2
    cy = image_height_px / 2
    
    # 构建内参矩阵
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return {
        'sensor_width_mm': sensor_width_mm,
        'sensor_height_mm': sensor_height_mm,
        'diagonal_length_mm': diag_length_mm,
        'pixel_size_mm': pixel_size_mm,
        'focal_length_px': fx,
        'principal_point': (cx, cy),
        'intrinsic_matrix': K,
        'calculation_method': method
    }

# 示例使用（基于您提供的参数）
if __name__ == "__main__":
    # 输入参数
    focal_length = 7 # mm
    fov = 82  # 度
    width = 3840  # 像素
    height = 2160  # 像素
    
    # 三种计算方法
    methods = ['diagonal']
    
    for method in methods:
        print(f"\n=== 使用 {method} 方法计算 ===")
        params = calculate_sensor_params(focal_length, fov, width, height, method)
        
        print(f"传感器宽度: {params['sensor_width_mm']:.4f} mm")
        print(f"传感器高度: {params['sensor_height_mm']:.4f} mm")
        print(f"对角线长度: {params['diagonal_length_mm']:.4f} mm")
        print(f"像素尺寸: {params['pixel_size_mm']:.6f} mm")
        print(f"焦距(像素单位): {params['focal_length_px']:.2f} px")
        print(f"主点坐标: ({params['principal_point'][0]}, {params['principal_point'][1]})")
        print("内参矩阵 K:")
        print(params['intrinsic_matrix'])