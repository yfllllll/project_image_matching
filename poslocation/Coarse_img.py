import os
import numpy as np
import time
from poslocation.ExifData import *
from poslocation.EoData import *
from poslocation.Boundary import boundary
from poslocation.BackprojectionResample import rectify_plane_parallel, createGeoTiff
from rich.console import Console
from rich.table import Table
from retrieve.Desmodel import Sample4Geo
from retrieve.utils import calculate_overlap, parse_bbox_from_filename, get_tif_metadata

Support_search_model = {'Sample4Geo':Sample4Geo}

class CameraImg:
    
    def __init__(self, file_path, sensor_width = 6.3, ground_height = 0, epsg = 3857, gsd = 0.1):
        self.file_path = file_path
        self.sensor_width = sensor_width
        self.image = cv2.imread(file_path, -1)
        self.ground_height =ground_height
        self.epsg = epsg
        self.gsd = gsd
        self.info = self.get_info(file_path)
        self.sensor_width = self.info['sensor_width']
        self.image = restoreOrientation(self.image, self.info['orientation'])
        self.image_rows = self.image.shape[0]
        self.image_cols = self.image.shape[1]
        self.R = Rot3D(self.info['eo']) # 相机外方角元素导出的旋转矩阵
        self.pixel_size = (self.sensor_width / self.image_cols) /1000 #每个像素的大小，单位m
        self.bbox = self.get_boundry()  
        self.get_tif_rows_cols()   
        # self.createGeoTiff()
        # self.info_for_project_point_to_orthoimage = {
        #         'eo': list(self.info['eo']),  # 强制转列表
        #         'R': self.R.tolist(),         # numpy数组 → 列表
        #         'ground_height': float(self.ground_height),
        #         'pixel_size': float(self.pixel_size) if isinstance(self.pixel_size, float) else tuple(self.pixel_size),
        #         'focal_length': float(self.info['focal_length']),
        #         "boundary": list(self.bbox),  # 元组/数组 → 列表
        #         "image_shape": [int(self.image_rows), int(self.image_cols)]  # 确保整数
        #     }
        self.info_for_project_point_to_orthoimage = {
            'eo': self.info['eo'],  # 强制转列表
            'R': self.R,         # numpy数组 → 列表
            'ground_height': self.ground_height,
            'pixel_size': self.pixel_size,
            'focal_length': self.info['focal_length'],
            "boundary": self.bbox,  # 元组/数组 → 列表
            "image_shape": [int(self.image_rows), int(self.image_cols)],  # 确保整数
            "gsd": self.gsd
        }
    def get_boundry(self):
        bbox = boundary(self.image, self.info['eo'], self.R, self.ground_height, self.pixel_size, self.info['focal_length'])
        return bbox

    def get_tif_rows_cols(self):
        # self.bbox = self.get_boundry()
        # gsd = self.gsd
        if self.gsd == 0:
            self.gsd = (self.pixel_size * (self.info['eo'][2] - self.ground_height)) / self.info['focal_length']  # unit: m/px
        self.boundary_cols = int((self.bbox[1, 0] - self.bbox[0, 0]) / self.gsd)
        self.boundary_rows = int((self.bbox[3, 0] - self.bbox[2, 0]) / self.gsd)
        
    def get_coarse_tif(self):
        b, g, r, a = self.coarse_loc()
        coarse_tif = np.dstack([b, g, r])
        return coarse_tif
        
        
    def coarse_loc(self):

        console = Console()  
        start_time = time.time()
        print('DEM & GSD')
        start_time = time.time()
        # 3. Extract a projected boundary of the image
        bbox = self.bbox
        dem_time = time.time() - start_time
        console.print(f"DEM time: {dem_time:.2f} sec", style="blink bold red underline")

        print('Rectify & Resampling')
        start_time = time.time()
        b, g, r, a = rectify_plane_parallel(bbox, self.boundary_rows, self.boundary_cols, self.gsd, self.info['eo'], self.ground_height,
                                            self.R, self.info['focal_length'], self.pixel_size, self.image)
        rectify_time = time.time() - start_time
        console.print(f"Rectify time: {rectify_time:.2f} sec", style="blink bold red underline")
        
        return b,g,r,a


    def get_info(self, file_path):
        extension =  os.path.splitext(os.path.basename(file_path))[1]
        info = {'focal_length':None,
                'orientation':None,
                'eo':None,
                'maker':None,
                'sensor_width':None}
        
        if extension == '.JPG' or extension == '.jpg':
            
            # 1. Extract metadata from a image
            focal_length, orientation, eo, maker, sensor_width = get_metadata(file_path)  # unit: m, _, ndarray
            eo = geographic2plane(eo, self.epsg)
            opk = rpy_to_opk(eo[3:], maker) # 获取 相机位姿
            eo[3:] = opk * np.pi / 180   # degree to radian
            info['focal_length'] = focal_length
            info['orientation'] = orientation
            info['eo'] = eo
            info['maker'] = maker
            info['sensor_width'] = sensor_width
            # 2. Restore the image based on orientation information
            
        return info
            
    def createGeoTiff(self):
        dst = self.file_path
        b, g, r, a = self.coarse_loc()
        createGeoTiff(b, g, r, a, self.bbox, self.gsd, self.epsg, self.boundary_rows, self.boundary_cols, dst)
        
    
class SmartImage:  
    def __init__(self, file_path, search_model='Sample4Geo', device=0, sensor_width=None, ground_height=0, epsg=3857, gsd=0.2, **kwargs):  
        self.file_path = file_path  
        self.image = cv2.imread(file_path, -1)  
        self._camera_img = None  
        self._has_metadata = None  
        self.sensor_width = sensor_width
        self.ground_height = ground_height
        self.epsg = epsg
        self.gsd = gsd
        self.search_model = Support_search_model[search_model](device)
        
    @property  
    def has_metadata(self):  
        """懒加载检测是否有元数据"""  
        if self._has_metadata is None:  
            self._has_metadata = self._detect_metadata()  
        return self._has_metadata  
      
    @property  
    def camera_img(self):  
        """懒加载CameraImg对象"""  
        if self._camera_img is None and self.has_metadata:  
            self._camera_img = CameraImg(self.file_path, self.sensor_width, self.ground_height, self.epsg, self.gsd)  
        return self._camera_img  
      
    def _detect_metadata(self):  
        """检测元数据，同时缓存CameraImg对象"""  
        try:  
            self._camera_img = CameraImg(self.file_path, self.sensor_width, self.ground_height, self.epsg, self.gsd)  
            info = self._camera_img.info  
            return (info['focal_length'] is not None and   
                   info['eo'] is not None and   
                   len(info['eo']) == 6 and  
                   info['eo'][0] != 0 and info['eo'][1] != 0)  
        except Exception:  
            return False  
    # def _detect_metadata(self):  
    #     """检测元数据，同时缓存CameraImg对象"""  
     
    #     self._camera_img = CameraImg(self.file_path, self.sensor_width, self.ground_height, self.epsg, self.gsd)  
    #     info = self._camera_img.info  
    #     return (info['focal_length'] is not None and   
    #             info['eo'] is not None and   
    #             len(info['eo']) == 6 and  
    #             info['eo'][0] != 0 and info['eo'][1] != 0)  
    
    def do_retrieval_(self, remote_sensing_tile_dir, top_k=1):  
        if self.has_metadata:  
            geo_bbox = self.camera_img.bbox
            geo_bbox = [geo_bbox[0],geo_bbox[2],geo_bbox[1],geo_bbox[3]]  
            top_list = {self.camera_img.file_path:self.get_topk_overlap(geo_bbox, remote_sensing_tile_dir,top_k=top_k)}  
            return top_list, self.camera_img
        else:  
            top_list = self.search_model(self.file_path, remote_sensing_tile_dir,   
                                     top_k=top_k, gallery_batch_size=16) 
            return top_list, None 
      

    def get_topk_overlap(self, geo_bbox, remote_sensing_tile_dir, top_k=2):
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

