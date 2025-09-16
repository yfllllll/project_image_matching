from osgeo import gdal
import os

def slice_image(input_image_path, output_dir, tile_size_x=1024, tile_size_y=1024, step_size_x=256, step_size_y=512):
    """
    使用GDAL对遥感影像进行自适应坐标方向的切片，并确保切片的地理坐标与原影像一致。

    :param input_image_path: 输入遥感影像的路径
    :param output_dir: 输出切片文件的目录
    :param tile_size_x: 切片的宽度（像素）
    :param tile_size_y: 切片的高度（像素）
    :param step_size_x: x方向的步长（像素）
    :param step_size_y: y方向的步长（像素）
    """
    # 打开输入影像
    dataset = gdal.Open(input_image_path)
    if dataset is None:
        raise Exception(f"无法打开影像文件: {input_image_path}")

    # 获取影像的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 获取影像的地理转换信息
    geotransform = dataset.GetGeoTransform()
    origin_x = geotransform[0]  # 左上角X坐标
    origin_y = geotransform[3]  # 左上角Y坐标
    pixel_width = geotransform[1]  # X方向像素分辨率
    pixel_height = geotransform[5]  # Y方向像素分辨率（可能是负数）

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    # 遍历影像进行切片
    for y in range(0, height, step_size_y):  # y方向步长
        for x in range(0, width, step_size_x):  # x方向步长
            # 计算切片的实际大小
            current_tile_size_x = min(tile_size_x, width - x)
            current_tile_size_y = min(tile_size_y, height - y)

            # 计算切片的地理范围（确保方向正确）
            min_x = origin_x + x * pixel_width
            min_y = origin_y + y * pixel_height  # 直接使用 pixel_height

            # 计算右下角坐标（用于文件命名）
            max_x = min_x + current_tile_size_x * pixel_width
            max_y = min_y + current_tile_size_y * pixel_height  # 直接使用 pixel_height

            # 确保 `SetGeoTransform()` 左上角坐标正确
            output_geotransform = (min_x, pixel_width, 0, min_y, 0, pixel_height)

            # 构建输出文件名，使用空间分辨率和地理范围
            bbox = [min(min_x, max_x), min(min_y, max_y), max(min_x, max_x), max(min_y, max_y)]
            output_filename = f"spatial_res_{abs(pixel_width)}_{abs(pixel_height)}_x_{bbox[0]}_y_{bbox[1]}_x_{bbox[2]}_y_{bbox[3]}.tif"
            output_path = os.path.join(output_dir, output_filename)

            # 创建输出数据集
            driver = gdal.GetDriverByName('GTiff')
            output_dataset = driver.Create(output_path, current_tile_size_x, current_tile_size_y,
                                           dataset.RasterCount, dataset.GetRasterBand(1).DataType)
            output_dataset.SetGeoTransform(output_geotransform)
            output_dataset.SetProjection(dataset.GetProjection())

            # 读取并写入切片数据
            for band in range(1, dataset.RasterCount + 1):
                in_band = dataset.GetRasterBand(band)
                out_band = output_dataset.GetRasterBand(band)
                data = in_band.ReadAsArray(x, y, current_tile_size_x, current_tile_size_y)
                out_band.WriteArray(data)

            # 释放资源
            output_dataset.FlushCache()
            output_dataset = None
            in_band = None
            out_band = None

    # 关闭输入数据集
    dataset = None