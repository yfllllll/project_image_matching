from tiles.create_tile import slice_image

sat_path = "datasets/source/satellite/云南省西双版纳州勐腊县尚勇镇南坡村北.tif"
save_dir = 'datasets/slice/'
#根据sat_path的文件名，创建文件夹目录
save_dir = save_dir + sat_path.split('/')[-1].split('.')[0] + '/'

slice_image(sat_path, save_dir)