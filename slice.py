from tiles.create_tile import slice_image

sat_path = "datasets/imgs/source/50cm/range2_50cm.tif"
save_dir = 'datasets/imgs/slice/50cm/'
#根据sat_path的文件名，创建文件夹目录
save_dir = save_dir + sat_path.split('/')[-1].split('.')[0] + '/'

slice_image(sat_path, save_dir)