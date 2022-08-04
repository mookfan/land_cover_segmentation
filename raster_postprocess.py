'''
    Post-processing after get segmentaion results 
    1. stitch sub-rasters to raster 
    TODO: 2. add gcp upper-left & lower-right if no gcp data in raw-raster
    TODO: 3. create shapefile from segmented raster
'''
import numpy as np
import os
import rasterio as rio
from rasterio.merge import merge
from rasterio.plot import show
from config.default import get_cfg_from_file
from utils.io_utils import load_yaml
from utils.raster_utils import raster_to_np, np_to_raster

import matplotlib.pyplot as plt

def raster_mosiac(list_path, visualize=False):
    src_files_to_mosaic = []
    for fp in seg_list_path:
        src = rio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    if visualize:
        show(mosaic, cmap='terrain')
    return mosaic, out_trans, src_files_to_mosaic


seg_dir = 'debug_results/S2A_2022-01-01_2022-01-31_median_256_mean_std_channels_stats/raster'
cfg_path = 'config/weighted_loss_more_snow_data_aug_hrnet.yml'

cfg = get_cfg_from_file(cfg_path)
mask_config_path = cfg.DATASET.MASK.CONFIG
mask_config = load_yaml(mask_config_path)

seg_list_path = [os.path.join(seg_dir, fname) for fname in os.listdir(seg_dir)]

raster, out_trans, scr_files_DatasetReader  = raster_mosiac(seg_list_path, visualize=True)
raster_crs = scr_files_DatasetReader[0].crs

raster = raster.transpose(1, 2, 0) # from (ch, w, h) > (w, h, ch)
raster = raster.astype(np.uint8)

colors_dict = mask_config['colors']
labels_dict = mask_config['class2label']
image = np.ones((raster.shape[0], raster.shape[1]), dtype=np.uint8)
for class_int, color in colors_dict.items():
    dummy_image = image.copy()
    mask = np.where(raster == color, class_int, -1)[:, :, 0] # focus only with, height
    class_mask = np.where(mask == class_int)
    image[class_mask] = class_int
# get segmented raster 'test.tif' which has pixel-value = class_int
with rio.open(
        'test.tif',
        "w",
        driver="GTiff",
        dtype=image.dtype,
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        crs=raster_crs,
        transform=out_trans,
    ) as dst:
        dst.write(image, 1)

# # ==============to shape file ========================
# from rasterio.features import shapes
# from shapely.geometry import shape, Polygon

# from geopandas import GeoDataFrame
# from pandas import DataFrame

# with rio.open('test.tif') as src:
#     data = src.read(1)
#     mask = data != -1
#     # # Use a generator instead of a list
#     shape_gen = ((s, v) for s, v in shapes(data, mask=mask, transform=out_trans))

#     # # either build a pd.DataFrame
#     df = DataFrame(shape_gen, columns=['geometry', 'class'])
#     gdf = GeoDataFrame(df["class"], geometry=df.geometry, crs=src.crs)

#     print()
#     # # or build a dict from unpacked shapes
#     gdf = GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=src.crs)
#     GeoDataFrame.to_file('test.geojson', driver='GeoJSON')
#     print(gdf)

# # ======================================
# import rasterio
# from rasterio.features import shapes
# mask = None
# results=None
# with rasterio.Env():
#     with rasterio.open('test.tif') as src:
#         image = src.read(1) # first band
#         results = [{'properties': {'raster_val': v}, 'geometry': s}
#         for i, [s, v] in enumerate(shapes(image, mask=mask, transform=src.transform))]