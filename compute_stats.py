import rasterio

bands = []
path = '/Volumes/Backup Plus/erudite/drone/land_cover_segmentation/rasters/2017.jpg'
with rasterio.open(path) as src:
    bands.append(src.read(1))
    bands.append(src.read(2))
    bands.append(src.read(3))

for ind, b in enumerate(bands):
    print(f'{ind}: mean={b.mean()}, std={b.std()}')