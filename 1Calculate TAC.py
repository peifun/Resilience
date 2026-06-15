import rasterio
import numpy as np
from osgeo import gdal
from joblib import Parallel, delayed
from tqdm import tqdm
import statsmodels.api as sm

input_raster2 = rasterio.open(r"F:\kNDVI.tif")
input_raster3 = gdal.Open(r"D:\NDVI_1982_1.tif")

cols2 = input_raster2.width
rows2 = input_raster2.height
raster_col = input_raster3.RasterXSize
raster_row = input_raster3.RasterYSize
data2 = input_raster2.read(1)
result = np.empty((1, cols2))

def calculate_resilience(col_idx, data2):
    col_data2 = data2[:, col_idx]
    if np.sum(col_data2) == 0:
        return np.nan
    acf_values = sm.tsa.acf(col_data2, nlags=1, fft=True)
    TAC1 = acf_values[1] if len(acf_values) > 1 else np.nan
    return TAC1
resilience_values = Parallel(n_jobs=-1)(
    delayed(calculate_resilience)(col, data2) for col in tqdm(range(cols2)))
result = np.array(resilience_values).reshape((1, cols2))
result = result.reshape(raster_row, raster_col)
result[np.isinf(result) | np.isnan(result)] = 0
output_tif_path = r"F:\kNDVI_TAC.tif"
driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(
    output_tif_path,
    raster_col,
    raster_row,
    1,
    gdal.GDT_Float32,
)
output_dataset.SetProjection(input_raster3.GetProjection())
output_dataset.SetGeoTransform(input_raster3.GetGeoTransform())
output_dataset.GetRasterBand(1).WriteArray(result)
output_dataset.FlushCache()
output_dataset = None
