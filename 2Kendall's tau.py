import rasterio
import numpy as np
from osgeo import gdal
from joblib import Parallel, delayed
from tqdm import tqdm
import pymannkendall as mk  # 用于 Kendall's tau 统计量

input_raster2 = rasterio.open(r"F:\2AT_CV.tif")
input_raster3 = gdal.Open(r"F:\1984.tif")
cols2 = input_raster2.width
rows2 = input_raster2.height
raster_col = input_raster3.RasterXSize
raster_row = input_raster3.RasterYSize
data2 = input_raster2.read(1)
def calculate_kendall_tau(col_idx, data2):
    col_data = data2[:, col_idx]
    if np.all(col_data == col_data[0]):
        return np.nan,np.nan
    try:
        mk_result = mk.original_test(col_data)
        return mk_result.Tau, mk_result.p
    except ZeroDivisionError:
        return np.nan, np.nan

kendall_tau_values = Parallel(n_jobs=-1)(
    delayed(calculate_kendall_tau)(col, data2)
    for col in tqdm(range(cols2), desc="Calculating Kendall's tau"))
tau_values = [r[0] for r in kendall_tau_values]
p_values = [r[1] for r in kendall_tau_values]

resulttau = np.array(tau_values).reshape((1, cols2))
resulttau = resulttau.reshape(raster_row, raster_col)
resulttau[np.isinf(resulttau) | np.isnan(resulttau)] = 0

resultP = np.array(p_values).reshape((1, cols2))
resultP = resultP.reshape(raster_row, raster_col)
resultP[np.isinf(resultP) | np.isnan(resultP)] = 0

output_tif_path = r"E:\1AT_CV_tau.tif"
driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(
    output_tif_path,
    raster_col,
    raster_row,
    1,
    gdal.GDT_Float32
)
output_dataset.SetProjection(input_raster3.GetProjection())
output_dataset.SetGeoTransform(input_raster3.GetGeoTransform())
output_dataset.GetRasterBand(1).WriteArray(resulttau)
output_dataset.FlushCache()
output_dataset = None

output_tif_path = r"E:\1AT_CV_Kendall_P.tif"
driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(
    output_tif_path,
    raster_col,
    raster_row,
    1,
    gdal.GDT_Float32
)
output_dataset.SetProjection(input_raster3.GetProjection())
output_dataset.SetGeoTransform(input_raster3.GetGeoTransform())
output_dataset.GetRasterBand(1).WriteArray(resultP)
output_dataset.FlushCache()
output_dataset = None

