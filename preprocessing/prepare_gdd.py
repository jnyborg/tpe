from datetime import datetime, timedelta
import geopandas as gpd
from argparse import ArgumentParser
import os
from pathlib import Path
import pickle
from typing import List

import numpy as np
from tqdm import tqdm

from preprocessing.downloader import S2Image, s2_download

from osgeo import osr, gdal


def get_bbox(raster, projection):
    # Create the transformation from raster to epsg 4326 (lat lng) coordinates
    transform_src = osr.SpatialReference(wkt=raster.GetProjection())
    transform_target = osr.SpatialReference(wkt=projection)
    if int(gdal.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        transform_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform_target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(transform_src, transform_target)

    # Extract the rasters bounding box
    xmin, xpixel, _, ymax, _, ypixel = raster.GetGeoTransform()
    width, height = raster.RasterXSize, raster.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel

    ul = transform.TransformPoint(xmin, ymax)
    lr = transform.TransformPoint(xmax, ymin)

    return (ul, lr)


def prepare_gdd(config):
    s2_images: List[S2Image] = s2_download(
        config.tile,
        config.start_date,
        config.end_date,
        config.download_dir,
        min_coverage=config.min_coverage,
        max_cloudy_pct=config.max_cloudy_percentage,
        sort_by_date=True,
        bands=config.bands,
        data_collection=config.data_collection,
        # previews=True,
    )

    # Load Sentinel-2 data
    sentinel_image = str(s2_images[0].bands_10m[0])
    # Load E-OBS data
    min_temp_path = (
        "/media/data/eobs_weather_data/tn_ens_mean_0.1deg_reg_2011-2020_v23.1e.nc"
    )
    max_temp_path = (
        "/media/data/eobs_weather_data/tx_ens_mean_0.1deg_reg_2011-2020_v23.1e.nc"
    )
    # precipitation_path = (
    #     "/media/data/eobs_weather_data/rr_ens_mean_0.1deg_reg_2011-2021_v24.0e.nc"
    # )

    t_min, t_min_geotransform = read_eobs(min_temp_path, sentinel_image)
    t_max, _ = read_eobs(max_temp_path, sentinel_image)
    # rain, _ = read_eobs(precipitation_path, sentinel_image)

    # np.save(f'weather_data/{config.tile}_t_min.npy', t_min)
    # np.save(f'weather_data/{config.tile}_t_max.npy', t_max)
    # np.save(f'weather_data/{config.tile}_rain.npy', rain)

    # Calculate GDD for each parcel in dateset
    print("reading ground truth file")
    print(config.meta_dir)
    parcels_file = list((config.meta_dir / "parcels").glob("*.shp"))[0]
    parcels = gpd.read_file(parcels_file)
    parcels.to_crs("EPSG:4326", inplace=True)

    t_min_data = []
    t_max_data = []
    if config.tile == '32VNH':
        # hacky forward fill to change water temp to nearst (to left) surface temp
        # only "works" for 32VNH since water is to the right
        for row_idx in range(t_min.shape[1]):
            for col_idx in range(1, t_min.shape[2]):
                if t_min[0, row_idx, col_idx] == -9999.0:
                    t_min[:, row_idx, col_idx] = t_min[:, row_idx, col_idx - 1]

        assert not (t_min == -9999.0).any()
        for row_idx in range(t_max.shape[1]):
            for col_idx in range(1, t_max.shape[2]):
                if t_max[0, row_idx, col_idx] == -9999.0:
                    t_max[:, row_idx, col_idx] = t_max[:, row_idx, col_idx - 1]

        assert not (t_max == -9999.0).any()

    for i, row in tqdm(parcels.iterrows(), total=len(parcels), desc="loading gdd"):
        geometry = row.geometry
        centroid = geometry.centroid
        pixel, line = world_to_pixel(
            t_min_geotransform, centroid.x, centroid.y
        )
        mn = t_min[:, line, pixel]
        mx = t_max[:, line, pixel]
        t_min_data.append(mn)
        t_max_data.append(mx)

    print("Writing weather data...")
    metadata = {'t_min': t_min_data, 't_max': t_max_data}
    pickle.dump(metadata, open(config.meta_dir / "weather_data.pkl", "wb"))


def read_eobs(file_path, s2_image_path):
    # Load Sentinel-2 data
    sentinel_raster = gdal.Open(s2_image_path)

    weather_raster = gdal.Open(file_path)
    weather_raster.SetProjection("EPSG:4326")

    days_since_weather = list(
        map(
            int,
            weather_raster.GetMetadata()["NETCDF_DIM_time_VALUES"]
            .replace("{", "")
            .replace("}", "")
            .split(","),
        )
    )
    eobs_start_date = datetime(1950, 1, 1)
    dates_weather = list(
        map(lambda days: eobs_start_date + timedelta(days=days), days_since_weather)
    )

    start_date_s2 = datetime.strptime(config.start_date, "%Y%m%d")
    end_date_s2 = datetime.strptime(config.end_date, "%Y%m%d")

    relevant_bands = [
        i for i, d in enumerate(dates_weather) if start_date_s2 <= d <= end_date_s2
    ]

    # Read S2 bounding box from weather data
    bbox = get_bbox(sentinel_raster, weather_raster.GetProjection())
    bbox_px = [
        world_to_pixel(weather_raster.GetGeoTransform(), x, y) for x, y, _ in bbox
    ]

    options_list = [
        f"-srcwin {bbox_px[0][0]} {bbox_px[0][1]} {bbox_px[1][0] - bbox_px[0][0] + 1} {bbox_px[1][1] - bbox_px[0][1] + 1}",
        "-of",
        "MEM",
        f'-b {" -b ".join(map(str, relevant_bands))}',
    ]

    options_string = " ".join(options_list)
    weather_raster = gdal.Translate("", weather_raster, options=options_string)
    data = weather_raster.ReadAsArray()
    geotransform = weather_raster.GetGeoTransform()

    return data, geotransform


def world_to_pixel(geo_matrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ul_x = geo_matrix[0]
    ul_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    y_dist = geo_matrix[5]
    pixel = int((x - ul_x) / x_dist)
    line = -int((ul_y - y) / y_dist)
    return pixel, line


def expoweight(theta, beta=0.9):
    v0 = 0
    vs = np.zeros(theta.shape)
    T = theta.shape[0]
    for t in range(1, T + 1):
        vt = beta * v0 + (1 - beta) * theta[t - 1]
        vs[t - 1] = vt
        v0 = vt
    return vs


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--tile", type=str, default="32VNH")
    args.add_argument(
        "--country",
        type=str,
        choices=["denmark", "france", "austria"],
        default="denmark",
        help="ground truth file to use",
    )
    args.add_argument("--years", nargs="+", required=True)
    args.add_argument("--start", default="0101", help="start date in format mmdd")
    args.add_argument("--end", default="1231", help="end date in format mmdd")
    args.add_argument("--max_cloudy_percentage", type=float, default=80.0)
    args.add_argument("--min_coverage", type=float, default=50.0)
    args.add_argument("--download_dir", default="/media/data/s2")
    args.add_argument("--ground_truth_dir", default="/media/data/parcels")
    args.add_argument("--output_dir", default="/media/data/timematch_data")
    args.add_argument(
        "--data_collection", type=str, default="l1c", choices=["l1c", "l2a"]
    )
    args.add_argument("--block_size", type=int, default=1098)
    args.add_argument(
        "--margin", type=int, default=0
    )  # default 0, as we remove parcels that overlap between blocks
    args.add_argument(
        "--buffer_size",
        type=int,
        default=10,
        help="number of S2 images to load to memory at once",
    )

    config = args.parse_args()

    if config.data_collection == "l1c":
        config.bands = [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ]
    else:
        config.bands = [
            "R10m/B02",
            "R10m/B03",
            "R10m/B04",
            "R20m/B05",
            "R20m/B06",
            "R20m/B07",
            "R10m/B08",
            "R20m/B8A",
            "R20m/B11",
            "R20m/B12",
        ]

    config.download_dir = os.path.join(config.download_dir, config.data_collection)

    for year in config.years:
        config.year = year
        output_dir = (
            Path(config.output_dir) / config.country / config.tile / config.year
        )
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True, parents=True)
        meta_dir = output_dir / "meta"
        meta_dir.mkdir(exist_ok=True, parents=True)
        config.meta_dir = meta_dir
        config.data_dir = data_dir

        config.end_date = config.year + config.end
        if int(config.end) < int(config.start):
            config.start_date = str(int(config.year) - 1) + config.start
        else:
            config.start_date = config.year + config.start
        print(f"start_date={config.start_date}, end_date={config.end_date}")

        print(config)
        prepare_gdd(config)
