"""Study areas, NAIP imagery and reference data.

Everything here is open data:
- USDA NAIP 4-band imagery from the Microsoft Planetary Computer STAC (`naip`).
- Microsoft Building Footprints from the Planetary Computer (`ms-buildings`).
- OpenStreetMap buildings and farmland polygons from the Overpass API (ODbL).
- USDA Cropland Data Layer (`usda-cdl`) as a coarse 30 m check for fields.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UTM_CRS = "EPSG:32614"  # UTM zone 14N covers Bryan-College Station
RES = 0.6  # metres, NAIP native resolution in Texas since 2020
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ~1 km x 1 km squares around Bryan-College Station, TX (lon, lat of the centre).
# `fields`/`buildings`: which extractions to run; `role`: dev = building rules tuned here.
AOIS = {
    "residential": {"center": (-96.2975, 30.5845), "label": "Residential, south College Station",
                    "fields": False, "role": "dev"},
    "agricultural": {"center": (-96.4300, 30.5480), "label": "Brazos River bottom farmland",
                     "fields": True, "buildings": False, "role": "fields"},
    "mixed": {"center": (-96.3530, 30.6560), "label": "Mixed urban edge, Bryan",
              "fields": False, "role": "held-out"},
}


@dataclass
class Grid:
    """A north-up raster grid in UTM metres."""

    x0: float
    y1: float
    width: int
    height: int
    res: float = RES
    crs: str = UTM_CRS

    @property
    def transform(self):
        from rasterio.transform import from_origin

        return from_origin(self.x0, self.y1, self.res, self.res)

    @property
    def bounds(self):
        return (self.x0, self.y1 - self.height * self.res, self.x0 + self.width * self.res, self.y1)

    def lonlat_bbox(self):
        from rasterio.warp import transform_bounds

        return transform_bounds(self.crs, "EPSG:4326", *self.bounds, densify_pts=21)


def grid_from_center(lon: float, lat: float, size_m: float = 1000.0, res: float = RES) -> Grid:
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True).transform(lon, lat)
    n = int(math.ceil(size_m / res))
    x0 = math.floor((x - size_m / 2) / res) * res
    y1 = math.floor((y + size_m / 2) / res) * res
    return Grid(x0=x0, y1=y1, width=n, height=n, res=res)


def grid_from_bbox(west: float, south: float, east: float, north: float, res: float = RES) -> Grid:
    from rasterio.warp import transform_bounds

    l, b, r, t = transform_bounds("EPSG:4326", UTM_CRS, west, south, east, north, densify_pts=21)
    x0 = math.floor(l / res) * res
    y1 = math.ceil(t / res) * res
    return Grid(x0=x0, y1=y1, width=int(math.ceil((r - x0) / res)), height=int(math.ceil((y1 - b) / res)), res=res)


def _catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def fetch_naip(grid: Grid, out_path: Path) -> dict:
    """Mosaic the most recent NAIP year onto `grid`; writes a 4-band uint8 GeoTIFF."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())

    items = list(_catalog().search(collections=["naip"], bbox=grid.lonlat_bbox()).items())
    if not items:
        raise RuntimeError("No NAIP items found for this area")
    latest = max(i.datetime.year for i in items)
    items = sorted([i for i in items if i.datetime.year == latest], key=lambda i: i.datetime)

    dst = np.zeros((4, grid.height, grid.width), dtype=np.uint8)
    for item in items:
        with rasterio.open(item.assets["image"].href) as src:
            tmp = np.zeros_like(dst)
            reproject(
                source=rasterio.band(src, [1, 2, 3, 4]),
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )
        empty = dst.sum(0) == 0
        dst[:, empty] = tmp[:, empty]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=4, dtype="uint8",
                   crs=grid.crs, transform=grid.transform, compress="deflate", tiled=True)
    with rasterio.open(out_path, "w", **profile) as dst_ds:
        dst_ds.write(dst)
        dst_ds.descriptions = ("red", "green", "blue", "nir")
    meta = {
        "year": latest,
        "dates": sorted({i.datetime.date().isoformat() for i in items}),
        "items": [i.id for i in items],
        "gsd_m": items[0].properties.get("gsd"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def read_naip(path: Path):
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read()
    rgb = np.moveaxis(arr[:3], 0, -1).copy()
    nir = arr[3].astype(np.float32)
    red = arr[0].astype(np.float32)
    ndvi = (nir - red) / np.maximum(nir + red, 1.0)
    return rgb, ndvi, arr


def _clip_to_grid(gdf, grid: Grid):
    """Reproject to UTM and keep features whose centroid lies inside the grid."""
    from shapely.geometry import box

    if gdf.empty:
        return gdf.to_crs(grid.crs)
    g = gdf.to_crs(grid.crs)
    g = g[g.geometry.notna() & g.is_valid]
    inside = g.geometry.centroid.within(box(*grid.bounds))
    return g[inside].reset_index(drop=True)


def fetch_ms_buildings(grid: Grid, out_path: Path, release: str = "United States_2022-07-06"):
    """Microsoft Building Footprints (global ML release, US region) from the Planetary Computer.
    The GeoParquet is partitioned by zoom-9 quadkey; only the partitions under the AOI are read."""
    import adlfs
    import geopandas as gpd
    import mercantile
    import pandas as pd
    from shapely import wkb
    from shapely.geometry import box

    out_path = Path(out_path)
    if out_path.exists():
        return gpd.read_file(out_path)
    w, s, e, n = grid.lonlat_bbox()
    item = _catalog().get_collection("ms-buildings").get_item(release)
    asset = item.assets["data"]
    fs = adlfs.AzureBlobFileSystem(**asset.extra_fields["table:storage_options"])
    root = asset.href.replace("abfs://", "")
    frames = []
    for qk in {mercantile.quadkey(t) for t in mercantile.tiles(w, s, e, n, zooms=9)}:
        for part in fs.ls(f"{root}/quadkey={qk}"):
            with fs.open(part, "rb") as f:
                df = pd.read_parquet(f)
            geoms = df["geometry"].apply(lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g)
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms), crs="EPSG:4326")
            frames.append(gdf[gdf.intersects(box(w, s, e, n))])
    if not frames:
        raise RuntimeError("No Microsoft building partitions found")
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    gdf = _clip_to_grid(gdf, grid)
    gdf.to_file(out_path, driver="GeoJSON")
    return gdf


def _overpass(query: str) -> dict:
    import requests

    headers = {"User-Agent": "samnaip-portfolio/0.1 (research demo; small bbox queries)"}
    for attempt in range(3):
        r = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=180)
        if r.status_code == 200:
            return r.json()
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    return {}


def _osm_polygons(elements) -> list:
    """Closed ways -> polygons (multipolygon relations: outer rings only)."""
    from shapely.geometry import Polygon

    polys = []
    for el in elements:
        rings = []
        if el["type"] == "way" and "geometry" in el:
            rings = [el["geometry"]]
        elif el["type"] == "relation":
            rings = [m["geometry"] for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m]
        for ring in rings:
            coords = [(p["lon"], p["lat"]) for p in ring]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                p = Polygon(coords)
                if p.is_valid and p.area > 0:
                    polys.append(p)
    return polys


def fetch_osm(grid: Grid, out_path: Path, kind: str = "building"):
    """OSM buildings (kind='building') or farmland/meadow polygons (kind='farmland')."""
    import geopandas as gpd

    out_path = Path(out_path)
    if out_path.exists():
        return gpd.read_file(out_path)
    w, s, e, n = grid.lonlat_bbox()
    bb = f"{s},{w},{n},{e}"
    if kind == "building":
        body = f'way["building"]({bb});relation["building"]({bb});'
    else:
        body = f'way["landuse"~"farmland|meadow"]({bb});relation["landuse"~"farmland|meadow"]({bb});'
    data = _overpass(f"[out:json][timeout:120];({body});out geom;")
    gdf = gpd.GeoDataFrame(geometry=_osm_polygons(data.get("elements", [])), crs="EPSG:4326")
    gdf = _clip_to_grid(gdf, grid)
    gdf.to_file(out_path, driver="GeoJSON")
    return gdf


# CDL codes treated as cultivated fields (row crops, small grains, hay, fallow, double crops).
CDL_FIELD_CODES = set(range(1, 62)) | set(range(66, 78)) | set(range(204, 255))


def fetch_cdl_fields(grid: Grid, out_path: Path) -> np.ndarray:
    """Boolean 'cultivated field' mask from the latest USDA CDL, resampled (nearest) to `grid`."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    out_path = Path(out_path)
    if out_path.exists():
        return np.load(out_path)
    items = list(_catalog().search(collections=["usda-cdl"], bbox=grid.lonlat_bbox()).items())
    items = [i for i in items if "cropland" in i.assets]
    item = max(items, key=lambda i: i.datetime or i.properties.get("start_datetime"))
    dst = np.zeros((grid.height, grid.width), dtype=np.uint8)
    with rasterio.open(item.assets["cropland"].href) as src:
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid.transform, dst_crs=grid.crs, resampling=Resampling.nearest)
    mask = np.isin(dst, list(CDL_FIELD_CODES))
    np.save(out_path, mask)
    out_path.with_suffix(".json").write_text(json.dumps({"item": item.id}))
    return mask
