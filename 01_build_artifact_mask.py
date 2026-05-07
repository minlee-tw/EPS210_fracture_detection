# %%
"""
Build artifact_mask.npz from image_artifact.shp + EW_high_800.kmz extent.
Rasterizes the 8 LineString artifact features to the EW grid using
Bresenham's algorithm, then dilates by 20 pixels (matching the original mask).
"""

import os
import zipfile
import io
import xml.etree.ElementTree as ET

import numpy as np
import geopandas as gpd
from PIL import Image
from scipy.ndimage import binary_dilation

DATA_DIR = os.path.dirname (os.path.abspath (__file__))
RESULTS_DIR = os.path.join (DATA_DIR, "results")
os.makedirs (RESULTS_DIR, exist_ok = True)

KML_NS = "{http://www.opengis.net/kml/2.2}"
BUFFER_PX = 20

def parse_kmz_overlay (filepath):
    with zipfile.ZipFile (filepath) as zf:
        kml_name = [n for n in zf.namelist () if n.endswith (".kml")] [0]
        with zf.open (kml_name) as kf:
            tree = ET.parse (kf)
        root = tree.getroot ()
        for go in root.iter (f"{KML_NS}GroundOverlay"):
            href = go.find (f".//{KML_NS}href")
            lb = go.find (f".//{KML_NS}LatLonBox")
            if href is not None and lb is not None:
                img_name = href.text.strip ()
                bounds = {d: float (lb.find (f"{KML_NS}{d}").text)
                          for d in ["north", "south", "east", "west"]}
                with zf.open (img_name) as imgf:
                    img = np.array (Image.open (io.BytesIO (imgf.read ())))
                return img, bounds
    raise RuntimeError (f"No GroundOverlay in {filepath}")

def bresenham (r0, c0, r1, c1):
    rows, cols = [], []
    dr = abs (r1 - r0); dc = abs (c1 - c0)
    sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1
    err = dr - dc
    while True:
        rows.append (r0); cols.append (c0)
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r0 += sr
        if e2 < dr:  err += dr; c0 += sc
    return np.array (rows), np.array (cols)

# Get canonical grid from EW KMZ (this matches the displacement grid exactly)
img, bounds = parse_kmz_overlay (os.path.join (DATA_DIR, "EW_high_800.kmz"))
H, W = img.shape [:2]
lon = np.linspace (bounds ["west"], bounds ["east"], W)
lat = np.linspace (bounds ["south"], bounds ["north"], H)
print (f"Grid: {H} x {W}, extent lon = [{lon [0]:.3f}, {lon [-1]:.3f}], "
      f"lat = [{lat [0]:.3f}, {lat [-1]:.3f}]")

# Read artifact LineStrings
gdf = gpd.read_file (os.path.join (DATA_DIR, "image_artifact.shp"))
print (f"Read {len (gdf)} artifact features (CRS = {gdf.crs})")

# Rasterize each LineString onto grid
mask = np.zeros ((H, W), dtype = np.uint8)
lon_min, lon_max = lon [0], lon [-1]
lat_min, lat_max = lat [0], lat [-1]

for geom in gdf.geometry:
    coords = np.array (geom.coords)
    for k in range (len (coords) - 1):
        c0 = int (round ((coords [k, 0]   - lon_min) / (lon_max - lon_min) * (W - 1)))
        r0 = int (round ((coords [k, 1]   - lat_min) / (lat_max - lat_min) * (H - 1)))
        c1 = int (round ((coords [k+1, 0] - lon_min) / (lon_max - lon_min) * (W - 1)))
        r1 = int (round ((coords [k+1, 1] - lat_min) / (lat_max - lat_min) * (H - 1)))
        c0, c1 = np.clip ([c0, c1], 0, W - 1)
        r0, r1 = np.clip ([r0, r1], 0, H - 1)
        rr, cc = bresenham (r0, c0, r1, c1)
        mask [rr, cc] = 1

# The displacement grid is flipped (row 0 = south) in the training scripts via z [::-1].
# The fracture rasterizer uses lat ascending (lat [0] = south, lat [-1] = north), which matches
# the post-flip orientation. So mask above is already in post-flip coordinates.
# (Quick sanity check below: dilated count should be ~10^5-10^6)

print (f"Rasterized line pixels: {mask.sum ():,}")

# Dilate by BUFFER_PX
struct = np.ones ((3, 3), dtype = bool)
artifact_mask = mask.copy ()
for _ in range (BUFFER_PX):
    artifact_mask = binary_dilation (artifact_mask, structure = struct)
artifact_mask = artifact_mask.astype (np.uint8)

print (f"After {BUFFER_PX}-px dilation: {artifact_mask.sum ():,} pixels "
      f"({100*artifact_mask.mean ():.2f}%)")

out_path = os.path.join (RESULTS_DIR, "artifact_mask.npz")
np.savez_compressed (out_path, artifact_mask = artifact_mask, buffer_px = BUFFER_PX)
print (f"Saved -> {out_path}")
