
# %%
"""
Combined 2x2 figure:
  Top left:  Fracture length distribution (log scale)
  Top right: Cumulative pixel contribution by length
  Bottom left:  Recall by fracture length bin (uniform vs weighted)
  Bottom right: Recall change (weighted - uniform) by fracture length bin
"""

import os
import zipfile
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

DATA_DIR = os.path.dirname (os.path.abspath (__file__))
if not any (f.endswith (".kmz") for f in os.listdir (DATA_DIR)):
    _alt = os.path.join (os.path.expanduser ("~"), "OneDrive", "Dokumente",
                        "Academics", "Coursework", "Spring 2026", "AI for EPS")
    if os.path.isdir (_alt) and any (f.endswith (".kmz") for f in os.listdir (_alt)):
        DATA_DIR = _alt

RESULTS_DIR = os.path.join (DATA_DIR, "results")
KML_NS = "{http://www.opengis.net/kml/2.2}"

# Parse fracture LineStrings
frac_file = None
for f in os.listdir (DATA_DIR):
    if f.endswith (".kmz") and "fracture" in f.lower ():
        frac_file = os.path.join (DATA_DIR, f)
        break

fracture_lines = []
with zipfile.ZipFile (frac_file) as zf:
    kml_name = [n for n in zf.namelist () if n.endswith (".kml")] [0]
    with zf.open (kml_name) as kf:
        tree = ET.parse (kf)
    for pm in tree.getroot ().iter (f"{KML_NS}Placemark"):
        for ls in pm.iter (f"{KML_NS}LineString"):
            coords_text = ls.find (f"{KML_NS}coordinates").text.strip ()
            coords = []
            for pt in coords_text.split ():
                parts = pt.split (",")
                coords.append ((float (parts [0]), float (parts [1])))
            fracture_lines.append (np.array (coords))

# Compute fracture lengths
LAT_DEG_TO_M = 111_320.0
LON_DEG_TO_M = 111_320.0 * np.cos (np.radians (35.7))

lengths_m = []
for coords in fracture_lines:
    total = 0.0
    for k in range (len (coords) - 1):
        dlon = (coords [k+1, 0] - coords [k, 0]) * LON_DEG_TO_M
        dlat = (coords [k+1, 1] - coords [k, 1]) * LAT_DEG_TO_M
        total += np.sqrt (dlon**2 + dlat**2)
    lengths_m.append (total)
lengths_m = np.array (lengths_m)

# Rasterize each fracture individually for pixel counts
gt = np.load (os.path.join (RESULTS_DIR, "ground_truth.npz"))
fracture_mask = gt ["fracture_mask"]
extent = gt ["extent"]
H, W = fracture_mask.shape
lon_min, lon_max, lat_min, lat_max = extent

def bresenham_line (r0, c0, r1, c1):
    rows, cols = [], []
    dr = abs (r1 - r0); dc = abs (c1 - c0)
    sr = 1 if r0 < r1 else -1; sc = 1 if c0 < c1 else -1
    err = dr - dc
    while True:
        rows.append (r0); cols.append (c0)
        if r0 == r1 and c0 == c1: break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r0 += sr
        if e2 < dr:  err += dr; c0 += sc
    return np.array (rows), np.array (cols)

struct = np.ones ((3, 3), dtype = bool)
pixels_per_fracture = []
frac_length_map = np.zeros ((H, W), dtype = np.float32)

for fi, coords in enumerate (fracture_lines):
    mask_i = np.zeros ((H, W), dtype = np.uint8)
    for k in range (len (coords) - 1):
        c0 = int (round ((coords [k, 0] - lon_min) / (lon_max - lon_min) * (W - 1)))
        r0 = int (round ((coords [k, 1] - lat_min) / (lat_max - lat_min) * (H - 1)))
        c1 = int (round ((coords [k+1, 0] - lon_min) / (lon_max - lon_min) * (W - 1)))
        r1 = int (round ((coords [k+1, 1] - lat_min) / (lat_max - lat_min) * (H - 1)))
        c0, c1 = np.clip ([c0, c1], 0, W - 1)
        r0, r1 = np.clip ([r0, r1], 0, H - 1)
        rr, cc = bresenham_line (r0, c0, r1, c1)
        mask_i [rr, cc] = 1
    mask_i = binary_dilation (mask_i, structure = struct).astype (np.uint8)
    pixels_per_fracture.append (int (mask_i.sum ()))
    new_pixels = (mask_i == 1)
    overwrite = new_pixels & ((frac_length_map == 0) | (lengths_m [fi] < frac_length_map))
    frac_length_map [overwrite] = lengths_m [fi]

pixels_per_fracture = np.array (pixels_per_fracture)

# Load uniform and weighted predictions
model_names = ["Random Forest", "XGBoost", "SVM (Linear)"]
safe_names = ["random_forest", "xgboost", "svm_linear"]

preds_uniform = {}
preds_weighted = {}
for mname, sname in zip (model_names, safe_names):
    preds_uniform [mname] = np.load (
        os.path.join (RESULTS_DIR, f"pred_{sname}_uniform.npz")) ["pred_map"]
    preds_weighted [mname] = np.load (
        os.path.join (RESULTS_DIR, f"pred_{sname}_weighted.npz")) ["pred_map"]

# Compute recall by length bin
bins_m = [0, 500, 1000, 2000, 5000, np.inf]
bin_labels = ["<500 m", "0.5-1 km", "1-2 km", "2-5 km", ">5 km"]
n_bins = len (bin_labels)

frac_length_flat = frac_length_map.ravel ()
y_flat = fracture_mask.ravel ()

recall_data = {}
for mname in model_names:
    recall_data [mname] = {}
    for strat, preds_dict in [ ("uniform", preds_uniform), ("weighted", preds_weighted)]:
        pred_flat = preds_dict [mname].ravel ()
        recs = []
        for bi in range (n_bins):
            lo, hi = bins_m [bi], bins_m [bi + 1]
            in_bin = (frac_length_flat >= lo) & (frac_length_flat < hi) & (y_flat == 1)
            n_bin = in_bin.sum ()
            if n_bin > 0:
                tp = ((pred_flat == 1) & in_bin).sum ()
                recs.append (tp / n_bin)
            else:
                recs.append (0.0)
        recall_data [mname] [strat] = np.array (recs)

# Combined 2x2 figure
colors_model = ["#1f77b4", "#ff7f0e", "#2ca02c"]
x = np.arange (n_bins)
bar_w = 0.25

fig, axes = plt.subplots (2, 2, figsize = (16, 12))
_labels = ["(a)", "(b)", "(c)", "(d)"]
_axes_flat = [axes [0, 0], axes [0, 1], axes [1, 0], axes [1, 1]]
for _ax, _lbl in zip (_axes_flat, _labels):
    _ax.text (0.02, 0.98, _lbl, transform = _ax.transAxes, fontsize = 18,
             fontweight = "bold", va = "top", ha = "left")

# Top left: Log-scale histogram of fracture lengths
ax = axes [0, 0]
log_bins = np.logspace (np.log10 (max (lengths_m.min (), 1)), np.log10 (lengths_m.max ()), 40)
ax.hist (lengths_m, bins = log_bins, color = "#2ca02c", edgecolor = "white")
ax.set_xscale ("log")
ax.set_xlabel ("Fracture length (m)", fontsize = 15)
ax.set_ylabel ("Count", fontsize = 15)
ax.set_title ("Distribution of GT Fracture Lengths (log scale)",
             fontsize = 18, fontweight = "bold")
ax.axvline (np.median (lengths_m), color = "red", ls = "--", lw = 1.5,
           label = f"Median = {np.median (lengths_m):.0f} m")
ax.legend (fontsize = 15)
ax.tick_params (labelsize = 15)

# Top right: Cumulative pixel contribution
ax = axes [0, 1]
sort_idx = np.argsort (lengths_m)
sorted_lengths = lengths_m [sort_idx]
sorted_pixels = pixels_per_fracture [sort_idx]
cum_pixels = np.cumsum (sorted_pixels) / sorted_pixels.sum () * 100

ax.plot (sorted_lengths, cum_pixels, color = "#d62728", lw = 2)
ax.set_xlabel ("Fracture length (m)", fontsize = 15)
ax.set_ylabel ("Cumulative % of fracture pixels", fontsize = 15)
ax.set_title ("Cumulative Pixel Contribution by Length",
             fontsize = 18, fontweight = "bold")
ax.axhline (50, color = "grey", ls = ":", alpha = 0.7)
ax.set_xscale ("log")
ax.grid (alpha = 0.3)
ax.tick_params (labelsize = 15)

# Bottom left: Recall by bin (uniform vs weighted)
ax = axes [1, 0]
for mi, mname in enumerate (model_names):
    offset = (mi - 1) * bar_w
    rec_u = recall_data [mname] ["uniform"]
    rec_w = recall_data [mname] ["weighted"]
    ax.bar (x + offset - bar_w/4, rec_u, width = bar_w/2, color = colors_model [mi],
           alpha = 0.4, edgecolor = colors_model [mi], label = f"{mname} (uniform)")
    ax.bar (x + offset + bar_w/4, rec_w, width = bar_w/2, color = colors_model [mi],
           alpha = 1.0, edgecolor = "white", label = f"{mname} (weighted)")

ax.set_xticks (x)
ax.set_xticklabels (bin_labels, fontsize = 15)
ax.set_xlabel ("Ground Truth Fracture Length", fontsize = 15)
ax.set_ylabel ("Recall = TP / (TP + FN)", fontsize = 15)
ax.set_title ("Recall by Fracture Length Bin", fontsize = 18, fontweight = "bold")
ax.legend (fontsize = 10.5, ncol = 2)
ax.grid (axis = "y", alpha = 0.3)
ax.tick_params (labelsize = 15)

# Bottom right: Recall change (weighted - uniform)
ax = axes [1, 1]
_delta_texts = []  # collect (x, y, text) for overlap avoidance
for mi, mname in enumerate (model_names):
    delta = recall_data [mname] ["weighted"] - recall_data [mname] ["uniform"]
    ax.bar (x + (mi - 1) * bar_w, delta, width = bar_w, color = colors_model [mi],
           edgecolor = "white", label = mname)
    for xi, d in zip (x + (mi - 1) * bar_w, delta):
        # Place all labels above bars (above zero line)
        y_pos = max (d, 0) + 0.002
        _delta_texts.append ((xi, y_pos, f"{d:+.3f}"))

# Sort by x position within each bin group, then adjust vertical spacing
_delta_texts.sort (key = lambda t: (round (t [0]), -t [1]))
_min_gap = 0.008  # minimum vertical gap between labels
for i in range (len (_delta_texts)):
    xi, yi, txt = _delta_texts [i]
    # Check all previous labels in same bin (within bar_w * 2 x-distance)
    for j in range (i):
        xj, yj, _ = _delta_texts [j]
        if abs (xi - xj) < bar_w * 1.5 and abs (yi - yj) < _min_gap:
            yi = yj + _min_gap
            _delta_texts [i] = (xi, yi, txt)
    ax.text (xi, yi, txt, ha = "center", va = "bottom",
            fontsize = 10.5, fontweight = "bold")

ax.axhline (0, color = "black", lw = 0.8)
ax.set_xticks (x)
ax.set_xticklabels (bin_labels, fontsize = 15)
ax.set_xlabel ("Ground Truth Fracture Length", fontsize = 15)
ax.set_ylabel (r"$\Delta$ Recall (weighted $-$ uniform)", fontsize = 15)
ax.set_title ("Effect of Length-Weighted Sampling on Recall",
             fontsize = 18, fontweight = "bold")
ax.legend (fontsize = 13.5)
ax.grid (axis = "y", alpha = 0.3)
ax.tick_params (labelsize = 15)

plt.suptitle ("Fracture Length Analysis & Length-Weighted Sampling Impact",
             fontsize = 22, fontweight = "bold")
plt.tight_layout ()
plt.savefig (os.path.join (RESULTS_DIR, "combined_length_figure.png"),
            dpi = 200, bbox_inches = "tight")
print (f"Saved -> {RESULTS_DIR}/combined_length_figure.png")
plt.show ()
