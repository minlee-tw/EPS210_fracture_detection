
# %%
"""
Compare uniform vs length-weighted fracture sampling for RF, XGBoost, SVM.
Trains each model twice (uniform + weighted), saves predictions under
separate filenames, and produces bar-plot comparison.
Also shows recall as a function of fracture length bin.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score,  roc_auc_score,  classification_report,  confusion_matrix
from xgboost import XGBClassifier

DATA_DIR = os.path.dirname (os.path.abspath (__file__))
if not any (f.endswith (".kmz") for f in os.listdir (DATA_DIR)):
    _alt = os.path.join (os.path.expanduser ("~"),  "OneDrive",  "Dokumente",
                        "Academics",  "Coursework",  "Spring 2026",  "AI for EPS")
    if os.path.isdir (_alt) and any (f.endswith (".kmz") for f in os.listdir (_alt)):
        DATA_DIR = _alt

RESULTS_DIR = os.path.join (DATA_DIR,  "results")

# Load ground truth and precomputed features

import zipfile
import xml.etree.ElementTree as ET
from scipy.ndimage import binary_dilation,  uniform_filter,  laplace

KML_NS = "{http://www.opengis.net/kml/2.2}"

def parse_kmz (filepath):
    overlays,  lines = [],  []
    with zipfile.ZipFile (filepath) as zf:
        kml_name = [n for n in zf.namelist () if n.endswith (".kml")] [0]
        with zf.open (kml_name) as kf:
            tree = ET.parse (kf)
        root = tree.getroot ()
        # Ground overlays
        for go in root.iter (f"{KML_NS}GroundOverlay"):
            href = go.find (f".//{KML_NS}href")
            lb = go.find (f".//{KML_NS}LatLonBox")
            if href is not None and lb is not None:
                img_name = href.text.strip ()
                bounds = {d: float (lb.find (f"{KML_NS}{d}").text)
                          for d in ["north",  "south",  "east",  "west"]}
                from PIL import Image
                import io
                with zf.open (img_name) as imgf:
                    img = np.array (Image.open (io.BytesIO (imgf.read ())))
                overlays.append ((img,  bounds))
        # LineStrings
        for pm in root.iter (f"{KML_NS}Placemark"):
            for ls in pm.iter (f"{KML_NS}LineString"):
                coords_text = ls.find (f"{KML_NS}coordinates").text.strip ()
                coords = []
                for pt in coords_text.split ():
                    parts = pt.split (",")
                    coords.append ((float (parts [0]),  float (parts [1])))
                lines.append (np.array (coords))
    return overlays,  lines


def build_grid (img,  bounds):
    h,  w = img.shape [:2]
    lon = np.linspace (bounds ["west"],  bounds ["east"],  w).astype (np.float32)
    lat = np.linspace (bounds ["south"],  bounds ["north"],  h).astype (np.float32)

    r = img [:,  :,  0].astype (np.float32)
    g = img [:,  :,  1].astype (np.float32)
    b = img [:,  :,  2].astype (np.float32)
    a = img [:,  :,  3].astype (np.float32) if img.shape [2] == 4 else np.full ((h,  w),  255,  np.float32)

    nodata = a < 128
    if np.allclose (r,  b,  atol = 5):
        z = r / 127.5 - 1.0
    else:
        z = (r - b) / 255.0

    z [nodata] = np.nan
    # Flip so row 0 = south (origin = "lower" convention matching lat array)
    z = z [::-1]
    nodata = nodata [::-1]
    z_filled = np.nan_to_num (z,  nan = 0.0)
    return lon,  lat,  z,  z_filled,  nodata


def compute_features (z_filled,  nodata):
    H,  W = z_filled.shape
    gy,  gx = np.gradient (z_filled)
    grad_mag = np.sqrt (gx**2 + gy**2).astype (np.float32)
    grad_dir = np.arctan2 (gy,  gx).astype (np.float32)

    from scipy.ndimage import sobel as sobel_filter
    sx = sobel_filter (z_filled,  axis = 1).astype (np.float32)
    sy = sobel_filter (z_filled,  axis = 0).astype (np.float32)
    sobel_mag = np.sqrt (sx**2 + sy**2).astype (np.float32)

    mean3 = uniform_filter (z_filled,  size = 3).astype (np.float32)
    mean5 = uniform_filter (z_filled,  size = 5).astype (np.float32)
    sq3 = uniform_filter (z_filled**2,  size = 3).astype (np.float32)
    sq5 = uniform_filter (z_filled**2,  size = 5).astype (np.float32)
    var3 = np.clip (sq3 - mean3**2,  0,  None).astype (np.float32)
    var5 = np.clip (sq5 - mean5**2,  0,  None).astype (np.float32)
    std3 = np.sqrt (var3)
    std5 = np.sqrt (var5)
    lap = laplace (z_filled).astype (np.float32)

    features = np.stack ([gx.astype (np.float32),  gy.astype (np.float32),
                         grad_mag,  grad_dir,  sx,  sy,  sobel_mag,
                         var3,  std3,  var5,  std5,  mean3,  mean5,  lap],  axis = -1)
    return features


def crop_and_resample (z_src,  lon_src,  lat_src,  lon_ref,  lat_ref):
    """Crop z_src to the extent of the reference grid and resample to its shape."""
    from scipy.ndimage import zoom
    # Find crop indices in source grid
    col_lo = np.searchsorted (lon_src,  lon_ref [0],  side = "left")
    col_hi = np.searchsorted (lon_src,  lon_ref [-1],  side = "right")
    row_lo = np.searchsorted (lat_src,  lat_ref [0],  side = "left")
    row_hi = np.searchsorted (lat_src,  lat_ref [-1],  side = "right")
    # Clamp
    col_lo = max (col_lo - 1,  0)
    row_lo = max (row_lo - 1,  0)
    col_hi = min (col_hi + 1,  len (lon_src))
    row_hi = min (row_hi + 1,  len (lat_src))
    z_crop = z_src [row_lo:row_hi,  col_lo:col_hi]
    H_ref,  W_ref = len (lat_ref),  len (lon_ref)
    zoom_r = H_ref / z_crop.shape [0]
    zoom_c = W_ref / z_crop.shape [1]
    z_out = zoom (np.nan_to_num (z_crop,  nan = 0.0),  (zoom_r,  zoom_c),  order = 1).astype (np.float32)
    # Resample nodata mask (nearest-neighbor)
    nd_crop = np.isnan (z_crop).astype (np.float32)
    nd_out = zoom (nd_crop,  (zoom_r,  zoom_c),  order = 0) > 0.5
    z_out [nd_out] = np.nan
    return z_out,  nd_out


def bresenham_line (r0,  c0,  r1,  c1):
    rows,  cols = [],  []
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
    return np.array (rows),  np.array (cols)


def compute_fracture_lengths (fracture_lines,  ref_lat = 35.7):
    LAT_DEG_TO_M = 111_320.0
    LON_DEG_TO_M = 111_320.0 * np.cos (np.radians (ref_lat))
    lengths = []
    for coords in fracture_lines:
        total = 0.0
        for k in range (len (coords) - 1):
            dlon = (coords [k+1,  0] - coords [k,  0]) * LON_DEG_TO_M
            dlat = (coords [k+1,  1] - coords [k,  1]) * LAT_DEG_TO_M
            total += np.sqrt (dlon**2 + dlat**2)
        lengths.append (total)
    return np.array (lengths)


def rasterize_fractures (fracture_lines,  lon,  lat):
    H,  W = len (lat),  len (lon)
    mask = np.zeros ((H,  W),  dtype = np.uint8)
    length_map = np.zeros ((H,  W),  dtype = np.float32)
    lon_min,  lon_max = lon [0],  lon [-1]
    lat_min,  lat_max = lat [0],  lat [-1]
    frac_lengths = compute_fracture_lengths (fracture_lines)
    struct = np.ones ((3,  3),  dtype = bool)

    for fi,  coords in enumerate (fracture_lines):
        mask_i = np.zeros ((H,  W),  dtype = np.uint8)
        for k in range (len (coords) - 1):
            c0 = int (round ((coords [k,  0] - lon_min) / (lon_max - lon_min) * (W - 1)))
            r0 = int (round ((coords [k,  1] - lat_min) / (lat_max - lat_min) * (H - 1)))
            c1 = int (round ((coords [k+1,  0] - lon_min) / (lon_max - lon_min) * (W - 1)))
            r1 = int (round ((coords [k+1,  1] - lat_min) / (lat_max - lat_min) * (H - 1)))
            c0,  c1 = np.clip ([c0,  c1],  0,  W - 1)
            r0,  r1 = np.clip ([r0,  r1],  0,  H - 1)
            rr,  cc = bresenham_line (r0,  c0,  r1,  c1)
            mask_i [rr,  cc] = 1
        mask_i = binary_dilation (mask_i,  structure = struct).astype (np.uint8)
        new_pixels = (mask_i == 1)
        overwrite = new_pixels & ((length_map == 0) | (frac_lengths [fi] < length_map))
        length_map [overwrite] = frac_lengths [fi]
        mask |= mask_i
    return mask,  length_map


# Load data
print ("Loading KMZ data...")
kmz_files = [f for f in os.listdir (DATA_DIR) if f.endswith (".kmz")]

fracture_lines = None
grid_ew = grid_up = grid_s1 = None
bounds_ew = None

for f in kmz_files:
    overlays,  lines = parse_kmz (os.path.join (DATA_DIR,  f))
    if lines:
        fracture_lines = lines
    if overlays:
        img,  bounds = overlays [0]
        if "EW_high_800" in f:
            bounds_ew = bounds
            grid_ew = build_grid (img,  bounds)
        elif "UP_high_800" in f:
            grid_up = build_grid (img,  bounds)
        elif "phasegradient" in f.lower ():
            # Different extent/resolution — crop and resample after EW is loaded
            _s1_img,  _s1_bounds = img,  bounds
            _s1_raw = build_grid (img,  bounds)
            grid_s1 = _s1_raw  # will be resampled below

lon,  lat,  z_ew,  z_filled_ew,  nodata_ew = grid_ew
H,  W = z_ew.shape
print (f"EW grid shape: {z_ew.shape}")

# UP grid
_,  _,  z_up,  z_filled_up,  nodata_up = grid_up
print (f"UP grid shape: {z_up.shape}")

# S1 phase gradient (crop & resample)
lon_s1,  lat_s1,  z_s1_raw,  _,  _ = grid_s1
z_s1_resampled,  nodata_s1 = crop_and_resample (z_s1_raw,  lon_s1,  lat_s1,  lon,  lat)
z_filled_s1 = np.nan_to_num (z_s1_resampled,  nan = 0.0).astype (np.float32)
print (f"S1 grid resampled to: {z_filled_s1.shape}")

# Combined nodata mask
nodata = nodata_ew | nodata_up | nodata_s1

print ("Computing features (3 grids × 15 features = 45 features)...")
features_ew = compute_features (z_filled_ew,  nodata)
features_up = compute_features (z_filled_up,  nodata)
features_s1 = compute_features (z_filled_s1,  nodata)

print ("Rasterizing fractures...")
fracture_mask,  fracture_length_map = rasterize_fractures (fracture_lines,  lon,  lat)
print (f"Fracture pixels: {fracture_mask.sum ():,  }")

# Save canonical ground truth for future plotting scripts.
_lon64 = np.linspace (bounds_ew ["west"],  bounds_ew ["east"],  W)
_lat64 = np.linspace (bounds_ew ["south"],  bounds_ew ["north"],  H)
_fracture_mask_canonical,  _ = rasterize_fractures (fracture_lines,  _lon64,  _lat64)
_extent = np.array ([_lon64 [0],  _lon64 [-1],  _lat64 [0],  _lat64 [-1]],  dtype = np.float64)
os.makedirs (RESULTS_DIR,  exist_ok = True)
np.savez_compressed (os.path.join (RESULTS_DIR,  "ground_truth.npz"),
                    fracture_mask = _fracture_mask_canonical,
                    extent = _extent,
                    displacement = z_ew,
                    grid_name = "EW_high_800")
print (f"Saved ground truth ({_fracture_mask_canonical.sum ():,  } fracture px) "
      f"-> {RESULTS_DIR}/ground_truth.npz")

# Build 45 features: [EW disp + 14 feat,  UP disp + 14 feat,  S1 disp + 14 feat]
X_ew = np.concatenate ([z_filled_ew [:,  :,  np.newaxis],  features_ew],  axis = -1)
X_up = np.concatenate ([z_filled_up [:,  :,  np.newaxis],  features_up],  axis = -1)
X_s1 = np.concatenate ([z_filled_s1 [:,  :,  np.newaxis],  features_s1],  axis = -1)
X_all = np.concatenate ([X_ew,  X_up,  X_s1],  axis = -1)
n_feat_total = X_all.shape [-1]
print (f"Total features: {n_feat_total}")
X_flat = X_all.reshape (-1,  n_feat_total)
y_flat = fracture_mask.reshape (-1)
nodata_flat = nodata.reshape (-1)
valid_mask = ~nodata_flat

# Exclude artifact pixels
_artifact_file = os.path.join (RESULTS_DIR,  "artifact_mask.npz")
if os.path.exists (_artifact_file):
    artifact_mask = np.load (_artifact_file) ["artifact_mask"].ravel ()
    valid_mask = valid_mask & (artifact_mask == 0)
    print (f"Artifact mask applied: {artifact_mask.sum ():,  } pixels excluded")
else:
    artifact_mask = np.zeros_like (nodata_flat)
    print ("No artifact mask found,  proceeding without")

X_flat = np.nan_to_num (X_flat,  nan = 0.0,  posinf = 0.0,  neginf = 0.0)

frac_idx = np.where ((y_flat == 1) & valid_mask) [0]
nonfrac_idx = np.where ((y_flat == 0) & valid_mask) [0]
print (f"Valid fracture pixels: {len (frac_idx):,  }")
print (f"Valid non-fracture pixels: {len (nonfrac_idx):,  }")

# Fracture length info for weighting
frac_length_flat = fracture_length_map.reshape (-1)
frac_lengths_at_idx = frac_length_flat [frac_idx]

# Define sampling strategies
max_pos = 50000

def sample_uniform (rng):
    if len (frac_idx) > max_pos:
        fs = rng.choice (frac_idx,  size = max_pos,  replace = False)
    else:
        fs = frac_idx.copy ()
    n_neg = min (len (nonfrac_idx),  3 * len (fs))
    ns = rng.choice (nonfrac_idx,  size = n_neg,  replace = False)
    idx = np.concatenate ([fs,  ns])
    rng.shuffle (idx)
    return idx

def sample_weighted (rng):
    weights = 1.0 / np.sqrt (np.maximum (frac_lengths_at_idx,  1.0))
    weights /= weights.sum ()
    fs = rng.choice (frac_idx,  size = min (max_pos,  len (frac_idx)),
                    replace = False,  p = weights)
    n_neg = min (len (nonfrac_idx),  3 * len (fs))
    ns = rng.choice (nonfrac_idx,  size = n_neg,  replace = False)
    idx = np.concatenate ([fs,  ns])
    rng.shuffle (idx)
    return idx

# Train & predict for both strategies
def predict_chunked (model,  X,  chunk_size = 500000,  scaler = None):
    preds = []
    for i in range (0,  len (X),  chunk_size):
        chunk = X [i:i+chunk_size]
        if scaler is not None:
            chunk = scaler.transform (chunk)
        preds.append (model.predict (chunk))
    return np.concatenate (preds)

strategies = {
    "uniform":  sample_uniform,
    "weighted": sample_weighted,
}

extent = [lon [0],  lon [-1],  lat [0],  lat [-1]]
all_results = {}

for strat_name,  sampler in strategies.items ():
    print (f"\n{" = "*70}")
    print (f"  Strategy: {strat_name}")
    print (f"{" = "*70}")

    rng = np.random.RandomState (42)
    idx = sampler (rng)
    X_sampled = X_flat [idx]
    y_sampled = y_flat [idx]

    X_train,  X_test,  y_train,  y_test = train_test_split (
        X_sampled,  y_sampled,  test_size = 0.3,  random_state = 42,  stratify = y_sampled
    )
    print (f"Train: {len (y_train):,  }  Test: {len (y_test):,  }  "
          f"Pos: {y_sampled.sum ():,  } ({100*y_sampled.mean ():.1f}%)")

    scaler = StandardScaler ()
    X_train_sc = scaler.fit_transform (X_train)
    X_test_sc = scaler.transform (X_test)

    model_defs = [
        ("Random Forest",  "unscaled",  RandomForestClassifier (
            n_estimators = 200,  max_depth = 15,  min_samples_leaf = 5,
            class_weight = "balanced",  random_state = 42,  n_jobs = -1)),
        ("XGBoost",  "unscaled",  XGBClassifier (
            n_estimators = 200,  max_depth = 8,  learning_rate = 0.1,
            scale_pos_weight = (y_train == 0).sum () / max ((y_train == 1).sum (),  1),
            eval_metric = "logloss",  random_state = 42,  n_jobs = -1,
            use_label_encoder = False)),
        ("SVM (Linear)",  "scaled",  None),   # built below
    ]

    strat_results = {}

    for mname,  scale_type,  model in model_defs:
        print (f"  Training {mname}...")
        X_tr = X_train_sc if scale_type == "scaled" else X_train

        if mname == "SVM (Linear)":
            lsvm = LinearSVC (C = 1.0,  class_weight = "balanced",  max_iter = 2000,  random_state = 42)
            lsvm.fit (X_tr,  y_train)
            model = CalibratedClassifierCV (lsvm,  cv = 3,  method = "sigmoid")
            model.fit (X_tr,  y_train)
        else:
            model.fit (X_tr,  y_train)

        # Full-grid prediction (scale in separate pieces for SVM to save memory)
        sc = scaler if scale_type == "scaled" else None
        pred_full = predict_chunked (model,  X_flat,  scaler = sc).reshape (H,  W)
        pred_full [artifact_mask.reshape (H,  W) == 1] = 0  # exclude artifacts

        # Full-grid metrics (excluding artifact pixels)
        eval_mask = (artifact_mask == 0)
        pf = pred_full.ravel ()
        TP = int (((y_flat == 1) & (pf == 1) & eval_mask).sum ())
        TN = int (((y_flat == 0) & (pf == 0) & eval_mask).sum ())
        FP = int (((y_flat == 0) & (pf == 1) & eval_mask).sum ())
        FN = int (((y_flat == 1) & (pf == 0) & eval_mask).sum ())

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        bal_acc   = (recall + specificity) / 2.0
        iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        strat_results [mname] = {
            "pred_map": pred_full,
            "TP": TP,  "TN": TN,  "FP": FP,  "FN": FN,
            "Precision": precision,  "Recall": recall,  "F1": f1,
            "Specificity": specificity,  "FPR": fpr,
            "Balanced Accuracy": bal_acc,  "IoU": iou,
        }

        print (f"    TP = {TP:>7,  } TN = {TN:>10,  } FP = {FP:>10,  } FN = {FN:>6,  }  "
              f"F1 = {f1:.4f} Recall = {recall:.4f} Prec = {precision:.4f}")

        # Save prediction
        safe = mname.lower ().replace (" ",  "_").replace ("(",  "").replace (")",  "")
        np.savez_compressed (
            os.path.join (RESULTS_DIR,  f"pred_{safe}_{strat_name}.npz"),
            pred_map = pred_full,  model_name = mname,  strategy = strat_name)

        # Save trained model (and scaler if SVM)
        models_dir = os.path.join (DATA_DIR,  "models")
        os.makedirs (models_dir,  exist_ok = True)
        joblib.dump (model,  os.path.join (models_dir,  f"{safe}_{strat_name}.joblib"))
        if scale_type == "scaled":
            joblib.dump (scaler,  os.path.join (models_dir,  f"scaler_{strat_name}.joblib"))

    all_results [strat_name] = strat_results

# Also save the "winning" weighted predictions as the default pred files
for mname,  res in all_results ["weighted"].items ():
    safe = mname.lower ().replace (" ",  "_").replace ("(",  "").replace (")",  "")
    np.savez_compressed (
        os.path.join (RESULTS_DIR,  f"pred_{safe}.npz"),
        pred_map = res ["pred_map"],  model_name = mname)

print (f"\nSaved all predictions to {RESULTS_DIR}/")

# Recall by fracture length bin
print (f"\n{" = "*70}")
print ("  Recall by fracture length bin (full grid)")
print (f"{" = "*70}")

bins_m = [0,  500,  1000,  2000,  5000,  np.inf]
bin_labels = ["<500m",  "500m-1km",  "1-2km",  "2-5km",  ">5km"]

model_names = ["Random Forest",  "XGBoost",  "SVM (Linear)"]

# Header
hdr = f"{"Bin":<12s}"
for mn in model_names:
    hdr += f"  {mn+"(U)":>16s}  {mn+"(W)":>16s}"
print (hdr)
print ("-" * len (hdr))

bin_recall = {s: {m: [] for m in model_names} for s in strategies}

for bi in range (len (bins_m) - 1):
    lo,  hi = bins_m [bi],  bins_m [bi+1]
    # Pixels belonging to fractures in this length bin
    in_bin = (frac_length_flat >= lo) & (frac_length_flat < hi) & (y_flat == 1)
    n_bin = in_bin.sum ()
    if n_bin == 0:
        continue

    row = f"{bin_labels [bi]:<12s}"
    for mn in model_names:
        for sn,  label in [ ("uniform",  "U"),  ("weighted",  "W")]:
            pred = all_results [sn] [mn] ["pred_map"].ravel ()
            tp_bin = ((pred == 1) & in_bin).sum ()
            rec_bin = tp_bin / n_bin if n_bin > 0 else 0.0
            bin_recall [sn] [mn].append (rec_bin)
            row += f"  {rec_bin:>15.4f} "
    row += f"  ({n_bin:>6,  } px)"
    print (row)

# Side-by-side bar plots
metric_names = ["Precision",  "Recall",  "F1",  "Specificity",  "FPR",
                "Balanced Accuracy",  "IoU"]

fig,  axes = plt.subplots (2,  4,  figsize = (24,  10))
axes = axes.ravel ()

x = np.arange (len (model_names))
w = 0.35

colors_u = ["#7bafd4",  "#ffb87a",  "#7ec87e"]  # lighter
colors_w = ["#1f77b4",  "#ff7f0e",  "#2ca02c"]  # darker

for i,  metric in enumerate (metric_names):
    ax = axes [i]
    vals_u = [all_results ["uniform"] [m] [metric] for m in model_names]
    vals_w = [all_results ["weighted"] [m] [metric] for m in model_names]

    bars_u = ax.bar (x - w/2,  vals_u,  width = w,  label = "Uniform",  color = colors_u,
                    edgecolor = "white")
    bars_w = ax.bar (x + w/2,  vals_w,  width = w,  label = "Weighted",  color = colors_w,
                    edgecolor = "white")

    for bar,  v in zip (bars_u,  vals_u):
        ax.text (bar.get_x () + bar.get_width ()/2,  bar.get_height () + 0.002,
                f"{v:.4f}",  ha = "center",  va = "bottom",  fontsize = 7)
    for bar,  v in zip (bars_w,  vals_w):
        ax.text (bar.get_x () + bar.get_width ()/2,  bar.get_height () + 0.002,
                f"{v:.4f}",  ha = "center",  va = "bottom",  fontsize = 7)

    ax.set_title (metric,  fontsize = 12,  fontweight = "bold")
    ax.set_xticks (x)
    ax.set_xticklabels (model_names,  fontsize = 9)
    ax.set_ylim (0,  min (max (max (vals_u),  max (vals_w)) * 1.25,  1.05))
    ax.grid (axis = "y",  alpha = 0.3)
    if i == 0:
        ax.legend (fontsize = 9)

# Recall by length bin plot
ax = axes [7]
bin_x = np.arange (len (bin_labels))
bw = 0.12
offsets = [-2.5,  -1.5,  -0.5,  0.5,  1.5,  2.5]
bar_colors_all = [colors_u [0],  colors_w [0],  colors_u [1],  colors_w [1],
                  colors_u [2],  colors_w [2]]
bar_labels_all = ["RF (U)",  "RF (W)",  "XGB (U)",  "XGB (W)",
                  "SVM (U)",  "SVM (W)"]

for mi,  mn in enumerate (model_names):
    for si,  sn in enumerate (["uniform",  "weighted"]):
        idx = mi * 2 + si
        vals = bin_recall [sn] [mn]
        ax.bar (bin_x + offsets [idx] * bw,  vals,  width = bw,
               color = bar_colors_all [idx],  label = bar_labels_all [idx],
               edgecolor = "white")

ax.set_title ("Recall by Fracture Length",  fontsize = 12,  fontweight = "bold")
ax.set_xticks (bin_x)
ax.set_xticklabels (bin_labels,  fontsize = 9)
ax.set_ylabel ("Recall")
ax.set_ylim (0,  1.05)
ax.legend (fontsize = 7,  ncol = 3)
ax.grid (axis = "y",  alpha = 0.3)

plt.suptitle ("Uniform vs Length-Weighted Sampling — Full Grid Metrics",
             fontsize = 15,  fontweight = "bold")
plt.tight_layout ()
plt.savefig (os.path.join (RESULTS_DIR,  "sampling_comparison.png"),
            dpi = 200,  bbox_inches = "tight")
print (f"\nSaved comparison plot -> {RESULTS_DIR}/sampling_comparison.png")
plt.show ()
