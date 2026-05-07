
# %%
"""
Augmented Autoencoder Anomaly Detection for Fracture Mapping
Same anomaly-detection approach as autoencoder_fracture.py, but the training
data is augmented with geometric transforms (rotations, flips) and additive
Gaussian noise applied to the displacement grid *before* feature extraction.
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter, sobel, binary_dilation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Parse KMZ files

KML_NS = "{http://www.opengis.net/kml/2.2}"
DATA_DIR = os.path.dirname (os.path.abspath (__file__))

if not any (f.endswith (".kmz") for f in os.listdir (DATA_DIR)):
    _alt = os.path.join (os.path.expanduser ("~"),
                        "OneDrive", "Dokumente", "Academics", "Coursework",
                        "Spring 2026", "AI for EPS")
    if os.path.isdir (_alt) and any (f.endswith (".kmz") for f in os.listdir (_alt)):
        DATA_DIR = _alt
        print (f"No KMZ files next to script — using {DATA_DIR}")

def parse_ground_overlay (kml_root, zf):
    overlays = []
    for go in kml_root.iter (f"{KML_NS}GroundOverlay"):
        icon = go.find (f"{KML_NS}Icon")
        href = icon.find (f"{KML_NS}href").text if icon is not None else None
        lb = go.find (f".//{KML_NS}LatLonBox")
        if lb is not None and href is not None:
            bbox = {
                "north": float (lb.find (f"{KML_NS}north").text),
                "south": float (lb.find (f"{KML_NS}south").text),
                "east": float (lb.find (f"{KML_NS}east").text),
                "west": float (lb.find (f"{KML_NS}west").text),
            }
            img_data = zf.read (href)
            img = Image.open (BytesIO (img_data))
            overlays.append ({"image": img, "bbox": bbox})
    return overlays

def parse_placemarks (kml_root):
    lines = []
    for pm in kml_root.iter (f"{KML_NS}Placemark"):
        for ls in pm.iter (f"{KML_NS}LineString"):
            coords_text = ls.find (f"{KML_NS}coordinates").text.strip ()
            coords = []
            for pt in coords_text.split ():
                parts = pt.split (",")
                lon, lat = float (parts [0]), float (parts [1])
                coords.append ((lon, lat))
            lines.append (np.array (coords))
    return lines

kmz_files = sorted ([f for f in os.listdir (DATA_DIR) if f.endswith (".kmz")])
if not kmz_files:
    raise SystemExit (f"ERROR: No .kmz files found in {DATA_DIR}")
print (f"Found {len (kmz_files)} KMZ files:")
for f in kmz_files:
    print (f"  - {f}")

all_overlays = {}
all_lines = {}

for kmz_file in kmz_files:
    path = os.path.join (DATA_DIR, kmz_file)
    with zipfile.ZipFile (path, "r") as zf:
        kml_names = [n for n in zf.namelist () if n.endswith (".kml")]
        for kml_name in kml_names:
            kml_content = zf.read (kml_name)
            root = ET.fromstring (kml_content)
            overlays = parse_ground_overlay (root, zf)
            if overlays:
                all_overlays [kmz_file] = overlays
                print (f"{kmz_file}: {len (overlays)} ground overlay (s)")
            lines = parse_placemarks (root)
            if lines:
                all_lines [kmz_file] = lines
                print (f"{kmz_file}: {len (lines)} line feature (s)")

# Build displacement grid

grids = {}
for kmz_name, overlays in all_overlays.items ():
    name = kmz_name.replace (".kmz", "")
    for ov in overlays:
        bbox = ov ["bbox"]
        img_arr = np.array (ov ["image"])
        H_img, W_img = img_arr.shape [:2]
        lon = np.linspace (bbox ["west"], bbox ["east"], W_img)
        lat = np.linspace (bbox ["south"], bbox ["north"], H_img)

        r = img_arr [:, :, 0].astype (np.float32)
        b = img_arr [:, :, 2].astype (np.float32)

        if np.allclose (r, b):
            z = r / 127.5 - 1.0
        else:
            z = (r - b) / 255.0

        if img_arr.shape [2] >= 4:
            alpha = img_arr [:, :, 3]
            z [alpha == 0] = np.nan

        z = z [::-1]
        grids [name] = {"lon": lon, "lat": lat, "z": z}
        print (f"  {kmz_name}: shape = {z.shape}, range = [{np.nanmin (z):.2f}, {np.nanmax (z):.2f}]")

grid_name = "EW_high_800" if "EW_high_800" in grids else next (iter (grids), None)
if grid_name is None:
    raise SystemExit ("ERROR: No displacement grids were built.")
grid = grids [grid_name]
print (f"Reference grid: {grid_name}")

up_name = "UP_high_800" if "UP_high_800" in grids else None
s1_name = next ((n for n in grids if "phasegradient" in n.lower ()), None)
grid_up = grids [up_name] if up_name else None
grid_s1 = grids [s1_name] if s1_name else None
print (f"UP grid: {up_name}, S1 grid: {s1_name}")

def crop_and_resample (z_src, lon_src, lat_src, lon_ref, lat_ref):
    """Crop z_src to the extent of the reference grid and resample to its shape."""
    from scipy.ndimage import zoom
    col_lo = max (np.searchsorted (lon_src, lon_ref [0], side = "left") - 1, 0)
    col_hi = min (np.searchsorted (lon_src, lon_ref [-1], side = "right") + 1, len (lon_src))
    row_lo = max (np.searchsorted (lat_src, lat_ref [0], side = "left") - 1, 0)
    row_hi = min (np.searchsorted (lat_src, lat_ref [-1], side = "right") + 1, len (lat_src))
    z_crop = z_src [row_lo:row_hi, col_lo:col_hi]
    H_ref, W_ref = len (lat_ref), len (lon_ref)
    zoom_r = H_ref / z_crop.shape [0]
    zoom_c = W_ref / z_crop.shape [1]
    z_out = zoom (np.nan_to_num (z_crop, nan = 0.0), (zoom_r, zoom_c), order = 1).astype (np.float32)
    nd_crop = np.isnan (z_crop).astype (np.float32)
    nd_out = zoom (nd_crop, (zoom_r, zoom_c), order = 0) > 0.5
    z_out [nd_out] = np.nan
    return z_out, nd_out

# Spatial feature computation

feature_order = [
    "grad_x", "grad_y", "grad_magnitude", "grad_direction",
    "sobel_x", "sobel_y", "sobel_magnitude",
    "variance_3x3", "std_3x3", "variance_5x5", "std_5x5",
    "mean_3x3", "mean_5x5", "laplacian",
]

def compute_spatial_features (z):
    z = z.astype (np.float32)
    features = {}
    grad_y, grad_x = np.gradient (z)
    features ["grad_x"] = grad_x
    features ["grad_y"] = grad_y
    features ["grad_magnitude"] = np.sqrt (grad_x**2 + grad_y**2)
    features ["grad_direction"] = np.arctan2 (grad_y, grad_x)

    sobel_x = sobel (z, axis = 1)
    sobel_y = sobel (z, axis = 0)
    features ["sobel_x"] = sobel_x
    features ["sobel_y"] = sobel_y
    features ["sobel_magnitude"] = np.sqrt (sobel_x**2 + sobel_y**2)

    for win_size in [3, 5]:
        local_mean = uniform_filter (z, size = win_size, mode = "nearest")
        local_mean_sq = uniform_filter (z**2, size = win_size, mode = "nearest")
        local_var = np.clip (local_mean_sq - local_mean**2, 0, None)
        features [f"variance_{win_size}x{win_size}"] = local_var
        features [f"std_{win_size}x{win_size}"] = np.sqrt (local_var)

    features ["mean_3x3"] = uniform_filter (z, size = 3, mode = "nearest")
    features ["mean_5x5"] = uniform_filter (z, size = 5, mode = "nearest")

    laplacian = (np.roll (z, 1, axis = 0) + np.roll (z, -1, axis = 0) +
                 np.roll (z, 1, axis = 1) + np.roll (z, -1, axis = 1) - 4 * z)
    features ["laplacian"] = laplacian
    return features

def grid_to_feature_matrix_single (z_filled):
    """Compute features for a single grid and return (N, 15) matrix.
    Builds 2D features one at a time to save meory."""
    feats = compute_spatial_features (z_filled)
    H, W = z_filled.shape
    N = H * W
    X = np.empty ((N, 1 + len (feature_order)), dtype = np.float32)
    X [:, 0] = z_filled.ravel ()
    for i, f in enumerate (feature_order):
        X [:, i + 1] = feats [f].astype (np.float32).ravel ()
    return X

def grid_to_feature_matrix (z_ew, z_up, z_s1):
    """Compute features for 3 grids and return (N, 45) matrix."""
    X_ew = grid_to_feature_matrix_single (z_ew)
    X_up = grid_to_feature_matrix_single (z_up)
    X_s1 = grid_to_feature_matrix_single (z_s1)
    return np.hstack ([X_ew, X_up, X_s1])

# EW grid
z_orig_ew = grid ["z"]
nodata_ew = np.isnan (z_orig_ew)
z_filled_ew = np.where (nodata_ew, np.float32 (0.0), z_orig_ew).astype (np.float32)
grid ["z_filled"] = z_filled_ew

# UP grid
z_orig_up = grid_up ["z"]
nodata_up = np.isnan (z_orig_up)
z_filled_up = np.where (nodata_up, np.float32 (0.0), z_orig_up).astype (np.float32)

# S1 phase gradient (crop & resample)
z_s1_resampled, nodata_s1 = crop_and_resample (
    grid_s1 ["z"], grid_s1 ["lon"], grid_s1 ["lat"], grid ["lon"], grid ["lat"])
z_filled_s1 = np.nan_to_num (z_s1_resampled, nan = 0.0).astype (np.float32)
print (f"S1: resampled to {z_filled_s1.shape}")

# Combined nodata
nodata = nodata_ew | nodata_up | nodata_s1
nodata_flat = nodata.reshape (-1)

feature_names = (["EW_" + n for n in ["displacement"] + feature_order]
                 + ["UP_" + n for n in ["displacement"] + feature_order]
                 + ["S1_" + n for n in ["displacement"] + feature_order])
n_feat_total = len (feature_names)

# Original feature matrix (used for test set and full-grid prediction)
# Build without taking too much memory, one grid at a time
H, W = z_filled_ew.shape
N = H * W
X_flat_orig = np.empty ((N, n_feat_total), dtype = np.float32)
_tmp = grid_to_feature_matrix_single (z_filled_ew)
X_flat_orig [:, :15] = _tmp; del _tmp
_tmp = grid_to_feature_matrix_single (z_filled_up)
X_flat_orig [:, 15:30] = _tmp; del _tmp
_tmp = grid_to_feature_matrix_single (z_filled_s1)
X_flat_orig [:, 30:45] = _tmp; del _tmp
np.nan_to_num (X_flat_orig, copy = False, nan = 0.0, posinf = 0.0, neginf = 0.0)
print (f"Original feature matrix: {X_flat_orig.shape} ({n_feat_total} features)")

# Rasterize fracture labels

def bresenham_line (r0, c0, r1, c1):
    rows, cols = [], []
    dr = abs (r1 - r0)
    dc = abs (c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    while True:
        rows.append (r0)
        cols.append (c0)
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc
    return np.array (rows), np.array (cols)

def rasterize_fractures (fracture_lines, lon, lat):
    H, W = len (lat), len (lon)
    mask = np.zeros ((H, W), dtype = np.uint8)
    lon_min, lon_max = lon [0], lon [-1]
    lat_min, lat_max = lat [0], lat [-1]
    for coords in fracture_lines:
        for k in range (len (coords) - 1):
            c0 = int (round ((coords [k, 0] - lon_min) / (lon_max - lon_min) * (W - 1)))
            r0 = int (round ((coords [k, 1] - lat_min) / (lat_max - lat_min) * (H - 1)))
            c1 = int (round ((coords [k+1, 0] - lon_min) / (lon_max - lon_min) * (W - 1)))
            r1 = int (round ((coords [k+1, 1] - lat_min) / (lat_max - lat_min) * (H - 1)))
            c0, c1 = np.clip ([c0, c1], 0, W - 1)
            r0, r1 = np.clip ([r0, r1], 0, H - 1)
            rr, cc = bresenham_line (r0, c0, r1, c1)
            mask [rr, cc] = 1
    struct = np.ones ((3, 3), dtype = bool)
    mask = binary_dilation (mask, structure = struct).astype (np.uint8)
    return mask

_frac_key = "fractures_Xu_et_al.kmz"
if _frac_key not in all_lines:
    _frac_key = next ((k for k in all_lines), None)
if _frac_key is None:
    raise SystemExit ("ERROR: No fracture LineStrings found in any KMZ file.")
fracture_lines = all_lines [_frac_key]
print (f"Using fracture lines from: {_frac_key} ({len (fracture_lines)} lines)")

fracture_mask = rasterize_fractures (fracture_lines, grid ["lon"], grid ["lat"])
y_flat = fracture_mask.reshape (-1)
valid_mask = ~nodata_flat

# Exclude artifact pixels
RESULTS_DIR = os.path.join (DATA_DIR, "results")
_artifact_file = os.path.join (RESULTS_DIR, "artifact_mask.npz")
if os.path.exists (_artifact_file):
    artifact_mask = np.load (_artifact_file) ["artifact_mask"].ravel ()
    valid_mask = valid_mask & (artifact_mask == 0)
    print (f"Artifact mask applied: {artifact_mask.sum ():,} pixels excluded")
else:
    artifact_mask = np.zeros_like (nodata_flat)
    print ("No artifact mask found, proceeding without")

n_frac = fracture_mask.sum ()
n_total = fracture_mask.size
print (f"Fracture pixels: {n_frac:,} / {n_total:,} ({100*n_frac/n_total:.2f}%)")

# Augment training data
# Apply geometric transforms to the displacement grid, recompute features,
# and collect non-fracture pixels from each augmented version.

rng = np.random.RandomState (42)

SAMPLES_PER_AUG = 300_000 # non-fracture pixels sampled from each augmented grid

def _apply_transform (fn, z_ew, z_up, z_s1, mask, nd):
    """Apply a geometric transform to all 3 grids, mask, and nodata."""
    return fn (z_ew), fn (z_up), fn (z_s1), fn (mask), fn (nd)

augmentations = {
    "original":  lambda x: x,
    "rot90":     lambda x: np.rot90 (x, 1),
    "rot180":    lambda x: np.rot90 (x, 2),
    "rot270":    lambda x: np.rot90 (x, 3),
    "flip_h":    lambda x: np.fliplr (x),
    "flip_v":    lambda x: np.flipud (x),
    "noise_low": None, # handled separately (Gaussian noise sigma = 0.02)
    "noise_high": None, # handled separately (Gaussian noise sigma = 0.05)
}

aug_samples = []

for aug_name, aug_fn in augmentations.items ():
    print (f"  Augmentation: {aug_name}", end = "")

    if aug_name.startswith ("noise"):
        sigma = 0.02 if aug_name == "noise_low" else 0.05
        z_ew_aug = z_filled_ew + rng.normal (0, sigma, z_filled_ew.shape).astype (np.float32)
        z_up_aug = z_filled_up + rng.normal (0, sigma, z_filled_up.shape).astype (np.float32)
        z_s1_aug = z_filled_s1 + rng.normal (0, sigma, z_filled_s1.shape).astype (np.float32)
        mask_aug = fracture_mask
        nodata_aug = nodata
    else:
        z_ew_aug, z_up_aug, z_s1_aug, mask_aug, nodata_aug = _apply_transform (
            aug_fn, z_filled_ew.copy (), z_filled_up.copy (), z_filled_s1.copy (),
            fracture_mask.copy (), nodata.copy ()
        )

    # Find sample indices FIRST (before computing features)
    y_aug = mask_aug.reshape (-1)
    valid_aug = ~nodata_aug.reshape (-1)
    nonfrac_valid = np.where ((y_aug == 0) & valid_aug) [0]
    n_sample = min (SAMPLES_PER_AUG, len (nonfrac_valid))
    idx = rng.choice (nonfrac_valid, size = n_sample, replace = False)

    # Compute features per grid and extract sampled rows
    X_ew_aug = grid_to_feature_matrix_single (z_ew_aug)
    sampled = X_ew_aug [idx].copy ()
    del X_ew_aug
    X_up_aug = grid_to_feature_matrix_single (z_up_aug)
    sampled = np.hstack ([sampled, X_up_aug [idx]])
    del X_up_aug
    X_s1_aug = grid_to_feature_matrix_single (z_s1_aug)
    sampled = np.hstack ([sampled, X_s1_aug [idx]])
    del X_s1_aug

    np.nan_to_num (sampled, copy = False, nan = 0.0, posinf = 0.0, neginf = 0.0)
    aug_samples.append (sampled)
    print (f" — sampled {n_sample:,} non-fracture pixels")

X_ae_train_all = np.concatenate (aug_samples, axis = 0)
rng.shuffle (X_ae_train_all)
print (f"\nTotal augmented training set: {len (X_ae_train_all):,} pixels "
      f"({len (augmentations)} augmentations x ~{SAMPLES_PER_AUG:,})")

# Build test set (from original grid only, no augmentation)

frac_idx = np.where ((y_flat == 1) & valid_mask) [0]
nonfrac_idx = np.where ((y_flat == 0) & valid_mask) [0]

max_test_pos = min (len (frac_idx), 50000)
frac_test = rng.choice (frac_idx, size = max_test_pos, replace = False)
n_neg_test = min (len (nonfrac_idx), 3 * len (frac_test))
nonfrac_test = rng.choice (nonfrac_idx, size = n_neg_test, replace = False)

test_idx = np.concatenate ([frac_test, nonfrac_test])
rng.shuffle (test_idx)

X_test = X_flat_orig [test_idx]
y_test = y_flat [test_idx]

print (f"Test set: {len (y_test):,} pixels "
      f"({ (y_test == 1).sum ():,} fracture, { (y_test == 0).sum ():,} non-fracture)")

# Scale — fit on augmented training data, apply to test & full grid
scaler = StandardScaler ()
X_ae_train_sc = scaler.fit_transform (X_ae_train_all)
X_test_sc = scaler.transform (X_test)

# Define and train autoencoder

class Autoencoder (nn.Module):
    def __init__ (self, n_features, bottleneck = 4):
        super ().__init__ ()
        self.encoder = nn.Sequential (
            nn.Linear (n_features, 32),
            nn.ReLU (),
            nn.Linear (32, 16),
            nn.ReLU (),
            nn.Linear (16, bottleneck),
        )
        self.decoder = nn.Sequential (
            nn.Linear (bottleneck, 16),
            nn.ReLU (),
            nn.Linear (16, 32),
            nn.ReLU (),
            nn.Linear (32, n_features),
        )

    def forward (self, x):
        return self.decoder (self.encoder (x))

ae_dataset = TensorDataset (torch.tensor (X_ae_train_sc, dtype = torch.float32))
ae_loader = DataLoader (ae_dataset, batch_size = 512, shuffle = True)

ae_model = Autoencoder (n_feat_total, bottleneck = 4)
optimizer = torch.optim.Adam (ae_model.parameters (), lr = 1e-3)
criterion = nn.MSELoss ()

N_EPOCHS = 50
print (f"\nTraining augmented autoencoder ({N_EPOCHS} epochs, "
      f"{len (X_ae_train_sc):,} samples)...")
ae_model.train ()
for epoch in range (N_EPOCHS):
    epoch_loss = 0.0
    for (batch,) in ae_loader:
        recon = ae_model (batch)
        loss = criterion (recon, batch)
        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()
        epoch_loss += loss.item () * len (batch)
    epoch_loss /= len (X_ae_train_sc)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print (f"Epoch {epoch+1:3d}/{N_EPOCHS}  loss = {epoch_loss:.6f}")
print ("Done.")

# Evaluate on test set

ae_model.eval ()
with torch.no_grad ():
    X_test_t = torch.tensor (X_test_sc, dtype = torch.float32)
    recon_test = ae_model (X_test_t).numpy ()
    mse_test = np.mean ((X_test_sc - recon_test) ** 2, axis = 1)

mse_frac = mse_test [y_test == 1]
mse_nonfrac = mse_test [y_test == 0]
print (f"\nReconstruction MSE — non-fracture: {mse_nonfrac.mean ():.6f} ± {mse_nonfrac.std ():.6f}")
print (f"Reconstruction MSE — fracture: {mse_frac.mean ():.6f} ± {mse_frac.std ():.6f}")

ae_threshold = np.percentile (mse_nonfrac, 95)
print (f"Anomaly threshold (95th pctl of non-fracture): {ae_threshold:.6f}")

y_pred_ae = (mse_test > ae_threshold).astype (int)
f1_ae = f1_score (y_test, y_pred_ae)
auc_ae = roc_auc_score (y_test, mse_test)

print (f"\n{"—"*70}")
print (f"  Augmented Autoencoder  |  F1 = {f1_ae:.4f}  |  AUC = {auc_ae:.4f}")
print (f"{"—"*70}")
print (classification_report (y_test, y_pred_ae, target_names = ["Non-fracture", "Fracture"]))
cm_ae = confusion_matrix (y_test, y_pred_ae)
print (f"  Confusion Matrix:")
print (f"    TN = {cm_ae [0,0]:>7,}  FP = {cm_ae [0,1]:>7,}")
print (f"    FN = {cm_ae [1,0]:>7,}  TP = {cm_ae [1,1]:>7,}")

# Reconstruction error distribution
fig, ax = plt.subplots (figsize = (10, 5))
ax.hist (mse_nonfrac, bins = 100, alpha = 0.6, label = "Non-fracture", density = True)
ax.hist (mse_frac, bins = 100, alpha = 0.6, label = "Fracture", density = True)
ax.axvline (ae_threshold, color = "k", linestyle = "--", label = f"Threshold = {ae_threshold:.4f}")
ax.set_xlabel ("Reconstruction MSE")
ax.set_ylabel ("Density")
ax.set_title ("Augmented Autoencoder — Reconstruction Error Distribution")
ax.legend ()
plt.tight_layout ()
# plt.show ()

# Full-grid anomaly map

ae_model.eval ()
CHUNK = 500000
mse_full = []
with torch.no_grad ():
    for i in range (0, len (X_flat_orig), CHUNK):
        chunk_sc = scaler.transform (X_flat_orig [i:i+CHUNK])
        chunk_t = torch.tensor (chunk_sc, dtype = torch.float32)
        recon = ae_model (chunk_t).numpy ()
        mse_full.append (np.mean ((chunk_sc - recon) ** 2, axis = 1))
mse_full = np.concatenate (mse_full)
mse_map = mse_full.reshape (H, W)

extent = [grid ["lon"] [0], grid ["lon"] [-1], grid ["lat"] [0], grid ["lat"] [-1]]

fig, axes = plt.subplots (1, 3, figsize = (20, 6))

axes [0].imshow (fracture_mask, extent = extent, origin = "lower", cmap = "Greys", aspect = "auto")
axes [0].set_title ("Ground Truth")

im1 = axes [1].imshow (mse_map, extent = extent, origin = "lower", cmap = "hot", aspect = "auto",
                      vmin = 0, vmax = np.percentile (mse_full, 99))
axes [1].set_title ("Reconstruction Error (MSE)")
fig.colorbar (im1, ax = axes [1], shrink = 0.7, label = "MSE")

pred_ae_map = (mse_map > ae_threshold).astype (int)
pred_ae_map [artifact_mask.reshape (H, W) == 1] = 0  # exclude artifacts
mse_map [artifact_mask.reshape (H, W) == 1] = 0.0
n_pred_ae = pred_ae_map.sum ()
axes [2].imshow (pred_ae_map, extent = extent, origin = "lower", cmap = "Greys", aspect = "auto")
axes [2].set_title (f"Augmented AE Predicted Fractures ({n_pred_ae:,} px)")

for ax in axes:
    ax.set_xlabel ("Longitude")
    ax.set_ylabel ("Latitude")

plt.suptitle (f"Augmented Autoencoder Anomaly Detection — {grid_name}",
             fontsize = 14, fontweight = "bold")
plt.tight_layout ()
# plt.show ()

# Overlay on displacement
fig, ax = plt.subplots (figsize = (12, 8))
ax.imshow (grid ["z"], extent = extent, origin = "lower", cmap = "gray", aspect = "auto", alpha = 0.4)
im = ax.imshow (np.ma.masked_where (mse_map < ae_threshold, mse_map),
                extent = extent, origin = "lower", cmap = "hot", aspect = "auto",
                vmin = ae_threshold, vmax = np.percentile (mse_full, 99.5), alpha = 0.8)
ax.set_title (f"Augmented Autoencoder — Anomaly Map Overlaid on Displacement\n{grid_name}")
ax.set_xlabel ("Longitude")
ax.set_ylabel ("Latitude")
fig.colorbar (im, ax = ax, shrink = 0.7, label = "Reconstruction MSE")
plt.tight_layout ()
# plt.show ()

# Save predictions to disk
_results_dir = os.path.join (DATA_DIR, "results")
os.makedirs (_results_dir, exist_ok = True)

np.savez_compressed (os.path.join (_results_dir, "pred_autoencoder_augmented.npz"),
                    pred_map = pred_ae_map,
                    mse_map = mse_map,
                    threshold = ae_threshold,
                    model_name = "Augmented Autoencoder")
print (f"Saved Augmented Autoencoder -> {_results_dir}/pred_autoencoder_augmented.npz")

# Save trained model weights + threshold
_models_dir = os.path.join (DATA_DIR, "models")
os.makedirs (_models_dir, exist_ok = True)
torch.save ({
    "state_dict": ae_model.state_dict (),
    "n_feat_total": n_feat_total,
    "bottleneck": 4,
    "threshold": float (ae_threshold),
}, os.path.join (_models_dir, "autoencoder_augmented.pt"))
print (f"Saved augmented AE weights -> {_models_dir}/autoencoder_augmented.pt")
