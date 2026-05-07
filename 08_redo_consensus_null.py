
# %%
"""
Redo the full 12-category breakdown map and the log-scale null hypothesis
plot using 4 models: class-balanced RF, XGBoost, SVM + augmented Autoencoder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import binom

DATA_DIR = os.path.dirname (os.path.abspath (__file__))
RESULTS_DIR = os.path.join (DATA_DIR, "results")

# Load ground truth
gt = np.load (os.path.join (RESULTS_DIR, "ground_truth.npz"))
fracture_mask = gt ["fracture_mask"]
extent = gt ["extent"]
nan_mask = np.isnan (gt ["displacement"])
H, W = fracture_mask.shape

# Load 4 model predictions
model_files = [
    ("Random Forest",        "pred_random_forest.npz"),
    ("XGBoost",              "pred_xgboost.npz"),
    ("SVM (Linear)",         "pred_svm_linear.npz"),
    ("Autoencoder",          "pred_autoencoder_augmented.npz"),
]

n_models = len (model_files)
predictions = {}
positive_rates = []

for name, fname in model_files:
    data = np.load (os.path.join (RESULTS_DIR, fname))
    predictions [name] = data ["pred_map"]
    p = data ["pred_map"].mean ()
    positive_rates.append (p)
    print (f"{name:25s}  positive rate = {p:.4f}")

# Compute agreement (0-4)
agreement = np.zeros ((H, W), dtype = np.uint8)
for pred_map in predictions.values ():
    agreement += (pred_map > 0).astype (np.uint8)

# FIGURE 1: Full breakdown — GT (yes/no) x number of models (0-4)
# Encoding: 0-4 = not GT, detected by 0-4 models
#           5-9 = GT fracture, detected by 0-4 models

pair_colors = [
    ("#ffffff", "#888888"),
    ("#a6cee3", "#1f78b4"),
    ("#b2df8a", "#33a02c"),
    ("#fdae6b", "#e6550d"),
    ("#fc9272", "#a50f15"),
]

full_colors = [light for light, _dark in pair_colors] + \
              [dark for _light, dark in pair_colors] + ["#000000"]
full_cmap = ListedColormap (full_colors)

NAN_CODE = 2 * (n_models + 1)  # = 10
full_map = np.zeros ((H, W), dtype = np.uint8)
for n in range (n_models + 1):
    full_map [ (fracture_mask == 0) & (agreement == n)] = n
    full_map [ (fracture_mask == 1) & (agreement == n)] = n + (n_models + 1)
full_map [nan_mask] = NAN_CODE

# Build legend: "No GT" in 2 left columns, "GT frac" in 2 right columns
legend_nogt = []
legend_gt = []
for n in range (n_models + 1):
    cnt_no  = int (((fracture_mask == 0) & (agreement == n)).sum ())
    cnt_yes = int (((fracture_mask == 1) & (agreement == n)).sum ())
    if n == 0:
        legend_nogt.append (Patch (facecolor = pair_colors [n] [0], edgecolor = "#cccccc",
                                 label = f"No GT, 0 models ({cnt_no:,} px)"))
    else:
        legend_nogt.append (Patch (facecolor = pair_colors [n] [0],
                                 label = f"No GT, {n} model{"s" if n>1 else ""} ({cnt_no:,} px)"))
    legend_gt.append (Patch (facecolor = pair_colors [n] [1],
                           label = f"GT frac, {n} model{"s" if n!= 1 else ""} ({cnt_yes:,} px)"))

# ncol = 4 fills column-first: items 0..2 → col 0, items 3..5 → col 1, etc.
# Layout: col0 = NoGT 0-2, col1 = NoGT 3-4+blank, col2 = GT 0-2, col3 = GT 3-4+blank
_blank = Patch (facecolor = "none", edgecolor = "none", label = "")
nan_count = int (nan_mask.sum ())
nan_patch = Patch (facecolor = "#000000", label = f"No data ({nan_count:,} px)")
n_half = (n_models + 1 + 1) // 2  # 3 rows
full_legend = (legend_nogt [:n_half]
               + legend_nogt [n_half:] + [nan_patch] * (n_half - len (legend_nogt [n_half:]))
               + legend_gt [:n_half]
               + legend_gt [n_half:] + [_blank] * (n_half - len (legend_gt [n_half:])))

fig, ax = plt.subplots (figsize = (14, 10))
ax.imshow (full_map, extent = extent, origin = "lower", cmap = full_cmap,
          vmin = 0, vmax = 2 * (n_models + 1), aspect = "auto")
ax.set_title ("Ground Truth vs Model Detection — Full Breakdown (4 models)",
             fontsize = 21, fontweight = "bold")
ax.set_xlabel ("Longitude", fontsize = 15)
ax.set_ylabel ("Latitude", fontsize = 15)
ax.tick_params (labelsize = 15)
fig.legend (handles = full_legend, loc = "lower center", ncol = 4, fontsize = 13.5,
           frameon = True, bbox_to_anchor = (0.5, -0.06))
plt.tight_layout ()
plt.subplots_adjust (bottom = 0.12)
plt.savefig (os.path.join (RESULTS_DIR, "full_breakdown_gt_vs_models.png"),
            dpi = 200, bbox_inches = "tight")
print (f"Saved -> {RESULTS_DIR}/full_breakdown_gt_vs_models.png")
plt.show ()

# FIGURE 2: Agreement vs null hypothesis (log scale)
gt_pos = fracture_mask.ravel () == 1
gt_neg = fracture_mask.ravel () == 0
agreement_flat = agreement.ravel ()
n_pos = gt_pos.sum ()
n_neg = gt_neg.sum ()
n_total = len (agreement_flat)

observed_all = np.array ([np.sum (agreement_flat == k) for k in range (n_models + 1)])
observed_pos = np.array ([np.sum (agreement_flat [gt_pos] == k) for k in range (n_models + 1)])
observed_neg = np.array ([np.sum (agreement_flat [gt_neg] == k) for k in range (n_models + 1)])

# Poisson-binomial PMF (exact)
def poisson_binomial_pmf (probs):
    pmf = np.array ([1.0 - probs [0], probs [0]])
    for i in range (1, len (probs)):
        new_pmf = np.zeros (len (pmf) + 1)
        new_pmf [:len (pmf)] += pmf * (1.0 - probs [i])
        new_pmf [1:len (pmf)+1] += pmf * probs [i]
        pmf = new_pmf
    return pmf

pb_pmf = poisson_binomial_pmf (np.array (positive_rates))

expected_pb_all = pb_pmf * n_total
expected_pb_neg = pb_pmf * n_neg
expected_pb_pos = pb_pmf * n_pos

# Print table
for label, obs, exp in [ ("All pixels", observed_all, expected_pb_all),
                         ("GT-negative", observed_neg, expected_pb_neg),
                         ("GT-positive", observed_pos, expected_pb_pos)]:
    print (f"\n{label}:")
    for k in range (n_models + 1):
        ratio = obs [k] / exp [k] if exp [k] > 0 else float ("inf")
        print (f"  k = {k}: observed = {obs [k]:>12,}  expected = {exp [k]:>12,.0f}  ratio = {ratio:.1f}x")

# Plot
titles = [
    "All Pixels",
    "GT-Negative Pixels\n (potential unmapped fractures)",
    "GT-Positive Pixels\n (known fractures)",
]
obs_arrays = [observed_all, observed_neg, observed_pos]
exp_arrays = [expected_pb_all, expected_pb_neg, expected_pb_pos]
totals = [n_total, n_neg, n_pos]

k_vals = np.arange (n_models + 1)
bar_width = 0.35

fig, axes = plt.subplots (1, 3, figsize = (21, 7))
_null_labels = ["(a)", "(b)", "(c)"]

for i, (ax, title, obs, exp, nt) in enumerate (
        zip (axes, titles, obs_arrays, exp_arrays, totals)):

    ax.text (0.98, 0.98, _null_labels [i], transform = ax.transAxes, fontsize = 18,
            fontweight = "bold", va = "top", ha = "right")

    obs_pct = 100 * obs / nt
    exp_pct = 100 * exp / nt

    ax.bar (k_vals - bar_width/2, obs_pct, width = bar_width,
           color = "#1f77b4", label = "Observed", edgecolor = "white")
    ax.bar (k_vals + bar_width/2, exp_pct, width = bar_width,
           color = "#ff7f0e", label = "Random",
           edgecolor = "white")

    ax.set_yscale ("log")
    ax.set_xlabel ("Number of models predicting fracture", fontsize = 15)
    ax.set_ylabel ("% of pixels (log scale)", fontsize = 15)
    ax.set_title (title, fontsize = 18, fontweight = "bold")
    ax.set_xticks (k_vals)
    ax.legend (fontsize = 13.5, loc = "lower left")
    ax.grid (axis = "y", alpha = 0.3)
    ax.tick_params (labelsize = 15)

    # Ratio annotations for k >= 2
    for k in range (2, n_models + 1):
        if exp [k] > 0:
            ratio = obs [k] / exp [k]
            ax.text (k, max (obs_pct [k], exp_pct [k]) * 1.5,
                    f"{ratio:.1f}x", ha = "center", fontsize = 13.5,
                    fontweight = "bold", color = "#d62728")

plt.suptitle ("Model Agreement vs Random Null (log scale) — 4 Models",
             fontsize = 21, fontweight = "bold")
plt.tight_layout ()
plt.savefig (os.path.join (RESULTS_DIR, "agreement_vs_null_log.png"),
            dpi = 200, bbox_inches = "tight")
print (f"Saved -> {RESULTS_DIR}/agreement_vs_null_log.png")
plt.show ()
