
# %%
"""
Compare non-augmented vs augmented autoencoder:
  1. Bar plot of evaluation metrics
  2. Reconstruction error (MSE) distributions for fracture vs non-fracture pixels
"""

import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname (os.path.abspath (__file__))
RESULTS_DIR = os.path.join (DATA_DIR, "results")

# Load data
gt = np.load (os.path.join (RESULTS_DIR, "ground_truth.npz"))
fracture_mask = gt ["fracture_mask"].ravel ()

ae_plain = np.load (os.path.join (RESULTS_DIR, "pred_autoencoder.npz"))
ae_aug = np.load (os.path.join (RESULTS_DIR, "pred_autoencoder_augmented.npz"))

models = {
    "Non-augmented": {
        "pred": ae_plain ["pred_map"].ravel (),
        "mse":  ae_plain ["mse_map"].ravel (),
        "threshold": float (ae_plain ["threshold"]),
    },
    "Augmented": {
        "pred": ae_aug ["pred_map"].ravel (),
        "mse":  ae_aug ["mse_map"].ravel (),
        "threshold": float (ae_aug ["threshold"]),
    },
}

# Compute metrics
metric_names = ["Precision", "Recall (TPR)", "F1 Score", "Specificity (TNR)",
                "FPR", "Balanced Accuracy", "Accuracy", "IoU (Jaccard)"]
metrics = {m: [] for m in metric_names}
model_labels = list (models.keys ())

for label, m in models.items ():
    pred = m ["pred"]
    TP = int (((fracture_mask == 1) & (pred == 1)).sum ())
    TN = int (((fracture_mask == 0) & (pred == 0)).sum ())
    FP = int (((fracture_mask == 0) & (pred == 1)).sum ())
    FN = int (((fracture_mask == 1) & (pred == 0)).sum ())

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    bal  = (rec + spec) / 2.0
    acc  = (TP + TN) / (TP + TN + FP + FN)
    iou  = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    metrics ["Precision"].append (prec)
    metrics ["Recall (TPR)"].append (rec)
    metrics ["F1 Score"].append (f1)
    metrics ["Specificity (TNR)"].append (spec)
    metrics ["FPR"].append (fpr)
    metrics ["Balanced Accuracy"].append (bal)
    metrics ["Accuracy"].append (acc)
    metrics ["IoU (Jaccard)"].append (iou)

    print (f"{label:20s}  TP = {TP:>6,} TN = {TN:>10,} FP = {FP:>8,} FN = {FN:>6,}")
    print (f"  Prec = {prec:.4f} Rec = {rec:.4f} F1 = {f1:.4f} Spec = {spec:.4f} "
          f"Threshold = {m ["threshold"]:.6f}")

# FIGURE: bar plots (top) + MSE distributions (bottom)
fig = plt.figure (figsize = (20, 12))

# Top row: bar plots (4 key metrics)
bar_metrics = [
    ("Precision",     r"Precision = $\frac{TP}{TP + FP}$"),
    ("Recall (TPR)",  r"Recall = $\frac{TP}{TP + FN}$"),
    ("F1 Score",      r"F1 = $\frac{2 \cdot Prec \cdot Rec}{Prec + Rec}$"),
    ("IoU (Jaccard)", r"IoU = $\frac{TP}{TP + FP + FN}$"),
]

colors = ["#1f77b4", "#ff7f0e"]
x = np.arange (len (model_labels))
bar_w = 0.5
_subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

for i, (key, title) in enumerate (bar_metrics):
    ax = fig.add_subplot (2, 4, i + 1)
    ax.text (0.02, 0.98, _subplot_labels [i], transform = ax.transAxes, fontsize = 18,
            fontweight = "bold", va = "top", ha = "left")
    vals = metrics [key]
    bars = ax.bar (x, vals, width = bar_w, color = colors, edgecolor = "white")
    for bar, v in zip (bars, vals):
        ax.text (bar.get_x () + bar.get_width () / 2, bar.get_height () + 0.0005,
                f"{v:.4f}", ha = "center", va = "bottom", fontsize = 15, fontweight = "bold")
    ax.set_title (title, fontsize = 16.5, fontweight = "bold")
    ax.set_xticks (x)
    ax.set_xticklabels (model_labels, fontsize = 15)
    ax.set_ylim (0, max (vals) * 1.3)
    ax.grid (axis = "y", alpha = 0.3)
    ax.tick_params (labelsize = 15)

# Additional bar plots
extra_metrics = [
    ("Specificity (TNR)", r"Specificity = $\frac{TN}{TN + FP}$"),
    ("FPR",               r"FPR = $\frac{FP}{FP + TN}$"),
    ("Balanced Accuracy", r"Balanced Acc = $\frac{TPR + TNR}{2}$"),
    ("Accuracy",          r"Accuracy = $\frac{TP+TN}{TP+TN+FP+FN}$"),
]

for i, (key, title) in enumerate (extra_metrics):
    ax = fig.add_subplot (2, 4, i + 5)
    ax.text (0.02, 0.98, _subplot_labels [i + 4], transform = ax.transAxes, fontsize = 18,
            fontweight = "bold", va = "top", ha = "left")
    vals = metrics [key]
    bars = ax.bar (x, vals, width = bar_w, color = colors, edgecolor = "white")
    for bar, v in zip (bars, vals):
        ax.text (bar.get_x () + bar.get_width () / 2, bar.get_height () + 0.0005,
                f"{v:.4f}", ha = "center", va = "bottom", fontsize = 15, fontweight = "bold")
    ax.set_title (title, fontsize = 16.5, fontweight = "bold")
    ax.set_xticks (x)
    ax.set_xticklabels (model_labels, fontsize = 15)
    ax.set_ylim (0, min (max (vals) * 1.15, 1.02))
    ax.grid (axis = "y", alpha = 0.3)
    ax.tick_params (labelsize = 15)

plt.suptitle ("Autoencoder: Non-Augmented vs Augmented — Evaluation Metrics",
             fontsize = 22, fontweight = "bold")
plt.tight_layout ()
plt.savefig (os.path.join (RESULTS_DIR, "autoencoder_comparison_metrics.png"),
            dpi = 200, bbox_inches = "tight")
print (f"\nSaved metrics -> {RESULTS_DIR}/autoencoder_comparison_metrics.png")
plt.show ()

# FIGURE 2: Reconstruction error distributions
fig, axes = plt.subplots (1, 2, figsize = (16, 6), sharey = True)

for si, (ax, (label, m)) in enumerate (zip (axes, models.items ())):
    ax.text (0.02, 0.98, f"({"ab" [si]})", transform = ax.transAxes, fontsize = 18,
            fontweight = "bold", va = "top", ha = "left")
    mse = m ["mse"]
    threshold = m ["threshold"]

    mse_frac = mse [fracture_mask == 1]
    mse_nonfrac = mse [fracture_mask == 0]

    # Use same bins for both
    clip_max = np.percentile (mse, 99.5)
    bins = np.linspace (0, clip_max, 100)

    ax.hist (mse_nonfrac, bins = bins, alpha = 0.7, color = "#1f77b4", density = True,
            label = f"Non-fracture (n = {len (mse_nonfrac):,})", edgecolor = "none")
    ax.hist (mse_frac, bins = bins, alpha = 0.7, color = "#d62728", density = True,
            label = f"Fracture (n = {len (mse_frac):,})", edgecolor = "none")

    ax.axvline (threshold, color = "black", ls = "--", lw = 2,
               label = f"Threshold = {threshold:.4f}")

    ax.set_xlabel ("Reconstruction MSE", fontsize = 16.5)
    ax.set_title (f"{label} Autoencoder", fontsize = 19.5, fontweight = "bold")
    ax.legend (fontsize = 13.5)
    ax.grid (axis = "y", alpha = 0.3)
    ax.tick_params (labelsize = 15)

    # Print stats
    print (f"\n{label}:")
    print (f"  Non-fracture MSE: mean = {mse_nonfrac.mean ():.4f} std = {mse_nonfrac.std ():.4f}")
    print (f"  Fracture MSE:     mean = {mse_frac.mean ():.4f} std = {mse_frac.std ():.4f}")
    print (f"  Threshold: {threshold:.4f}")

axes [0].set_ylabel ("Density", fontsize = 16.5)

plt.suptitle ("Reconstruction Error Distribution — Fracture vs Non-Fracture Pixels",
             fontsize = 21, fontweight = "bold")
plt.tight_layout ()
plt.savefig (os.path.join (RESULTS_DIR, "autoencoder_comparison_mse.png"),
            dpi = 200, bbox_inches = "tight")
print (f"\nSaved MSE distributions -> {RESULTS_DIR}/autoencoder_comparison_mse.png")
plt.show ()
