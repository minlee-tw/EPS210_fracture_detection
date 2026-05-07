
# %%
"""
Bar-plot comparison of all four models across multiple evaluation metrics.
Loads saved predictions from results/ and computes metrics on the full grid.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname (os.path.abspath (__file__))
RESULTS_DIR = os.path.join (DATA_DIR, "results")

# Load ground truth
gt = np.load (os.path.join (RESULTS_DIR, "ground_truth.npz"))
fracture_mask = gt ["fracture_mask"].ravel ()

# Load model predictions
model_files = [
    ("Random Forest",        "pred_random_forest.npz"),
    ("XGBoost",              "pred_xgboost.npz"),
    ("SVM (Linear)",         "pred_svm_linear.npz"),
    ("Autoencoder",          "pred_autoencoder_augmented.npz"),
]

names = []
metrics = {
    "Precision":        [],
    "Recall (TPR)":     [],
    "F1 Score":         [],
    "Positive Rate":    [],
    "FPR":              [],
    "Balanced Accuracy": [],
    "Accuracy":         [],
    "IoU (Jaccard)":    [],
}

for name, fname in model_files:
    pred = np.load (os.path.join (RESULTS_DIR, fname)) ["pred_map"].ravel ()
    names.append (name)

    TP = int (((fracture_mask == 1) & (pred == 1)).sum ())
    TN = int (((fracture_mask == 0) & (pred == 0)).sum ())
    FP = int (((fracture_mask == 0) & (pred == 1)).sum ())
    FN = int (((fracture_mask == 1) & (pred == 0)).sum ())

    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall      = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    pos_rate    = (TP + FP) / (TP + TN + FP + FN)
    fpr         = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    bal_acc     = (recall + specificity) / 2.0
    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    iou         = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    metrics ["Precision"].append (precision)
    metrics ["Recall (TPR)"].append (recall)
    metrics ["F1 Score"].append (f1)
    metrics ["Positive Rate"].append (pos_rate)
    metrics ["FPR"].append (fpr)
    metrics ["Balanced Accuracy"].append (bal_acc)
    metrics ["Accuracy"].append (accuracy)
    metrics ["IoU (Jaccard)"].append (iou)

    print (f"{name:25s}  TP = {TP:>7,}  TN = {TN:>10,}  FP = {FP:>10,}  FN = {FN:>6,}")

# Print table
print ()
header = f"{"Metric":<22s}" + "".join (f"{n:>14s}" for n in names)
print (header)
print ("-" * len (header))
for metric_name, vals in metrics.items ():
    row = f"{metric_name:<22s}" + "".join (f"{v:>14.4f}" for v in vals)
    print (row)

# Plot: one subplot per metric (except FPR which we combine with TPR)
bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

plot_metrics = [
    ("Precision",         r"Precision = $\frac{TP}{TP + FP}$"),
    ("Recall (TPR)",      r"Recall (TPR) = $\frac{TP}{TP + FN}$"),
    ("F1 Score",          r"F1 = $\frac{2 \cdot Prec \cdot Rec}{Prec + Rec}$"),
    ("Positive Rate",     r"Positive Rate = $\frac{TP + FP}{TP + TN + FP + FN}$"),
    ("FPR",               r"FPR = $\frac{FP}{FP + TN}$"),
    ("Balanced Accuracy", r"Balanced Accuracy = $\frac{TPR + TNR}{2}$"),
    ("Accuracy",          r"Accuracy = $\frac{TP + TN}{TP + TN + FP + FN}$"),
    ("IoU (Jaccard)",     r"IoU = $\frac{TP}{TP + FP + FN}$"),
]

fig, axes = plt.subplots (2, 4, figsize = (22, 10))
axes = axes.ravel ()

x = np.arange (len (names))
bar_width = 0.6
_subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

for i, (metric_key, metric_title) in enumerate (plot_metrics):
    ax = axes [i]
    ax.text (0.02, 0.98, _subplot_labels [i], transform = ax.transAxes, fontsize = 18,
            fontweight = "bold", va = "top", ha = "left")
    vals = metrics [metric_key]
    bars = ax.bar (x, vals, width = bar_width, color = bar_colors, edgecolor = "white")

    # Value labels on bars
    for bar, v in zip (bars, vals):
        ax.text (bar.get_x () + bar.get_width () / 2, bar.get_height () + 0.005,
                f"{v:.4f}", ha = "center", va = "bottom", fontsize = 12, fontweight = "bold")

    ax.set_title (metric_title, fontsize = 16.5, fontweight = "bold")
    ax.set_xticks (x)
    ax.set_xticklabels (names, fontsize = 12)
    ax.set_ylim (0, min (max (vals) * 1.25, 1.05))
    ax.grid (axis = "y", alpha = 0.3)
    ax.tick_params (labelsize = 12)

plt.suptitle ("Model Comparison — Evaluation Metrics (Full Grid)",
             fontsize = 22, fontweight = "bold")
plt.tight_layout ()
plt.savefig (os.path.join (RESULTS_DIR, "metric_barplots.png"),
            dpi = 200, bbox_inches = "tight")
print (f"\nSaved bar plots -> {RESULTS_DIR}/metric_barplots.png")
plt.show ()
