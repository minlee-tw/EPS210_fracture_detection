# EPS210_fracture_detection
AI for Earth and Planetary Sciences Final Project 

Script overview:
01_build_artifact_mask.py: rasterizes the artifact LineStrings (from image_artifact.XXX files) into python array artifact_mask.npz
02_compare_sampling.py: trains RF, XGB, and SVM with both uniform and fracture length-weighted training data (EW_high_800.kmz, UP_high_800.kmz, S1_T064_asc_stack_phasegradient_y.kmz), with the predictions saved as .npz files. Also saves ground_truth.npz after rasterizing the fracture data (from fractures_Xu_et_al.kmz) and plots sampling_comparison.png.
03_autoencoder_fracture.py: trains the autoencoder, with the prediction saved as .npz file.
04_autoencoder_augmented.py: trains the augmented autoencoder, with the prediction saved as .npz file.
05_combined_length_figure.py: plots combined_length_figure.png
06_compare_autoencoders.py: autoencoder_comparison_metrics.png and autoencoder_comparison_mse.
07_compare_barplots.py: plots metric_barplots.png.
08_redo_consensus_null.py: plots agreement_vs_null_log.png and full_breakdown_gt_vs_models.png.
Claude assisted in generating the codes, especially related to the autoencoder.
