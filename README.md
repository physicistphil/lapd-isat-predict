# Predicting Isat in the LAPD

2024-10-7

This repository contains code and data to replicate (a portion) of the poster at DPP 2024. All the data can be accessed in the `datasets` folder. The `dr-idx` files provide the index of the data broken down by datarun. The numpy `.npz` files are dictionaries containing `x`, `y`, `x_ptp`, `x_mean`, `y_ptp`, and `y_mean`, which are the normalized inputs and outputs to the model along with the scaling factor and offset. Recovering the orignial input or Isat values would require multiping by `ptp` followed by adding the `mean`. 

A model can be trained by running `train_dense_beta_NLL.py`. There's also a shell script provided (`train_NLL_wd-scan.sh`) to train many different models in a sequence (in this case, to scan over seed and weight decay coefficient). Trained models checkpoints are in `code/training_runs` and its subdirectories. 

The poster presented at DPP is also in the repository. A writeup containing work up to DPP 2024 is also contained.

Please note that this work is in progress.

This repository will not be updated.
