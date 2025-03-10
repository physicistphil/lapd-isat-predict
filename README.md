# Predicting Isat in the LAPD

## Repository details

This repository contains code and data to replicate most of the results in the paper "Machine-learned trends in mirror configurations in the Large Plasma Device" by Phil Travis, Jacob Bortnik, and Troy Carter. 

All data can be accessed in the `datasets` folder. The `dr-idx` files provide the index of the data broken down by datarun. The numpy `.npz` files are dictionaries containing `x`, `y`, `x_ptp`, `x_mean`, `y_ptp`, and `y_mean`, which are the normalized inputs and outputs to the model along with the scaling factor and offset. Recovering the orignial input or Isat values would require multiping by `ptp` followed by adding the `mean`. 

A model can be trained by running `train_dense_beta_NLL.py`. There's also a shell script provided (`train_NLL_wd-scan.sh`) to train many different models in a sequence (in this case, to scan over seed and weight decay coefficient). Checkpoints for trained models are in `code/training_runs` and its subdirectories. The last checkpoints saved for each model are the ones used in the paper.

The poster presented at DPP is also in the repository. A comprehensive (if disorganized) writeup of most of the work performed in this study is in `Writeup_PP1.pdf`.

## ML details

The journal where the trained ML models were tracked is in `ML_journal.pdf`. In addition, most (if not all) runs were tracked on the Weights and Biases project page: [https://wandb.ai/phil/profile-predict](https://wandb.ai/phil/profile-predict).

