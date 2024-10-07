# !/bin/bash

# eval "$(conda shell.bash hook)"
# conda activate gr-id-pinn-2

# wd=(0.01 0.1 1.0 10.0)
seeds=(0 1 2 3 4)

for j in ${!seeds[@]}; do
	python train_dense_beta_NLL.py --lr 3e-4 --num_epochs 500 --num_layers 4 --layer_width 256 \
	--ensemble="beta-NLL_wd-$2" \
	--port $1 \
	--seed ${seeds[$j]} \
	--dataset DR_combo_PP1_isat_04_train_cv-0.npz \
	--dataset_valid DR_combo_PP1_isat_04_valid_cv-0.npz \
	--weight_decay $2
done

# for j in ${!seeds[@]}; do
# 	python train_dense_beta_NLL.py --lr 3e-4 --num_epochs 500 --num_layers 4 --layer_width 256 \
# 	--ensemble="beta-NLL_wd-$3" \
# 	--port $1 \
# 	--seed ${seeds[$j]} \
# 	--dataset DR_combo_PP1_isat_04_train_cv-0.npz \
# 	--dataset_valid DR_combo_PP1_isat_04_valid_cv-0.npz \
# 	--weight_decay $3
# done
