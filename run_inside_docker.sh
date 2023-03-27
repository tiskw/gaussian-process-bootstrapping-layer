#!/bin/sh

# python3 train.py > output_1.txt
# python3 train.py > output_2.txt
# python3 train.py > output_3.txt
# python3 train.py > output_4.txt
# python3 train.py > output_5.txt

# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.1 > output_s0.1_1.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.1 > output_s0.1_2.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.1 > output_s0.1_3.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.1 > output_s0.1_4.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.1 > output_s0.1_5.txt
# 
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.2 > output_s0.2_1.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.2 > output_s0.2_2.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.2 > output_s0.2_3.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.2 > output_s0.2_4.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.2 > output_s0.2_5.txt
# 
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.3 > output_s0.3_1.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.3 > output_s0.3_2.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.3 > output_s0.3_3.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.3 > output_s0.3_4.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.3 > output_s0.3_5.txt
# 
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.4 > output_s0.4_1.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.4 > output_s0.4_2.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.4 > output_s0.4_3.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.4 > output_s0.4_4.txt
# python3 train.py --gpb_layer_pos 5 --gpb_std_error 0.4 > output_s0.4_5.txt
# 
# python3 train.py --gpb_layer_pos 012345 > output_all_1.txt
# python3 train.py --gpb_layer_pos 012345 > output_all_2.txt
# python3 train.py --gpb_layer_pos 012345 > output_all_3.txt
# python3 train.py --gpb_layer_pos 012345 > output_all_4.txt
# python3 train.py --gpb_layer_pos 012345 > output_all_5.txt

# python3 train.py --gpb_layer_pos 012345 --gpb_std_error 0.1 --gpb_dec_epochs 20 > output_all_1.txt
# python3 train.py --gpb_layer_pos 012345 --gpb_std_error 0.1 --gpb_dec_epochs 20 > output_all_2.txt
# python3 train.py --gpb_layer_pos 012345 --gpb_std_error 0.1 --gpb_dec_epochs 20 > output_all_3.txt
# python3 train.py --gpb_layer_pos 012345 --gpb_std_error 0.1 --gpb_dec_epochs 20 > output_all_4.txt
# python3 train.py --gpb_layer_pos 012345 --gpb_std_error 0.1 --gpb_dec_epochs 20 > output_all_5.txt



python3 train.py --dataset cifar100 > output_cifar100_1.txt
python3 train.py --dataset cifar100 > output_cifar100_2.txt
python3 train.py --dataset cifar100 > output_cifar100_3.txt
python3 train.py --dataset cifar100 > output_cifar100_4.txt
python3 train.py --dataset cifar100 > output_cifar100_5.txt

python3 train.py --dataset cifar100 --gpb_layer_pos 012345 --gpb_std_error 0.1 > output_pos012345_s0.1_1.txt
python3 train.py --dataset cifar100 --gpb_layer_pos 012345 --gpb_std_error 0.1 > output_pos012345_s0.1_2.txt
python3 train.py --dataset cifar100 --gpb_layer_pos 012345 --gpb_std_error 0.1 > output_pos012345_s0.1_3.txt
python3 train.py --dataset cifar100 --gpb_layer_pos 012345 --gpb_std_error 0.1 > output_pos012345_s0.1_4.txt
python3 train.py --dataset cifar100 --gpb_layer_pos 012345 --gpb_std_error 0.1 > output_pos012345_s0.1_5.txt

