#!/bin/bash

model_list=("meta-llama/Llama-2-7b-chat-hf")
datasets="datasets/winogrande4MC,train,test;datasets/ai2_arc4MC,train,test;datasets/mmlu,train,test"
dataset_list=("datasets/winogrande4MC" "datasets/ai2_arc4MC" "datasets/mmlu")
trainsplit="train"
testsplit="test"

threshold_array=(10 20 30 40 50 60 70 80 90)
attn_threshold_array=(10 20 30 40 50 60 70 80 90)
for model in "${model_list[@]}"
do
    for threshold in "${threshold_array[@]}"
    do
        for attn_threshold in "${attn_threshold_array[@]}"
        do
            python attack.py --out-path results_search_grid --model $model --datasets $datasets --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovMlp,NCovPosAtten,NCovPairAtten
        done
    done
done
