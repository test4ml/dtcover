#!/bin/bash

model_list=("meta-llama/Llama-2-7b-hf")
datasets="XiaHan19/winogrande4MC,train,validation;XiaHan19/ai2_arc4MC,validation,test;cais/mmlu,validation,test"
dataset_list=("XiaHan19/winogrande4MC" "XiaHan19/ai2_arc4MC" "cais/mmlu")
devsplit="validation"
testsplit="test"

samplek_array=(100)
threshold_array=(10 20 30 40 50 60 70 80 90)
attn_threshold_array=(10 20 30 40 50 60 70 80 90)
for model in "${model_list[@]}"
do
    for samplek in "${samplek_array[@]}"
    do
        for threshold in "${threshold_array[@]}"
        do
            for attn_threshold in "${attn_threshold_array[@]}"
            do
                python attack.py --out-path results_search_threshold --model $model --datasets $datasets --dataset-sample 1000 --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold $threshold --attn-threshold $attn_threshold --samplek $samplek --alpha 0 --cover-strategy all --trans HotFlip,HotFlipNCovMlp,HotFlipNCovPosAtten,HotFlipNCovPairAtten
            done
        done
    done
done
