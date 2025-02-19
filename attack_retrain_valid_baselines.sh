#!/bin/bash
# python scripts/split_datasets.py

trainsplit="train"
testsplit="validation"
model_list=("meta-llama/Llama-2-7b-chat-hf")
dataset_list=("datasets/mmlu")

threshold=80
attn_threshold=80

for model in "${model_list[@]}"
do
    for dataset in "${dataset_list[@]}"
    do
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe naive --trans WordSwapNeighboringCharacterSwap
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe naive --trans WordSwapRandomCharacterDeletion
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe naive --trans WordSwapRandomCharacterInsertion
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe naive --trans WordSwapQWERTY

        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe fgsm
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe hotflip
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe leap
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe a2t
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe checklist
        # python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe iga
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe pruthi

        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovMlp
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovPosAtten
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovPairAtten
        python attack.py --out-path results_valid_baselines --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovMlp,NCovPosAtten,NCovPairAtten
    done
done
