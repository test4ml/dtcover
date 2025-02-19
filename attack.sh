#!/bin/bash
# python scripts/split_datasets.py

trainsplit="train"
testsplit="test"
model_list=("meta-llama/Llama-2-7b-chat-hf" "mistralai/Mistral-7B-Instruct-v0.2" "internlm/internlm2-chat-7b")
dataset_list=("datasets/winogrande4MC" "datasets/ai2_arc4MC" "datasets/mmlu")

threshold=80
attn_threshold=80

for model in "${model_list[@]}"
do
    for dataset in "${dataset_list[@]}"
    do
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe naive --trans WordSwapNeighboringCharacterSwap
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe naive --trans WordSwapRandomCharacterDeletion
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe naive --trans WordSwapRandomCharacterInsertion
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe naive --trans WordSwapQWERTY

        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe fgsm
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe hotflip
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe leap
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe a2t
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe checklist
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe pruthi

        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovMlp
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovPosAtten
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovPairAtten
        python attack.py --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovMlp,NCovPosAtten,NCovPairAtten
    done
done
