#!/bin/bash

model_list=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-Instruct-v0.2" "internlm/internlm2-7b")
dataset_list=("XiaHan19/winogrande4MC" "XiaHan19/ai2_arc4MC" "cais/mmlu")
for model in "${model_list[@]}"
do
    for dataset in "${dataset_list[@]}"
    do
        if [ "$dataset" = "XiaHan19/winogrande4MC" ]; then
            devsplit="train"
            testsplit="validation"
        else
            devsplit="validation"
            testsplit="test"
        fi

        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --trans WordSwapNeighboringCharacterSwap
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --trans WordSwapRandomCharacterDeletion
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --trans WordSwapRandomCharacterInsertion
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --trans WordSwapQWERTY

        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --trans HotFlip
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --trans FGSM
        
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans HotFlipNCovMlp
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans HotFlipNCovPosAtten
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans HotFlipNCovPairAtten

        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans HotFlip,HotFlipNCovMlp
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans HotFlip,HotFlipNCovMlp,HotFlipNCovPosAtten,HotFlipNCovPairAtten

        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans FGSMNCovMlp
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans FGSMNCovPosAtten
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans FGSMNCovPairAtten

        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans FGSM,FGSMNCovMlp
        python attack.py --model $model --dataset $dataset --devsplit $devsplit --testsplit $testsplit --search-method greedy --threshold 30 --attn-threshold 30 --samplek 100  --alpha 0 --cover-strategy all --trans FGSM,FGSMNCovMlp,FGSMNCovPosAtten,FGSMNCovPairAtten

    done
done
