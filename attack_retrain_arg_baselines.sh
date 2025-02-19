#!/bin/bash
# python scripts/split_datasets.py
# 检查是否提供了recipe参数
if [ -z "$1" ]; then
  echo "Usage: $0 <recipe>"
  exit 1
fi

recipe="$1"

trainsplit="train"
testsplit="test"
model="./retrain_models/Llama-2-7b-chat-hf-${recipe}/lora/sft/checkpoint-100"
out_path="results_retrain/results_${recipe}"
dataset="datasets/mmlu"

threshold=80
attn_threshold=80



# 根据recipe选择对应的attack行
if [ "$recipe" == "naive_WordSwapNeighboringCharacterSwap" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe naive --trans WordSwapNeighboringCharacterSwap
elif [ "$recipe" == "naive_WordSwapRandomCharacterDeletion" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe naive --trans WordSwapRandomCharacterDeletion
elif [ "$recipe" == "naive_WordSwapRandomCharacterInsertion" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe naive --trans WordSwapRandomCharacterInsertion
elif [ "$recipe" == "naive_WordSwapQWERTY" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe naive --trans WordSwapQWERTY
elif [ "$recipe" == "fgsm" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe fgsm

elif [ "$recipe" == "hotflip" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe hotflip

elif [ "$recipe" == "leap" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe leap

elif [ "$recipe" == "a2t" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe a2t

elif [ "$recipe" == "checklist" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe checklist

elif [ "$recipe" == "iga" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe iga

elif [ "$recipe" == "pruthi" ]; then
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --recipe pruthi

elif [ "$recipe" == "dtcover" ]; then
  # python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovMlp
  # python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovPosAtten
  # python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovPairAtten
  python attack.py --out-path $out_path --model $model --datasets $dataset --dataset-sample 1000 --trainsplit $trainsplit --testsplit $testsplit --threshold $threshold --attn-threshold $attn_threshold --recipe dtcover --trans NCovMlp,NCovPosAtten,NCovPairAtten

else
  echo "Unknown recipe: $recipe"
  exit 1
fi
