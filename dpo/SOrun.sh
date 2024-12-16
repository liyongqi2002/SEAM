time_now=$(date  "+%m%d%H%M")
CUDA_VISIBLE_DEVICES=3 python alpaca_sft.py --time $time_now


methods=("RebuttalCotWithDefinition" "RebuttalDebate")
for method in "${methods[@]}"
do
      CUDA_VISIBLE_DEVICES=3 python dpo_train.py  --method $method
done


methods=("RebuttalCotWithDefinition" "RebuttalDebate")
for method in "${methods[@]}"
do
      CUDA_VISIBLE_DEVICES=3 python eval.py  --method $method
done