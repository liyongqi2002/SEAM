time_now=$(date  "+%m%d%H%M")
CUDA_VISIBLE_DEVICES=3 python alpaca_sft.py --time $time_now


methods=("Ours" "OursFilter" "weak" "WeakSelf" "StrongSelf" "Ensemble" "Burns" "UFilter" "WSConsistency" "StrongCeiling")
for method in "${methods[@]}"
do
      CUDA_VISIBLE_DEVICES=3 python dpo_train.py  --method $method
done

methods=("WeakSelf")
for method in "${methods[@]}"
do
      CUDA_VISIBLE_DEVICES=3 python dpo_train.py  --method $method
done



methods=("weak" "WeakSelf" "StrongSelf" "Ensemble" "Burns" "UFilter" "WSConsistency" "Ours" "OursFilter" "StrongCeiling")
for method in "${methods[@]}"
do
      CUDA_VISIBLE_DEVICES=2 python eval.py  --method $method
done

methods=("SFT")

for method in "${methods[@]}"
do
      CUDA_VISIBLE_DEVICES=2 python eval.py  --method $method
done