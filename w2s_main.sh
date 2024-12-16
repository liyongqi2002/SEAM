
##################
# Note: Due to the positional bias in the preference task,
# we first select one order from the two orders with higher confidence based on the confidence.

time_now=$(date  "+%m%d%H%M")
datasets=('AHelpful' 'HelpSteer' 'AHarmless' 'CaiHarmless' 'AnthropicHH' 'SafeRLHF')
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python order_main.py --model_name Qwen2-1.5B-Instruct \
                                                --dataset_name $dataset

    CUDA_VISIBLE_DEVICES=0 python order_main.py --model_name Qwen2-7B \
                                                --dataset_name $dataset

done



##################
# Note: Due to GPU limitations, we cannot implement search and generation in the same command,
# so we split the entire thinking-while-searching into several sub-steps.

time_now=$(date  "+%m%d%H%M")
datasets=('AHelpful' 'HelpSteer' 'AHarmless' 'CaiHarmless' 'AnthropicHH' 'SafeRLHF')
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_calculate_prob \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B


    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step1 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B

    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step2 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B

    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step3 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B

    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step4 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B

    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step5 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B

    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step6 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B

    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step7 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B

    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step8 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B

    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step9 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B

    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_step10 \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B


    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_searching_collection \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B



    CUDA_VISIBLE_DEVICES=0 python w2s_WeakAnnotation.py --stage stage_tree_annotation \
                                              --dataset_name $dataset \
                                              --held_out_sample_num 5000 --time $time_now --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B


done


##################
# Note: The weak-to-strong fine-tuning stage and evaluate the W2SG performance.

time_now=$(date  "+%m%d%H%M")
datasets=('AHelpful' 'HelpSteer' 'AHarmless' 'CaiHarmless' 'AnthropicHH' 'SafeRLHF')
methods=("OursFilter")

for dataset in "${datasets[@]}"
do
  CUDA_VISIBLE_DEVICES=0 python evaluate_w2s.py --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B \
                                                         --dataset_name $dataset \
                                                         --RM_data_mode weak \
                                                         --time $time_now

  for method in "${methods[@]}"
  do
      # 用 method 标注的HeldOut 训练 strong 并测试其在Eval集上的性能
      CUDA_VISIBLE_DEVICES=0 python RM_sft.py --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B \
                                                         --dataset_name $dataset \
                                                         --data_mode $method \
                                                         --time $time_now

      CUDA_VISIBLE_DEVICES=0 python evaluate_w2s.py --weak_model_name Qwen2-1.5B-Instruct --strong_model_name Qwen2-7B \
                                                         --dataset_name $dataset \
                                                         --RM_data_mode $method \
                                                         --time $time_now
  done
done