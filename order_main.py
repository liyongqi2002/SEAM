import argparse
import json
import os
import random
import time

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import set_seed, compute_AI_feedback


def get_prompt_preference(user_query, given_response_1, given_response_2):
    with open(f'prompts/basic_prompt_preference.txt', mode='r') as f:
        basic_prompt_preference = f.read()
    prompt_preference = basic_prompt_preference.format(
        user_query=user_query,
        given_response_1=given_response_1,
        given_response_2=given_response_2,
    )

    return prompt_preference




def get_args():
    # 参数设置
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='Qwen2-7B', type=str,
                        help="Qwen2-7B Qwen2-1.5B-Instruct")

    parser.add_argument("--dataset_name", default='AnthropicHH', type=str, help="UltraFeedback AnthropicHH")

    parser.add_argument("--split_mode", default='HeldOut', type=str, help="HeldOut")

    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--held_out_sample_num", default=5000, type=int, help="HeldOut 样本数目，这在别的文件可能被定义为了 test_num")

    script_args = parser.parse_args()

    if script_args.split_mode == "HeldOut":
        script_args.dataset_name_or_path = f'./data/{script_args.dataset_name}_train.json'
    else:
        raise Exception("Split mode not recognized. Options: HeldOut.")

    # # 修改相关路径
    with open('./config/path_config.json', encoding='utf-8', mode='r') as f:
        path_config = json.loads(f.read())
        path_config_reversed = {}
        for key in path_config.keys():
            for name, path in zip(path_config[key].keys(), path_config[key].values()):
                path_config_reversed[path] = name

    if not os.path.exists(script_args.model_name):
        script_args.model_name_or_path = path_config["llm"][script_args.model_name]

    return script_args


if __name__ == '__main__':
    script_args = get_args()
    set_seed(script_args.seed)

    # 需要记录到三个文件夹，一个是记录过程；还有一个记录预测metric情况；一个是记录预测结果（一个模型有独特的order，所以可以直接保存暂用，一个文件即可）；
    dirs = ["response_record", "metric_results", "prediction_results"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
    time_str = time.strftime("%m%d%H%M")
    print(time_str)
    record_file_path = f"response_record/observe_order_SPLIT[{script_args.split_mode}]-M[{script_args.model_name}]-D[{script_args.dataset_name}]-{time_str}.log"
    record_file_path = open(record_file_path, encoding='utf-8', mode='a')

    with open(script_args.dataset_name_or_path, mode='r') as f:
        instances = json.load(f)

    instances=random.sample(instances,k=min(script_args.held_out_sample_num,len(instances)))

    # 1、先按照正反顺序分别构造prompt并计算概率；然后选择概率更大的作为输入order，并对应地调整golden preference的偏好
    ##################################################################
    script_args.tokenizer_builder = AutoTokenizer.from_pretrained
    script_args.model_builder = AutoModelForCausalLM.from_pretrained


    if "70B" in script_args.model_name or "72B" in script_args.model_name:
        script_args.model_kwargs = {
            "device_map": "auto",
        }
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16,
        )

        script_args.model_kwargs = {
            "quantization_config":bnb_config,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }


    ##################################################################
    reward_formatted_instances = {
        "order1": [],
        "order2": []
    }
    for instance in tqdm(instances, total=len(instances)):
        for order in ["order1", "order2"]:
            inner_instance = instance[order]
            user_query = inner_instance["user_query"]
            given_response_1 = inner_instance["given_response_1"]
            given_response_2 = inner_instance["given_response_2"]
            prompt_preference = get_prompt_preference(user_query=user_query,
                                                      given_response_1=given_response_1,
                                                      given_response_2=given_response_2)
            if order == "order1":
                reward_formatted_instance = {
                    "prompt": prompt_preference,
                    "text_chosen": "<Chosen>Response 1 from AI</Chosen>\n<Rejected>Response 2 from AI</Rejected>",
                    "text_rejected": "<Chosen>Response 2 from AI</Chosen>\n<Rejected>Response 1 from AI</Rejected>"
                }
                # reward_formatted_instance = {
                #     "prompt": prompt_preference,
                #     "text_chosen": "<Chosen>Response 1 from AI</Chosen>",
                #     "text_rejected": "<Chosen>Response 2 from AI</Chosen>"
                # }
            else:
                reward_formatted_instance = {
                    "prompt": prompt_preference,
                    "text_chosen": "<Chosen>Response 2 from AI</Chosen>\n<Rejected>Response 1 from AI</Rejected>",
                    "text_rejected": "<Chosen>Response 1 from AI</Chosen>\n<Rejected>Response 2 from AI</Rejected>"
                }
                # reward_formatted_instance = {
                #     "prompt": prompt_preference,
                #     "text_chosen": "<Chosen>Response 2 from AI</Chosen>",
                #     "text_rejected": "<Chosen>Response 1 from AI</Chosen>"
                # }
            reward_formatted_instances[order].append(reward_formatted_instance)

    order1_dataset = Dataset.from_list(reward_formatted_instances["order1"])
    order2_dataset = Dataset.from_list(reward_formatted_instances["order2"])

    (order1_chosen_scores,
     order1_rejected_scores,
     order2_chosen_scores,
     order2_rejected_scores) = compute_AI_feedback(script_args=script_args,
                                                   model_name_or_path=script_args.model_name_or_path,
                                                   datasets=(order1_dataset, order2_dataset))

    ordered_instances = []

    num_true_reward_score = 0

    for ins_idx, (score_1, score_2, score_3, score_4) in enumerate(
            zip(order1_chosen_scores, order1_rejected_scores,
                order2_chosen_scores, order2_rejected_scores)):
        # 根据置信度选择order
        if max([score_1, score_2]) > max([score_3, score_4]):
            order = "order1"
        else:
            order = "order2"


        if order == "order1":
            if score_1 > score_2:
                pred = "given_response_1"  # 与golden相同
            else:
                pred = "given_response_2"
        else:
            if score_3 > score_4:
                pred = "given_response_2"  # 与golden相同
            else:
                pred = "given_response_1"

        ordered_instance = instances[ins_idx][order]
        ordered_instance["reward_score_pred_preference"]=pred

        ordered_instance["order"]=order
        ordered_instance["order_scores"]={
            "order1":{
                "response_1_score":score_1,
                "response_2_score": score_2
            },
            "order2": {
                "response_1_score": score_4,
                "response_2_score": score_3
            },
        }

        ordered_instances.append(ordered_instance)

        if pred == ordered_instance["golden_preference"]:
            # 与对应order下的 golden_preference 做比较
            num_true_reward_score += 1

    print(f"num_true_reward_score / len(instances)",num_true_reward_score / len(instances), file=record_file_path)
    print(f"len(instances): {len(instances)}", file=record_file_path)


    prediction_results_file = f'prediction_results/ordered_SPLIT[{script_args.split_mode}]-M[{script_args.model_name}]-D[{script_args.dataset_name}].json'
    with open(prediction_results_file, encoding='utf-8', mode= 'w') as f:
        json.dump(ordered_instances, f, indent=2)
