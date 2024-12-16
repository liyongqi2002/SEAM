import argparse
import json
import os
import re

from openai import OpenAI
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Name and Path
    parser.add_argument("--weak_model_name", default='Qwen2-1.5B-Instruct', type=str, help="Qwen2-1.5B-Instruct")
    parser.add_argument("--strong_model_name", default='Qwen2-7B', type=str, help="Qwen2-7B")



    script_args = parser.parse_args()
    dirs = ["batches_gpt4_judge_results", "batches_gpt4_judge_requests", "batches_gpt4_judge_records"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    with open(f'../../prompts/basic_prompt_gpt4_eval.txt', mode='r') as f:
        basic_prompt_gpt4_eval = f.read()

    request_objects=[]
    # RM_methods=["weak", "WeakSelf", "StrongSelf", "Ensemble", "Burns", "UFilter", "WSConsistency", "Ours", "OursFilter", "StrongCeiling"]
    RM_methods=["RebuttalCotWithDefinition", "RebuttalDebate"]

    for RM_method in RM_methods:


        responses_SFT_filepath = f"./eval_responses-Method[SFT]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}].json"
        responses_DPO_with_RM_method_filepath = f"./eval_responses-Method[{RM_method}]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}].json"
        with open(responses_SFT_filepath, encoding='utf-8', mode='r') as f:
            responses_SFT_dict = json.load(f)
        with open(responses_DPO_with_RM_method_filepath, encoding='utf-8', mode='r') as f:
            responses_DPO_with_RM_method_dict = json.load(f)

        responses_SFT_dict=responses_SFT_dict[:1000]
        responses_DPO_with_RM_method_dict=responses_DPO_with_RM_method_dict[:1000]

        for response_idx in tqdm(range(len(responses_SFT_dict)),total=len(responses_SFT_dict),desc="GPT4 EVAL"):
            response_SFT_dict = responses_SFT_dict[response_idx]
            response_DPO_with_RM_method_dict = responses_DPO_with_RM_method_dict[response_idx]

            user_query = response_SFT_dict['prompt']

            response_SFT = response_SFT_dict['response']
            response_DPO = response_DPO_with_RM_method_dict['response']

            judge_prompt_order1 = basic_prompt_gpt4_eval.format(
                user_query=user_query,
                given_response_1=response_SFT,
                given_response_2=response_DPO,
            )
            judge_prompt_order2 = basic_prompt_gpt4_eval.format(
                user_query=user_query,
                given_response_1=response_DPO,
                given_response_2=response_SFT,
            )

            object_order1={"custom_id": f"request-Method[{RM_method}]-Idx[{response_idx}]-Order[order1]", "method": "POST", "url": "/v1/chat/completions",
                           "body": {"model": "gpt-4o",
                                    "messages": [{"role": "system", "content": "You're a helpful assistant for doing fair evaluations"},
                                                 {"role": "user", "content": judge_prompt_order1}], "max_tokens": 512}}

            object_order2={"custom_id": f"request-Method[{RM_method}]-Idx[{response_idx}]-Order[order2]", "method": "POST", "url": "/v1/chat/completions",
                           "body": {"model": "gpt-4o",
                                    "messages": [{"role": "system", "content": "You're a helpful assistant for doing fair evaluations"},
                                                 {"role": "user", "content": judge_prompt_order2}], "max_tokens": 512}}

            request_objects.append(object_order1)
            request_objects.append(object_order2)

    print(len(request_objects))
    judge_requests_file=f"batches_gpt4_judge_requests/judge_requests_gpt-4o_Rebuttal.jsonl"


    # with open(judge_requests_file, encoding='utf-8', mode= 'w') as f:
    #     for object in request_objects:
    #         f.writelines(str(object)+"\n")


    import jsonlines
    with jsonlines.open(judge_requests_file, mode='w') as writer:

        writer.write_all(request_objects)
