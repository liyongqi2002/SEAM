import argparse
import json
import os
import re

from openai import OpenAI
from tqdm import tqdm



def parse_judge_response(response):
    # 使用正则表达式找出 "[[]]" 中的部分
    match = re.search(r'\[\[(.*?)\]\]', response)
    if match:
        match_str= match.group(1)
    else:
        match_str= None

    if match_str in ["Response 1 from AI","Response 2 from AI","Tie"]:
        return match_str
    else:
        return "Tie"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Name and Path
    parser.add_argument("--weak_model_name", default='Qwen2-1.5B-Instruct', type=str, help="Qwen2-1.5B-Instruct")
    parser.add_argument("--strong_model_name", default='Qwen2-7B', type=str, help="Qwen2-7B")
    script_args = parser.parse_args()
    dirs = ["batches_gpt4_judge_results", "batches_gpt4_judge_records"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


    # results_judge_requests_gpt4omini_file= f"batches_gpt4_judge_requests/batch_cjU0DRuyYgI9XFgwhqUAzlEh_output.jsonl"
    # results_judge_requests_gpt4o_file=f"batches_gpt4_judge_requests/batch_9UI3QcT9Y5XhmFZqTTIvkYHU_output.jsonl"
    results_judge_requests_gpt4o_file=f"batches_gpt4_judge_requests/batch_675e26ae51ec8190a948933147cb1ee2_output.jsonl"

    results_judge_requests_gpt4_file=results_judge_requests_gpt4o_file


    import jsonlines
    with open(results_judge_requests_gpt4_file, mode='r') as f:
        lines=f.readlines()
        results_judge_requests_gpt4=[]
        for line in lines:
            results_judge_requests_gpt4.append(json.loads(line))
    # print(results_judge_requests_gpt4)
    # print(len(results_judge_requests_gpt4))


    # RM_methods=["weak", "WeakSelf", "StrongSelf", "Ensemble", "Burns", "UFilter", "WSConsistency", "Ours", "OursFilter", "StrongCeiling"]
    RM_methods=["RebuttalCotWithDefinition", "RebuttalDebate"]

    RM_methods_dict={}
    for RM_method in RM_methods:
        RM_methods_dict[RM_method]=[{"order1":{},"order2":{}} for idx in range(int(len(results_judge_requests_gpt4)/(2*len(RM_methods))))]

    pattern = r"request-Method\[(?P<RM_method>\w+)\]-Idx\[(?P<response_idx>\d+)\]-Order\[(?P<order>\w+)\]"
    for result_object in results_judge_requests_gpt4:
        custom_id=result_object["custom_id"]
        # 使用正则表达式进行匹配
        match = re.match(pattern, custom_id)
        RM_method = match.group('RM_method')
        response_idx = int(match.group('response_idx'))
        order = match.group('order')

        response_text=result_object["response"]["body"]["choices"][0]["message"]["content"]
        judge_order=parse_judge_response(response_text)

        RM_methods_dict[RM_method][response_idx][order]={
            "response_text":response_text,
            "judge_order":judge_order,
        }
    # print(RM_methods_dict)

    for RM_method in RM_methods:
        judges_RM_method=RM_methods_dict[RM_method]

        record_file=f"batches_gpt4_judge_records/SFT-{RM_method}.txt"
        record_file= open(record_file, encoding='utf-8', mode='a')

        judge_results=[]
        judge_decisions=[]
        for response_idx in range(len(judges_RM_method)):
            judge=judges_RM_method[response_idx]
            judge_order1_dict=judge["order1"]
            judge_order2_dict=judge["order2"]

            judge_order1=judge_order1_dict["judge_order"]
            judge_order2=judge_order2_dict["judge_order"]

            response_order1=judge_order1_dict["response_text"]
            response_order2=judge_order2_dict["response_text"]

            print(f"===============================================",file=record_file)
            print(f"序号：{response_idx}",file=record_file)
            print(f"---response_order1---\n{response_order1}",file=record_file)
            print(f"---response_order2---\n{response_order2}",file=record_file)
            print(f"---------------------------------",file=record_file)
            print(f"---judge_order1---\n{judge_order1}",file=record_file)
            print(f"---judge_order2---\n{judge_order2}",file=record_file)

            judge_decision="Tie"
            if judge_order1==judge_order2:
                # 在两个顺序下，输出的都是一样的结果；（都是res1/res2，说明有位置偏差，需要视为Tie；都是Tie，直接是Tie）
                judge_decision = "Tie"
            else:
                # 两者不一样的情况下，先看是不是有一方为 Tie：
                if judge_order1=="Tie":
                    # 如果第1个顺序是Tie，则按照judge_order2判断最终结果
                    if judge_order2=="Response 1 from AI":
                        judge_decision = "DPO"
                    elif judge_order2=="Response 2 from AI":
                        judge_decision = "SFT"
                elif judge_order2=="Tie":
                    # 如果第2个顺序是Tie，则按照judge_order1判断最终结果
                    if judge_order1=="Response 1 from AI":
                        judge_decision = "SFT"
                    elif judge_order1=="Response 2 from AI":
                        judge_decision = "DPO"
                else:
                    # 其余情况，即两个judge不同，且没有任何一个是tie，则说明两者达成了一致意见；按照任意一个judge解析即可
                    if judge_order1=="Response 1 from AI":
                        judge_decision = "SFT"
                    elif judge_order1=="Response 2 from AI":
                        judge_decision = "DPO"
            print(f"---judge_decision---\n{judge_decision}",file=record_file)
            print(f"===============================================",file=record_file)

            judge_result={
                "response_idx":response_idx,
                "judge_decision": judge_decision,
                "judge_information":{
                    "response_order1":response_order1,
                    "judge_order1": judge_order1,
                    "response_order2": response_order2,
                    "judge_order2": judge_order2,
                }
            }
            judge_results.append(judge_result)

            judge_decisions.append(judge_decision)


        judge_results_file=f"batches_gpt4_judge_results/judge_results_SFT-{RM_method}.json"
        with open(judge_results_file, encoding='utf-8', mode= 'w') as f:
            json.dump(judge_results, f, indent=2)

        judge_metric_file=f"batches_gpt4_judge_results/judge_metric_SFT-{RM_method}.txt"
        judge_metric_file= open(judge_metric_file, encoding='utf-8', mode='a')

        decision_counts = {
            "SFT": 0,
            "Tie": 0,
            "DPO": 0,
        }
        # 遍历judge_decisions列表并统计每个决策的个数
        for decision in judge_decisions:
            if decision in decision_counts:
                decision_counts[decision] += 1
        print(f"==============================",file=judge_metric_file)
        print(f"eval num: {len(judges_RM_method)}",file=judge_metric_file)
        print(f"SFT 方: SFT-Alpaca-52k",file=judge_metric_file)
        print(f"DPO 方: {RM_method}",file=judge_metric_file)
        print(f"decision_counts: {decision_counts}",file=judge_metric_file)
        print(f"==============================",file=judge_metric_file)

        print(f"==============================")
        print(f"eval num: {len(judges_RM_method)}")
        print(f"SFT 方: SFT-Alpaca-52k")
        print(f"DPO 方: {RM_method}")
        print(f"decision_counts: {decision_counts}")
        print(f"==============================")

