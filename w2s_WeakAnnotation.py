import copy
import math

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from llm_prob import batch_compute_prob
from utils import prepare_llm, compute_AI_feedback_single


import argparse
import json
import os

import torch

from utils import set_seed


def get_principle_info():
    with open('principles/principles.json', mode='r') as f:
        dict_principles = json.load(f)

    with open('principles/principle_demos.json', mode='r') as f:
        principle_demos = json.load(f)

    dict_principle_to_demos = {}
    for principle_demo in principle_demos:
        principle = principle_demo["principle"]

        if principle not in dict_principle_to_demos.keys():
            dict_principle_to_demos[principle] = []

        dict_principle_to_demos[principle].append(principle_demo)
    return dict_principles, dict_principle_to_demos


def read_ordered_instances(script_args):
    ordered_results_file = f'prediction_results/ordered_SPLIT[{script_args.split_mode}]-M[{script_args.strong_model_name}]-D[{script_args.dataset_name}].json'
    with open(ordered_results_file, encoding='utf-8', mode='r') as f:
        ordered_instances = json.load(f)
    ordered_instances = ordered_instances[:script_args.held_out_sample_num]

    # 我们选取其中一个顺序进行后续的操作，该顺序只是一种预处理的方式
    reformatted_ordered_instances = []
    for instance in ordered_instances:
        golden_response = instance[instance["golden_preference"]]

        # 为strong的置信度的顺序（并非最终答案）
        # 如果不做这一步，后续的操作将无法统一
        # 相当于让strong的选择做默认的正方
        if instance["reward_score_pred_preference"] == "given_response_1":
            reformatted_given_response_1 = instance["given_response_1"]
            reformatted_given_response_2 = instance["given_response_2"]
        else:
            # 调换顺序
            reformatted_given_response_1 = instance["given_response_2"]
            reformatted_given_response_2 = instance["given_response_1"]

        # 转回标识
        if golden_response == reformatted_given_response_1:
            golden_preference = "given_response_1"
        else:
            golden_preference = "given_response_2"

        reformatted_ordered_instance = {
            "user_query": instance["user_query"],
            "given_response_1": reformatted_given_response_1,
            "given_response_2": reformatted_given_response_2,
            "golden_preference": golden_preference,
        }
        reformatted_ordered_instances.append(reformatted_ordered_instance)

    return reformatted_ordered_instances


def get_args():
    # 参数设置
    parser = argparse.ArgumentParser()

    parser.add_argument("--weak_model_name", default='Qwen2-1.5B-Instruct', type=str,
                        help="Qwen2-7B Qwen2-1.5B-Instruct")

    parser.add_argument("--strong_model_name", default='Qwen2-7B', type=str,
                        help="Qwen2-7B Qwen2-1.5B-Instruct")

    parser.add_argument("--dataset_name", default='AnthropicHH', type=str, help="UltraFeedback AnthropicHH")

    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    parser.add_argument("--stage", default='stage_observe', type=str,
                        help="stage_principle (STRONG 发挥原则树构建作用) stage_observe（STRONG获得正负解释） stage_check（WEAK 做解释检查）")

    parser.add_argument("--held_out_sample_num", default=1000, type=int, help="选取样本数目")

    parser.add_argument("--time", default='07181648', type=str, help="time in bash.sh")

    script_args = parser.parse_args()

    script_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 本文件只涉及对 HeldOut split的观察和推理等；所以默认都是在 train set上做观察
    script_args.split_mode = "HeldOut"
    script_args.dataset_name_or_path = f'./data/{script_args.dataset_name}_train.json'

    # # 修改相关路径
    with open('./config/path_config.json', encoding='utf-8', mode='r') as f:
        path_config = json.loads(f.read())
        path_config_reversed = {}
        for key in path_config.keys():
            for name, path in zip(path_config[key].keys(), path_config[key].values()):
                path_config_reversed[path] = name

    if not os.path.exists(script_args.strong_model_name):
        script_args.strong_model_name_or_path = path_config["llm"][script_args.strong_model_name]

    if not os.path.exists(script_args.weak_model_name):
        script_args.weak_model_name_or_path = path_config["llm"][script_args.weak_model_name]

    return script_args


def calculate_probs(script_args, model_path, ordered_instances, all_candidate_principles):
    def get_prompts_bridge(script_args, basic_prompt_principle, ordered_instances, all_candidate_principles):
        prompts_bridge = []
        for ins_idx, ordered_instance in enumerate(ordered_instances):

            candidate_principles = all_candidate_principles[ins_idx]
            candidate_principles_str = ""
            for principle in candidate_principles:
                principle_desc = script_args.dict_principles[principle]
                candidate_principles_str += f"{principle_desc}\n"
            demonstrations_str = ""
            for principle in candidate_principles:
                demo = script_args.dict_principle_to_demos[principle][0]  # 当前框架中，每个原则只有一个demo对应

                demo_user_query = demo["user_query"]
                demo_given_response_1 = demo["chosen_response"]
                demo_given_response_2 = demo["rejected_response"]
                demo_relevant_principle = demo["principle"]
                demo_relevant_principle_desc = demo["principle_desc"]

                demo_str = ""
                demo_str += f"<Instance>\n"
                demo_str += f"<User>{demo_user_query}</User>\n"
                demo_str += f"<Response 1 from AI>{demo_given_response_1}</Response 1 from AI>\n"
                demo_str += f"<Response 2 from AI>{demo_given_response_2}</Response 2 from AI>\n"
                demo_str += f"<Relevant Principle>{demo_relevant_principle}</Relevant Principle>\n"
                demo_str += f"<Relevant Principle Description>{demo_relevant_principle_desc}</Relevant Principle Description>\n"
                demo_str += f"</Instance>\n"
                demonstrations_str += demo_str

            prompt_bridge = basic_prompt_principle.format(
                candidate_principles_str=candidate_principles_str,
                demonstrations_str=demonstrations_str,
                user_query=ordered_instance["user_query"],
                given_response_1=ordered_instance["given_response_1"],
                given_response_2=ordered_instance["given_response_2"],
            )

            prompts_bridge.append(prompt_bridge)

        return prompts_bridge

    prompts_bridge = get_prompts_bridge(script_args, script_args.basic_prompt_principle, ordered_instances,
                                        all_candidate_principles)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token


    if "70B" in model_path or "72B" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=bnb_config)


    # if no BOS token, set as pad token, e.g. Qwen models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 计算 各个 样本 第 topk 层 原则节点概率值
    all_candidate_principles_probs = batch_compute_prob(script_args, tokenizer, model,
                                                        all_principles_list=all_candidate_principles,
                                                        prompts=prompts_bridge)

    prob_distributions = []
    for ins_idx, prompt_bridge in enumerate(prompts_bridge):
        candidate_principles = all_candidate_principles[ins_idx]
        candidate_principles_probs = all_candidate_principles_probs[ins_idx]
        # candidate_principles_probs = torch.softmax(candidate_principles_probs, dim=-1)

        candidate_principles_probs=candidate_principles_probs/torch.sum(candidate_principles_probs)


        # 找到前两个最大概率的索引和对应的概率
        topk_probs, topk_indices = torch.topk(candidate_principles_probs,
                                              len(script_args.dict_principle_to_demos.keys()))
        # 将张量转换为列表
        topk_probs = topk_probs.cpu().numpy().tolist()
        topk_indices = topk_indices.cpu().numpy().tolist()
        topk_principles = [candidate_principles[indice] for indice in topk_indices]

        prob_distribution = {
            "topk_probs": topk_probs,
            "topk_principles": topk_principles,
        }

        prob_distributions.append(prob_distribution)

    return prob_distributions


def contrastive_think(script_args, ordered_instances, principle_chains, llm, sampling_params):
    # 获得principle-aware contrastive_think statement
    prompts_for_positive_principle_aware_contrastive_think_statement = []
    prompts_for_negative_principle_aware_contrastive_think_statement = []

    for ins_idx, ordered_instance in enumerate(ordered_instances):
        # deepcopy以避免出现在真正的principle_chain中顺序被调换
        principle_chain = copy.deepcopy(principle_chains[ins_idx])
        # 倒序，将重要的放在靠后的位置
        principle_chain.reverse()

        demonstrations_str = ""
        principle_chain_desc = ""
        for principle in principle_chain:
            principle_chain_desc += script_args.dict_principles[principle]

            # 整理 demo 最终放在prompt中的格式
            principle_related_demo = script_args.dict_principle_to_demos[principle][0]  # 当前框架中，每个原则只有一个demo对应

            demo_user_query = principle_related_demo["user_query"]
            demo_relevant_principle = principle_related_demo["principle_desc"]

            demo_chosen_response = principle_related_demo["chosen_response"]
            demo_rejected_response = principle_related_demo["rejected_response"]

            demo_chosen_explanation = principle_related_demo["chosen_explanation"]
            demo_rejected_explanation = principle_related_demo["rejected_explanation"]

            positive_demo_explanation = demo_chosen_explanation.replace("This response",
                                                                        "The response 1") + demo_rejected_explanation.replace(
                "This response", "The response 2")
            negative_demo_explanation = demo_rejected_explanation.replace("This response",
                                                                          "The response 1") + demo_chosen_explanation.replace(
                "This response", "The response 2")

            demonstration_str = ""
            demonstration_str += f"<Instance>\n"
            demonstration_str += f"<User>{demo_user_query}</User>\n"
            demonstration_str += f"<Response 1 from AI>{demo_chosen_response}</Response 1 from AI>\n"
            demonstration_str += f"<Response 2 from AI>{demo_rejected_response}</Response 2 from AI>\n"
            demonstration_str += f"<Relevant Principle>{demo_relevant_principle}</Relevant Principle>\n"
            demonstration_str += f"<Explanation for Statement>{positive_demo_explanation}</Explanation for Statement>\n"
            demonstration_str += f"</Instance>\n"
            demonstration_str += f"<Instance>\n"
            demonstration_str += f"<User>{demo_user_query}</User>\n"
            demonstration_str += f"<Response 1 from AI>{demo_rejected_response}</Response 1 from AI>\n"
            demonstration_str += f"<Response 2 from AI>{demo_chosen_response}</Response 2 from AI>\n"
            demonstration_str += f"<Relevant Principle>{demo_relevant_principle}</Relevant Principle>\n"
            demonstration_str += f"<Explanation for Statement>{negative_demo_explanation}</Explanation for Statement>\n"
            demonstration_str += f"</Instance>\n"

            demonstrations_str += demonstration_str

        prompt_for_positive_principle_aware_contrastive_think_statement = script_args.basic_prompt_principle_aware_contrastive_think.format(
            statement="The response 1 is more consistent with the given principle than the response 2.",

            demonstrations_str=demonstrations_str,

            user_query=ordered_instance["user_query"],
            response1=ordered_instance["given_response_1"],
            response2=ordered_instance["given_response_2"],
            relevant_principle=principle_chain_desc,
        )

        prompt_for_negative_principle_aware_contrastive_think_statement = script_args.basic_prompt_principle_aware_contrastive_think.format(
            statement="The response 1 is less consistent with the given principle than the response 2.",

            demonstrations_str=demonstrations_str,

            user_query=ordered_instance["user_query"],
            response1=ordered_instance["given_response_1"],
            response2=ordered_instance["given_response_2"],
            relevant_principle=principle_chain_desc,
        )
        prompts_for_positive_principle_aware_contrastive_think_statement.append(
            prompt_for_positive_principle_aware_contrastive_think_statement)
        prompts_for_negative_principle_aware_contrastive_think_statement.append(
            prompt_for_negative_principle_aware_contrastive_think_statement)

    prompts_for_principle_aware_contrastive_think_statement = prompts_for_positive_principle_aware_contrastive_think_statement + prompts_for_negative_principle_aware_contrastive_think_statement

    outputs_contrastive_think_statement = llm.generate(prompts_for_principle_aware_contrastive_think_statement,
                                                       sampling_params)
    outputs_positive_contrastive_think_statement = outputs_contrastive_think_statement[
                                                   :len(ordered_instances)]
    outputs_negative_contrastive_think_statement = outputs_contrastive_think_statement[
                                                   len(ordered_instances):]

    contrastive_thoughts = []
    for ins_idx, ordered_instance in enumerate(ordered_instances):
        output_positive = outputs_positive_contrastive_think_statement[ins_idx]
        generated_text_positive = output_positive.outputs[0].text

        output_negative = outputs_negative_contrastive_think_statement[ins_idx]
        generated_text_negative = output_negative.outputs[0].text

        explanation_positive = generated_text_positive.replace("<Explanation for Statement>", "").replace(
            "</Explanation for Statement>", "").replace("</Instance>", "")
        explanation_negative = generated_text_negative.replace("<Explanation for Statement>", "").replace(
            "</Explanation for Statement>", "").replace("</Instance>", "")

        contrastive_thought = {
            "contrastive_think_statement_positive": explanation_positive,
            "contrastive_think_statement_negative": explanation_negative,
        }
        contrastive_thoughts.append(contrastive_thought)
    return contrastive_thoughts


def get_principle_chains(prob_distributions, principle_pointers, level_names):
    principle_chains = []
    for ins_idx in range(len(prob_distributions)):
        prob_distribution = prob_distributions[ins_idx]
        principle_pointer = principle_pointers[ins_idx]

        topk_principles = prob_distribution["topk_principles"]

        principle_chain = []
        for level_name in level_names:
            # 根据每一层的指针 找到每一层的原则
            principle_chain.append(topk_principles[principle_pointer[level_name]])
        principle_chains.append(principle_chain)
    return principle_chains


def prepare_judge_prompt(script_args, principle_related_demo, ordered_instance, principle_chain, thought_node):
    principle_chain_desc = ""
    for principle in principle_chain:
        principle_chain_desc += script_args.dict_principles[principle]

    demo_user_query = principle_related_demo["user_query"]
    demo_relevant_principle = principle_related_demo["principle_desc"]

    demo_chosen_response = principle_related_demo["chosen_response"]
    demo_rejected_response = principle_related_demo["rejected_response"]

    demo_chosen_explanation = principle_related_demo["chosen_explanation"]
    demo_rejected_explanation = principle_related_demo["rejected_explanation"]

    positive_demo_explanation = (demo_chosen_explanation.replace("This response", "The response 1")
                                 + demo_rejected_explanation.replace("This response", "The response 2"))
    negative_demo_explanation = (demo_rejected_explanation.replace("This response", "The response 1")
                                 + demo_chosen_explanation.replace("This response", "The response 2"))

    check_demo = ""
    check_demo += "<Instance>\n"
    check_demo += f"<User Query>\"{demo_user_query}\"</User Query>\n"
    check_demo += f"<Response 1 from AI>\"{demo_chosen_response}\"</Response 1 from AI>\n"
    check_demo += f"<Response 2 from AI>\"{demo_rejected_response}\"</Response 2 from AI>\n"
    check_demo += f"<Relevant Principle>\"{demo_relevant_principle}\"</Relevant Principle>\n"
    check_demo += f"<Reasoning Process>\"{positive_demo_explanation}\"</Reasoning Process>\n"
    check_demo += f"<Answer>Based on the reasoning process, we can conclude that the one that is more consistent with the given relevant principle is Response 1</Answer>\n"
    check_demo += "</Instance>\n"
    check_demo += "<Instance>\n"
    check_demo += f"<User Query>\"{demo_user_query}\"</User Query>\n"
    check_demo += f"<Response 1 from AI>\"{demo_rejected_response}\"</Response 1 from AI>\n"
    check_demo += f"<Response 2 from AI>\"{demo_chosen_response}\"</Response 2 from AI>\n"
    check_demo += f"<Relevant Principle>\"{demo_relevant_principle}\"</Relevant Principle>\n"
    check_demo += f"<Reasoning Process>\"{negative_demo_explanation}\"</Reasoning Process>\n"
    check_demo += f"<Answer>Based on the reasoning process, we can conclude that the one that is more consistent with the given relevant principle is Response 2</Answer>\n"
    check_demo += "</Instance>\n"

    user_query = ordered_instance["user_query"]
    given_response_1 = ordered_instance["given_response_1"]
    given_response_2 = ordered_instance["given_response_2"]

    judge_prompt = f"{check_demo}<Instance>\n<User Query>\"{user_query}\"</User Query>\n<Response 1 from AI>\"{given_response_1}\"</Response 1 from AI>\n<Response 2 from AI>\"{given_response_2}\"</Response 2 from AI>\n<Relevant Principle>\"{principle_chain_desc}\"</Relevant Principle>\n<Reasoning Process>\"{thought_node}\"</Reasoning Process>\n<Answer>Based on the reasoning process, we can conclude that the one that is more consistent with the given relevant principle is "

    return judge_prompt


def cal_info_score(script_args, ordered_instances, principle_chains, thought_nodes):
    reward_formatted_instances = []
    for ins_idx, ordered_instance in enumerate(ordered_instances):
        principle_chain = principle_chains[ins_idx]
        thought_node = thought_nodes[ins_idx]

        principle_related_demo = script_args.dict_principle_to_demos[principle_chain[0]][0]  # 当前框架中，每个原则只有一个demo对应
        judge_prompt = prepare_judge_prompt(script_args, principle_related_demo, ordered_instance, principle_chain,
                                            thought_node)
        text_chosen = f"Response 1"
        text_rejected = f"Response 2"

        reward_formatted_instance = {
            "prompt": judge_prompt,
            "text_chosen": text_chosen,
            "text_rejected": text_rejected
        }
        reward_formatted_instances.append(reward_formatted_instance)
    dataset = Dataset.from_list(reward_formatted_instances)
    script_args.tokenizer_builder = AutoTokenizer.from_pretrained
    script_args.model_builder = AutoModelForCausalLM.from_pretrained
    script_args.model_kwargs = {
        "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
    }

    chosen_scores, rejected_scores = compute_AI_feedback_single(script_args, script_args.strong_model_name_or_path,
                                                                dataset)

    info_scores = []
    for ins_idx, ordered_instance in enumerate(ordered_instances):
        response_1_score = chosen_scores[ins_idx]
        response_2_score = rejected_scores[ins_idx]
        h_d=-(1/2*math.log2(1/2)+1/2*math.log2(1/2))
        h_a_d=-(math.log2(response_1_score) * response_1_score + math.log2(response_2_score) * response_2_score)

        info_score_value = h_d - h_a_d

        info_score = {
            "response_1_score": response_1_score,
            "response_2_score": response_2_score,
            "info_score": info_score_value,
        }
        info_scores.append(info_score)

    return info_scores


def heuristic_tree_search(script_args):
    ordered_instances = read_ordered_instances(script_args)
    with open(script_args.tmp_filepath_prob_distributions, encoding='utf-8', mode='r') as f:
        prob_distributions = json.load(f)
    with open(script_args.tmp_filepath_principle_pointers, encoding='utf-8', mode='r') as f:
        principle_pointers = json.load(f)
    with open(script_args.tmp_filepath_backtrace_information_list, encoding='utf-8', mode='r') as f:
        backtrace_information_list = json.load(f)
    with open(script_args.tmp_filepath_contrastive_thoughts, encoding='utf-8', mode='r') as f:
        contrastive_thoughts = json.load(f)
    with open(script_args.tmp_filepath_info_scores, encoding='utf-8', mode='r') as f:
        info_scores = json.load(f)

    if script_args.stage == "stage_tree_searching_step1":
        # 第1步：进行第1层的对比思考结果,将结果记录在第1层对比思考文件中
        level1_principle_chains = get_principle_chains(prob_distributions=prob_distributions,
                                                       principle_pointers=principle_pointers, level_names=["level1"])

        llm, sampling_params = prepare_llm(llm_type=script_args.strong_model_name_or_path)
        level1_contrastive_thoughts = contrastive_think(script_args=script_args,
                                                        ordered_instances=ordered_instances,
                                                        principle_chains=level1_principle_chains,
                                                        llm=llm,
                                                        sampling_params=sampling_params)

        for ins_idx, level1_contrastive_thought in enumerate(level1_contrastive_thoughts):
            contrastive_thoughts[ins_idx]["level1"] = level1_contrastive_thought
        with open(script_args.tmp_filepath_contrastive_thoughts, encoding='utf-8', mode='w') as f:
            json.dump(contrastive_thoughts, f, indent=2)

    elif script_args.stage == "stage_tree_searching_step2":
        # 第2步：计算第1层信息熵，记录到信息熵表
        level1_principle_chains = get_principle_chains(prob_distributions=prob_distributions,
                                                       principle_pointers=principle_pointers, level_names=["level1"])
        level1_thought_nodes = [" ### ".join(contrastive_thought["level1"].values()).replace("\n", "") for
                                contrastive_thought in contrastive_thoughts]

        level1_info_scores = cal_info_score(script_args=script_args,
                                            ordered_instances=ordered_instances,
                                            principle_chains=level1_principle_chains,
                                            thought_nodes=level1_thought_nodes,
                                            )
        for ins_idx, level1_info_score in enumerate(level1_info_scores):
            info_scores[ins_idx]["level1"] = level1_info_score
        with open(script_args.tmp_filepath_info_scores, encoding='utf-8', mode='w') as f:
            json.dump(info_scores, f, indent=2)
        pass

    elif script_args.stage == "stage_tree_searching_step3":
        # 第3步：进行第2层的对比思考结果（初始生成）,将结果记录在第2层对比思考文件中
        level2_principle_chains = get_principle_chains(prob_distributions=prob_distributions,
                                                       principle_pointers=principle_pointers,
                                                       level_names=["level1", "level2"])

        llm, sampling_params = prepare_llm(llm_type=script_args.strong_model_name_or_path)
        level2_contrastive_thoughts = contrastive_think(script_args=script_args,
                                                        ordered_instances=ordered_instances,
                                                        principle_chains=level2_principle_chains,
                                                        llm=llm,
                                                        sampling_params=sampling_params)

        for ins_idx, level2_contrastive_thought in enumerate(level2_contrastive_thoughts):
            contrastive_thoughts[ins_idx]["level2"] = level2_contrastive_thought
        with open(script_args.tmp_filepath_contrastive_thoughts, encoding='utf-8', mode='w') as f:
            json.dump(contrastive_thoughts, f, indent=2)
        pass
    elif script_args.stage == "stage_tree_searching_step4":
        # 第4步：计算第2层信息熵，记录到信息熵表；
        level2_principle_chains = get_principle_chains(prob_distributions=prob_distributions,
                                                       principle_pointers=principle_pointers,
                                                       level_names=["level1", "level2"])
        level2_thought_nodes = [" ### ".join(contrastive_thought["level2"].values()).replace("\n", "") for
                                contrastive_thought in contrastive_thoughts]

        level2_info_scores = cal_info_score(script_args=script_args,
                                            ordered_instances=ordered_instances,
                                            principle_chains=level2_principle_chains,
                                            thought_nodes=level2_thought_nodes,
                                            )
        for ins_idx, level2_info_score in enumerate(level2_info_scores):
            info_scores[ins_idx]["level2"] = level2_info_score
        with open(script_args.tmp_filepath_info_scores, encoding='utf-8', mode='w') as f:
            json.dump(info_scores, f, indent=2)

        # 决定是否进行回溯：如果需要回溯则更新到回溯信息表，更新 原则指针列表
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            info_score = info_scores[ins_idx]
            info_gain_level21 = info_score["level2"]["info_score"] - info_score["level1"]["info_score"]
            if info_gain_level21 < 0:
                # 需要回溯
                # 完成: 1)加入到回溯信息表; 2)更新 原则指针列表;
                backtrace_information_list[ins_idx]["level2"] = {
                    "before_backtrace": {
                        "info_gain": info_gain_level21,
                        "contrastive_thought": contrastive_thoughts[ins_idx]["level2"],
                        "principle_chain": level2_principle_chains[ins_idx]
                    },
                    "after_backtrace": {}
                }
                principle_pointers[ins_idx]["level2"] = principle_pointers[ins_idx]["level2"] + 1
                principle_pointers[ins_idx]["level3"] = principle_pointers[ins_idx]["level3"] + 1

        with open(script_args.tmp_filepath_backtrace_information_list, encoding='utf-8', mode='w') as f:
            json.dump(backtrace_information_list, f, indent=2)
        with open(script_args.tmp_filepath_principle_pointers, encoding='utf-8', mode='w') as f:
            json.dump(principle_pointers, f, indent=2)

        pass
    elif script_args.stage == "stage_tree_searching_step5":
        # 第5步：根据回溯信息表中需要回溯的样本，重新生成第2层对比思考结果(依据更新后的原则指针列表)；
        # 将重新生成的结果也记录在回溯信息表中
        # 将重新生成的对比思考结果更新在第2层对比思考文件中
        backtrace_prob_distributions = []
        backtrace_principle_pointers = []
        backtrace_ordered_instances = []
        dict_backtrace_idx2ins_idx = {}
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            backtrace_information = backtrace_information_list[ins_idx]["level2"]
            if backtrace_information is not None:
                backtrace_idx = len(backtrace_prob_distributions)
                dict_backtrace_idx2ins_idx[backtrace_idx] = ins_idx

                backtrace_prob_distributions.append(prob_distributions[ins_idx])
                backtrace_principle_pointers.append(principle_pointers[ins_idx])
                backtrace_ordered_instances.append(ordered_instance)

        backtrace_level2_principle_chains = get_principle_chains(prob_distributions=backtrace_prob_distributions,
                                                                 principle_pointers=backtrace_principle_pointers,
                                                                 level_names=["level1", "level2"])

        llm, sampling_params = prepare_llm(llm_type=script_args.strong_model_name_or_path)
        backtrace_level2_contrastive_thoughts = contrastive_think(script_args=script_args,
                                                                  ordered_instances=backtrace_ordered_instances,
                                                                  principle_chains=backtrace_level2_principle_chains,
                                                                  llm=llm,
                                                                  sampling_params=sampling_params)
        for backtrace_idx, backtrace_level2_contrastive_thought in enumerate(backtrace_level2_contrastive_thoughts):
            ins_idx = dict_backtrace_idx2ins_idx[backtrace_idx]
            # 更新到回溯信息表
            backtrace_information_list[ins_idx]["level2"]["after_backtrace"] = {
                "info_gain": None,
                "contrastive_thought": backtrace_level2_contrastive_thought,
                "principle_chain": backtrace_level2_principle_chains[backtrace_idx]
            }
            # 将重新生成的对比思考结果更新在第2层对比思考文件中
            contrastive_thoughts[ins_idx]["level2"] = backtrace_level2_contrastive_thought

        with open(script_args.tmp_filepath_backtrace_information_list, encoding='utf-8', mode='w') as f:
            json.dump(backtrace_information_list, f, indent=2)
        with open(script_args.tmp_filepath_contrastive_thoughts, encoding='utf-8', mode='w') as f:
            json.dump(contrastive_thoughts, f, indent=2)

        pass
    elif script_args.stage == "stage_tree_searching_step6":
        # 第6步：根据回溯信息表中需要回溯的样本，重新计算信息熵并更新到   回溯信息表    信息表info_scores中
        backtrace_prob_distributions = []
        backtrace_principle_pointers = []
        backtrace_ordered_instances = []
        backtrace_contrastive_thoughts = []
        dict_backtrace_idx2ins_idx = {}
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            backtrace_information = backtrace_information_list[ins_idx]["level2"]
            if backtrace_information is not None:
                backtrace_idx = len(backtrace_prob_distributions)
                dict_backtrace_idx2ins_idx[backtrace_idx] = ins_idx

                backtrace_prob_distributions.append(prob_distributions[ins_idx])
                backtrace_principle_pointers.append(principle_pointers[ins_idx])
                backtrace_ordered_instances.append(ordered_instance)
                backtrace_contrastive_thoughts.append(contrastive_thoughts[ins_idx])

        backtrace_level2_principle_chains = get_principle_chains(prob_distributions=backtrace_prob_distributions,
                                                                 principle_pointers=backtrace_principle_pointers,
                                                                 level_names=["level1", "level2"])
        backtrace_level2_thought_nodes = [
            " ### ".join(backtrace_contrastive_thought["level2"].values()).replace("\n", "") for
            backtrace_contrastive_thought in backtrace_contrastive_thoughts]

        backtrace_level2_info_scores = cal_info_score(script_args=script_args,
                                                      ordered_instances=backtrace_ordered_instances,
                                                      principle_chains=backtrace_level2_principle_chains,
                                                      thought_nodes=backtrace_level2_thought_nodes,
                                                      )
        for backtrace_idx, backtrace_level2_info_score in enumerate(backtrace_level2_info_scores):
            ins_idx = dict_backtrace_idx2ins_idx[backtrace_idx]
            # 更新到回溯信息表
            new_info_gain = backtrace_level2_info_score["info_score"] - info_scores[ins_idx]["level1"]["info_score"]
            backtrace_information_list[ins_idx]["level2"]["after_backtrace"]["info_gain"] = new_info_gain

            # 更新到信息表info_scores
            info_scores[ins_idx]["level2"] = backtrace_level2_info_score

        with open(script_args.tmp_filepath_backtrace_information_list, encoding='utf-8', mode='w') as f:
            json.dump(backtrace_information_list, f, indent=2)
        with open(script_args.tmp_filepath_info_scores, encoding='utf-8', mode='w') as f:
            json.dump(info_scores, f, indent=2)

        pass




    elif script_args.stage == "stage_tree_searching_step7":
        # 第7步：进行第3层的对比思考结果；（初始生成）
        level3_principle_chains = get_principle_chains(prob_distributions=prob_distributions,
                                                       principle_pointers=principle_pointers,
                                                       level_names=["level1", "level2", "level3"])

        llm, sampling_params = prepare_llm(llm_type=script_args.strong_model_name_or_path)
        level3_contrastive_thoughts = contrastive_think(script_args=script_args,
                                                        ordered_instances=ordered_instances,
                                                        principle_chains=level3_principle_chains,
                                                        llm=llm,
                                                        sampling_params=sampling_params)

        for ins_idx, level3_contrastive_thought in enumerate(level3_contrastive_thoughts):
            contrastive_thoughts[ins_idx]["level3"] = level3_contrastive_thought
        with open(script_args.tmp_filepath_contrastive_thoughts, encoding='utf-8', mode='w') as f:
            json.dump(contrastive_thoughts, f, indent=2)

        pass
    elif script_args.stage == "stage_tree_searching_step8":
        # 第8步：计算第3层信息熵，记录到信息熵表；
        level3_principle_chains = get_principle_chains(prob_distributions=prob_distributions,
                                                       principle_pointers=principle_pointers,
                                                       level_names=["level1", "level2", "level3"])
        level3_thought_nodes = [" ### ".join(contrastive_thought["level3"].values()).replace("\n", "") for
                                contrastive_thought in contrastive_thoughts]

        level3_info_scores = cal_info_score(script_args=script_args,
                                            ordered_instances=ordered_instances,
                                            principle_chains=level3_principle_chains,
                                            thought_nodes=level3_thought_nodes,
                                            )
        for ins_idx, level3_info_score in enumerate(level3_info_scores):
            info_scores[ins_idx]["level3"] = level3_info_score
        with open(script_args.tmp_filepath_info_scores, encoding='utf-8', mode='w') as f:
            json.dump(info_scores, f, indent=2)

        # 决定是否进行回溯：如果需要回溯则更新到回溯信息表，更新 原则指针列表
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            info_score = info_scores[ins_idx]
            info_gain_level32 = info_score["level3"]["info_score"] - info_score["level2"]["info_score"]
            if info_gain_level32 < 0:
                # 需要回溯
                # 完成: 1)加入到回溯信息表; 2)更新 原则指针列表;
                backtrace_information_list[ins_idx]["level3"] = {
                    "before_backtrace": {
                        "info_gain": info_gain_level32,
                        "contrastive_thought": contrastive_thoughts[ins_idx]["level3"],
                        "principle_chain": level3_principle_chains[ins_idx]
                    },
                    "after_backtrace": {}
                }
                principle_pointers[ins_idx]["level3"] = principle_pointers[ins_idx]["level3"] + 1

        with open(script_args.tmp_filepath_backtrace_information_list, encoding='utf-8', mode='w') as f:
            json.dump(backtrace_information_list, f, indent=2)
        with open(script_args.tmp_filepath_principle_pointers, encoding='utf-8', mode='w') as f:
            json.dump(principle_pointers, f, indent=2)


        pass
    elif script_args.stage == "stage_tree_searching_step9":
        # 第9步：根据回溯信息表中需要回溯的样本，重新生成第3层对比思考结果(依据更新后的原则指针列表)；
        # 将重新生成的结果也记录在回溯信息表中
        # 将重新生成的对比思考结果更新在第3层对比思考文件中
        backtrace_prob_distributions = []
        backtrace_principle_pointers = []
        backtrace_ordered_instances = []
        dict_backtrace_idx2ins_idx = {}
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            backtrace_information = backtrace_information_list[ins_idx]["level3"]
            if backtrace_information is not None:
                backtrace_idx = len(backtrace_prob_distributions)
                dict_backtrace_idx2ins_idx[backtrace_idx] = ins_idx

                backtrace_prob_distributions.append(prob_distributions[ins_idx])
                backtrace_principle_pointers.append(principle_pointers[ins_idx])
                backtrace_ordered_instances.append(ordered_instance)

        backtrace_level3_principle_chains = get_principle_chains(prob_distributions=backtrace_prob_distributions,
                                                                 principle_pointers=backtrace_principle_pointers,
                                                                 level_names=["level1", "level2", "level3"])

        llm, sampling_params = prepare_llm(llm_type=script_args.strong_model_name_or_path)
        backtrace_level3_contrastive_thoughts = contrastive_think(script_args=script_args,
                                                                  ordered_instances=backtrace_ordered_instances,
                                                                  principle_chains=backtrace_level3_principle_chains,
                                                                  llm=llm,
                                                                  sampling_params=sampling_params)
        for backtrace_idx, backtrace_level3_contrastive_thought in enumerate(backtrace_level3_contrastive_thoughts):
            ins_idx = dict_backtrace_idx2ins_idx[backtrace_idx]
            # 更新到回溯信息表
            backtrace_information_list[ins_idx]["level3"]["after_backtrace"] = {
                "info_gain": None,
                "contrastive_thought": backtrace_level3_contrastive_thought,
                "principle_chain": backtrace_level3_principle_chains[backtrace_idx]
            }
            # 将重新生成的对比思考结果更新在第2层对比思考文件中
            contrastive_thoughts[ins_idx]["level3"] = backtrace_level3_contrastive_thought

        with open(script_args.tmp_filepath_backtrace_information_list, encoding='utf-8', mode='w') as f:
            json.dump(backtrace_information_list, f, indent=2)
        with open(script_args.tmp_filepath_contrastive_thoughts, encoding='utf-8', mode='w') as f:
            json.dump(contrastive_thoughts, f, indent=2)

        pass
    elif script_args.stage == "stage_tree_searching_step10":
        # 第10步：根据回溯信息表中需要回溯的样本，重新计算信息熵并更新到信息熵中
        backtrace_prob_distributions = []
        backtrace_principle_pointers = []
        backtrace_ordered_instances = []
        backtrace_contrastive_thoughts = []
        dict_backtrace_idx2ins_idx = {}
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            backtrace_information = backtrace_information_list[ins_idx]["level3"]
            if backtrace_information is not None:
                backtrace_idx = len(backtrace_prob_distributions)
                dict_backtrace_idx2ins_idx[backtrace_idx] = ins_idx

                backtrace_prob_distributions.append(prob_distributions[ins_idx])
                backtrace_principle_pointers.append(principle_pointers[ins_idx])
                backtrace_ordered_instances.append(ordered_instance)
                backtrace_contrastive_thoughts.append(contrastive_thoughts[ins_idx])

        backtrace_level3_principle_chains = get_principle_chains(prob_distributions=backtrace_prob_distributions,
                                                                 principle_pointers=backtrace_principle_pointers,
                                                                 level_names=["level1", "level2", "level3"])
        backtrace_level3_thought_nodes = [
            " ### ".join(backtrace_contrastive_thought["level3"].values()).replace("\n", "") for
            backtrace_contrastive_thought in backtrace_contrastive_thoughts]

        backtrace_level3_info_scores = cal_info_score(script_args=script_args,
                                                      ordered_instances=backtrace_ordered_instances,
                                                      principle_chains=backtrace_level3_principle_chains,
                                                      thought_nodes=backtrace_level3_thought_nodes,
                                                      )
        for backtrace_idx, backtrace_level3_info_score in enumerate(backtrace_level3_info_scores):
            ins_idx = dict_backtrace_idx2ins_idx[backtrace_idx]
            # 更新到回溯信息表
            new_info_gain = backtrace_level3_info_score["info_score"] - info_scores[ins_idx]["level2"]["info_score"]
            backtrace_information_list[ins_idx]["level3"]["after_backtrace"]["info_gain"] = new_info_gain

            # 更新到信息表info_scores
            info_scores[ins_idx]["level3"] = backtrace_level3_info_score

        with open(script_args.tmp_filepath_backtrace_information_list, encoding='utf-8', mode='w') as f:
            json.dump(backtrace_information_list, f, indent=2)
        with open(script_args.tmp_filepath_info_scores, encoding='utf-8', mode='w') as f:
            json.dump(info_scores, f, indent=2)

        pass





    elif script_args.stage == "stage_tree_searching_collection":
        # 步骤：将上面的文件进行整合，整合为一个principle tree文件
        def generate_fast_and_frugal_tree(script_args, ordered_instance, prob_distribution, principle_pointer, contrastive_thought, info_score):

            topk_probs=prob_distribution["topk_probs"]
            topk_principles=prob_distribution["topk_principles"]

            principle_pointer_level1=principle_pointer["level1"]
            principle_pointer_level2=principle_pointer["level2"]
            principle_pointer_level3=principle_pointer["level3"]

            level1_principle_confidence=topk_probs[principle_pointer_level1]
            level2_principle_confidence=topk_probs[principle_pointer_level2]
            level3_principle_confidence=topk_probs[principle_pointer_level3]

            level1_principle_node=topk_principles[principle_pointer_level1]
            level2_principle_node=topk_principles[principle_pointer_level2]
            level3_principle_node=topk_principles[principle_pointer_level3]

            level1_thought_node=" ### ".join(contrastive_thought["level1"].values()).replace("\n","")
            level2_thought_node=" ### ".join(contrastive_thought["level2"].values()).replace("\n","")
            level3_thought_node=" ### ".join(contrastive_thought["level3"].values()).replace("\n","")

            level1_info_gain=info_score["level1"]["info_score"]-0
            level2_info_gain=info_score["level2"]["info_score"]-info_score["level1"]["info_score"]
            level3_info_gain=info_score["level3"]["info_score"]-info_score["level2"]["info_score"]

            fast_and_frugal_tree = {
                "user_query": ordered_instance["user_query"],
                "given_response_1": ordered_instance["given_response_1"],
                "given_response_2": ordered_instance["given_response_2"],
                "golden_preference": ordered_instance["golden_preference"],
                "level1": {
                    "principle_confidence": level1_principle_confidence,
                    "principle_node": level1_principle_node,
                    "thought_information_enrichment_score": level1_info_gain,
                    "thought_node": level1_thought_node,
                    "info_score":info_score["level1"],
                },
                "level2": {
                    "principle_confidence": level2_principle_confidence,
                    "principle_node": level2_principle_node,
                    "thought_information_enrichment_score": level2_info_gain,
                    "thought_node": level2_thought_node,
                    "info_score": info_score["level2"],
                },
                "level3": {
                    "principle_confidence": level3_principle_confidence,
                    "principle_node": level3_principle_node,
                    "thought_information_enrichment_score": level3_info_gain,
                    "thought_node": level3_thought_node,
                    "info_score": info_score["level3"],
                },
            }

            return fast_and_frugal_tree


        fast_and_frugal_trees = []
        for ins_idx in range(len(ordered_instances)):
            prob_distribution=prob_distributions[ins_idx]
            principle_pointer=principle_pointers[ins_idx]
            contrastive_thought=contrastive_thoughts[ins_idx]
            info_score=info_scores[ins_idx]

            fast_and_frugal_tree = generate_fast_and_frugal_tree(script_args,
                                                                 ordered_instances[ins_idx],
                                                                 prob_distribution,
                                                                 principle_pointer,
                                                                 contrastive_thought,
                                                                 info_score)

            fast_and_frugal_trees.append(fast_and_frugal_tree)

        with open(script_args.tmp_fast_and_frugal_trees_filepath, encoding='utf-8', mode='w') as f:
            json.dump(fast_and_frugal_trees, f, indent=2)

    pass


def heuristic_tree_annotation(script_args):
    ordered_instances = read_ordered_instances(script_args)

    def prepare_judge_prompt(script_args, principle_related_demo, ordered_instance, principle_chain, thought_node):
        principle_chain_desc = ""
        for principle in principle_chain:
            principle_chain_desc += script_args.dict_principles[principle]

        demo_user_query = principle_related_demo["user_query"]
        demo_relevant_principle = principle_related_demo["principle_desc"]

        demo_chosen_response = principle_related_demo["chosen_response"]
        demo_rejected_response = principle_related_demo["rejected_response"]

        demo_chosen_explanation = principle_related_demo["chosen_explanation"]
        demo_rejected_explanation = principle_related_demo["rejected_explanation"]

        positive_demo_explanation = (demo_chosen_explanation.replace("This response", "The response 1")
                                     + demo_rejected_explanation.replace("This response", "The response 2"))
        negative_demo_explanation = (demo_rejected_explanation.replace("This response", "The response 1")
                                     + demo_chosen_explanation.replace("This response", "The response 2"))

        check_demo = ""
        check_demo += "<Instance>\n"
        check_demo += f"<User Query>\"{demo_user_query}\"</User Query>\n"
        check_demo += f"<Response 1 from AI>\"{demo_chosen_response}\"</Response 1 from AI>\n"
        check_demo += f"<Response 2 from AI>\"{demo_rejected_response}\"</Response 2 from AI>\n"
        check_demo += f"<Relevant Principle>\"{demo_relevant_principle}\"</Relevant Principle>\n"
        check_demo += f"<Reasoning Process>\"{positive_demo_explanation}\"</Reasoning Process>\n"
        check_demo += f"<Answer>Based on the reasoning process, we can conclude that the one that is more consistent with the given relevant principle is Response 1</Answer>\n"
        check_demo += "</Instance>\n"
        check_demo += "<Instance>\n"
        check_demo += f"<User Query>\"{demo_user_query}\"</User Query>\n"
        check_demo += f"<Response 1 from AI>\"{demo_rejected_response}\"</Response 1 from AI>\n"
        check_demo += f"<Response 2 from AI>\"{demo_chosen_response}\"</Response 2 from AI>\n"
        check_demo += f"<Relevant Principle>\"{demo_relevant_principle}\"</Relevant Principle>\n"
        check_demo += f"<Reasoning Process>\"{negative_demo_explanation}\"</Reasoning Process>\n"
        check_demo += f"<Answer>Based on the reasoning process, we can conclude that the one that is more consistent with the given relevant principle is Response 2</Answer>\n"
        check_demo += "</Instance>\n"

        user_query = ordered_instance["user_query"]
        given_response_1 = ordered_instance["given_response_1"]
        given_response_2 = ordered_instance["given_response_2"]

        judge_prompt = f"{check_demo}<Instance>\n<User Query>\"{user_query}\"</User Query>\n<Response 1 from AI>\"{given_response_1}\"</Response 1 from AI>\n<Response 2 from AI>\"{given_response_2}\"</Response 2 from AI>\n<Relevant Principle>\"{principle_chain_desc}\"</Relevant Principle>\n<Reasoning Process>\"{thought_node}\"</Reasoning Process>\n<Answer>Based on the reasoning process, we can conclude that the one that is more consistent with the given relevant principle is "

        return judge_prompt

    with open(script_args.tmp_fast_and_frugal_trees_filepath, encoding='utf-8', mode='r') as f:
        fast_and_frugal_trees = json.load(f)

    # all_level1_thoughts = []
    # all_level2_thoughts = []
    # all_level3_thoughts = []

    # all_level1_principle_chains = []
    # all_level2_principle_chains = []
    # all_level3_principle_chains = []

    reward_formatted_instances = []

    for ins_idx, fast_and_frugal_tree in enumerate(fast_and_frugal_trees):
        level1_thought = fast_and_frugal_tree["level1"]["thought_node"]
        level2_thought = fast_and_frugal_tree["level2"]["thought_node"]
        level3_thought = fast_and_frugal_tree["level3"]["thought_node"]

        # all_level1_thoughts.append(level1_thought)
        # all_level2_thoughts.append(level2_thought)
        # all_level3_thoughts.append(level3_thought)

        level1_principle = fast_and_frugal_tree["level1"]["principle_node"]
        level2_principle = fast_and_frugal_tree["level2"]["principle_node"]
        level3_principle = fast_and_frugal_tree["level3"]["principle_node"]
        level1_principle_chain = [level1_principle]
        level2_principle_chain = [level1_principle, level2_principle]
        level3_principle_chain = [level1_principle, level2_principle, level3_principle]

        # all_level1_principle_chains.append(level1_principle_chain)
        # all_level2_principle_chains.append(level2_principle_chain)
        # all_level3_principle_chains.append(level3_principle_chain)

        # 默认以 level1_principle 找 demo
        principle_related_demo = script_args.dict_principle_to_demos[level1_principle][0]  # 当前框架中，每个原则只有一个demo对应

        level1_judge_prompt = prepare_judge_prompt(script_args, principle_related_demo, ordered_instances[ins_idx],
                                                   level1_principle_chain, level1_thought)
        level2_judge_prompt = prepare_judge_prompt(script_args, principle_related_demo, ordered_instances[ins_idx],
                                                   level2_principle_chain, level2_thought)
        level3_judge_prompt = prepare_judge_prompt(script_args, principle_related_demo, ordered_instances[ins_idx],
                                                   level3_principle_chain, level3_thought)
        text_chosen = f"Response 1"
        text_rejected = f"Response 2"

        level1_reward_formatted_instance = {
            "prompt": level1_judge_prompt,
            "text_chosen": text_chosen,
            "text_rejected": text_rejected
        }
        level2_reward_formatted_instance = {
            "prompt": level2_judge_prompt,
            "text_chosen": text_chosen,
            "text_rejected": text_rejected
        }
        level3_reward_formatted_instance = {
            "prompt": level3_judge_prompt,
            "text_chosen": text_chosen,
            "text_rejected": text_rejected
        }
        reward_formatted_instances.append(level1_reward_formatted_instance)
        reward_formatted_instances.append(level2_reward_formatted_instance)
        reward_formatted_instances.append(level3_reward_formatted_instance)
    dataset = Dataset.from_list(reward_formatted_instances)

    script_args.tokenizer_builder = AutoTokenizer.from_pretrained
    script_args.model_builder = AutoModelForCausalLM.from_pretrained
    script_args.model_kwargs = {
        "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
    }

    chosen_scores, rejected_scores = compute_AI_feedback_single(script_args, script_args.weak_model_name_or_path,
                                                                dataset)

    judged_fast_and_frugal_trees = []
    for ins_idx, fast_and_frugal_tree in enumerate(fast_and_frugal_trees):
        batched_chosen_scores = chosen_scores[ins_idx * 3:ins_idx * 3 + 3]
        batched_rejected_scores = rejected_scores[ins_idx * 3:ins_idx * 3 + 3]
        level1_response1_score = batched_chosen_scores[0]
        level2_response1_score = batched_chosen_scores[1]
        level3_response1_score = batched_chosen_scores[2]
        level1_response2_score = batched_rejected_scores[0]
        level2_response2_score = batched_rejected_scores[1]
        level3_response2_score = batched_rejected_scores[2]

        judged_fast_and_frugal_tree = copy.deepcopy(fast_and_frugal_tree)
        judged_fast_and_frugal_tree["level1"]["response1_score"] = level1_response1_score
        judged_fast_and_frugal_tree["level1"]["response2_score"] = level1_response2_score

        judged_fast_and_frugal_tree["level2"]["response1_score"] = level2_response1_score
        judged_fast_and_frugal_tree["level2"]["response2_score"] = level2_response2_score

        judged_fast_and_frugal_tree["level3"]["response1_score"] = level3_response1_score
        judged_fast_and_frugal_tree["level3"]["response2_score"] = level3_response2_score
        judged_fast_and_frugal_trees.append(judged_fast_and_frugal_tree)

    with open(script_args.judged_fast_and_frugal_trees_filepath, encoding='utf-8', mode='w') as f:
        json.dump(judged_fast_and_frugal_trees, f, indent=2)


if __name__ == '__main__':
    ############## logs ##############
    # 需要记录到三个文件夹，一个是记录过程；还有一个记录预测metric情况；一个是记录预测结果（一个模型有独特的order，所以可以直接保存暂用，一个文件即可）；
    dirs = ["response_record", "metric_results", "prediction_results", "tmp"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    ############## logs ##############

    script_args = get_args()
    set_seed(script_args.seed)

    # 读取 principles 和 principle demos相关的内容；并作为参数的一部分跟随整个框架
    script_args.dict_principles, script_args.dict_principle_to_demos = get_principle_info()

    # 读取所需要的prompt模板文件
    with open('prompts/basic_prompt_principle.txt', mode='r') as f:
        script_args.basic_prompt_principle = f.read()
    with open('prompts/basic_prompt_principle_aware_contrastive_think.txt', mode='r') as f:
        script_args.basic_prompt_principle_aware_contrastive_think = f.read()


    # script_args.run_name=f"[{script_args.time}]-SPLIT[{script_args.split_mode}]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}]-DATA[{script_args.dataset_name}]-HeldOutSIZE[{str(script_args.held_out_sample_num)}]"
    script_args.run_name = f"SPLIT[{script_args.split_mode}]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}]-DATA[{script_args.dataset_name}]-HeldOutSIZE[{str(script_args.held_out_sample_num)}]"

    # record_file_path = f"response_record/{script_args.run_name}.log"
    # record_file_path = open(record_file_path, encoding='utf-8', mode='a')

    # script_args.tmp_filepath_level1_principle_chains = f'tmp/level1_principle_chains-RUN[{script_args.run_name}].json'
    # script_args.tmp_filepath_level2_principle_chains = f'tmp/level2_principle_chains-RUN[{script_args.run_name}].json'
    # script_args.tmp_filepath_level3_principle_chains = f'tmp/level3_principle_chains-RUN[{script_args.run_name}].json'

    # script_args.tmp_filepath_level1_contrastive_thoughts = f'tmp/level1_contrastive_thoughts-RUN[{script_args.run_name}].json'
    # script_args.tmp_filepath_level2_contrastive_thoughts = f'tmp/level2_contrastive_thoughts-RUN[{script_args.run_name}].json'
    # script_args.tmp_filepath_level3_contrastive_thoughts = f'tmp/level3_contrastive_thoughts-RUN[{script_args.run_name}].json'

    script_args.tmp_fast_and_frugal_trees_filepath = f'tmp/fast_and_frugal_trees-RUN[{script_args.run_name}].json'
    script_args.judged_fast_and_frugal_trees_filepath = f'prediction_results/judged_fast_and_frugal_trees-RUN[{script_args.run_name}].json'

    script_args.tmp_filepath_prob_distributions = f'tmp/prob_distributions-RUN[{script_args.run_name}].json'
    script_args.tmp_filepath_principle_pointers = f'tmp/principle_pointers-RUN[{script_args.run_name}].json'  # 指向每一层的
    script_args.tmp_filepath_backtrace_information_list = f'tmp/backtrace_information_list-RUN[{script_args.run_name}].json'

    script_args.tmp_filepath_contrastive_thoughts = f'tmp/contrastive_thoughts-RUN[{script_args.run_name}].json'
    script_args.tmp_filepath_info_scores = f'tmp/info_scores-RUN[{script_args.run_name}].json'

    if "stage_calculate_prob" in script_args.stage:
        ordered_instances = read_ordered_instances(script_args)

        # 步骤0：计算每个principle的概率，作为后续搜索的依据；via strong；
        prob_distributions = calculate_probs(script_args=script_args,
                                             model_path=script_args.strong_model_name_or_path,
                                             ordered_instances=ordered_instances,
                                             all_candidate_principles=[list(script_args.dict_principle_to_demos.keys())
                                                                       for _ in range(len(ordered_instances))])
        with open(script_args.tmp_filepath_prob_distributions, encoding='utf-8', mode='w') as f:
            json.dump(prob_distributions, f, indent=2)

        # 初始化一个 principle_pointers
        principle_pointers = []
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            # 默认情况下，第一层取 top1原则... 等
            principle_pointer = {
                "level1": 0,
                "level2": 1,
                "level3": 2,
            }
            principle_pointers.append(principle_pointer)
        with open(script_args.tmp_filepath_principle_pointers, encoding='utf-8', mode='w') as f:
            json.dump(principle_pointers, f, indent=2)

        # 初始化一个 回溯信息表
        backtrace_information_list = []
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            # 默认情况下，每一层都不存在回溯
            backtrace_information = {
                "level1": None,
                "level2": None,
                "level3": None,
            }
            backtrace_information_list.append(backtrace_information)
        with open(script_args.tmp_filepath_backtrace_information_list, encoding='utf-8', mode='w') as f:
            json.dump(backtrace_information_list, f, indent=2)

        # 初始化一个 记录各个层的对比思考结果的表
        contrastive_thoughts = []
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            # 默认情况下，每一层都不存在回溯
            contrastive_thought = {
                "level1": None,
                "level2": None,
                "level3": None,
            }
            contrastive_thoughts.append(contrastive_thought)
        with open(script_args.tmp_filepath_contrastive_thoughts, encoding='utf-8', mode='w') as f:
            json.dump(contrastive_thoughts, f, indent=2)

        # 初始化一个 记录各个层的 信息量 的列表
        info_scores = []
        for ins_idx, ordered_instance in enumerate(ordered_instances):
            # 默认情况下，每一层都不存在回溯
            info_score = {
                "level1": None,
                "level2": None,
                "level3": None,
            }
            info_scores.append(info_score)
        with open(script_args.tmp_filepath_info_scores, encoding='utf-8', mode='w') as f:
            json.dump(info_scores, f, indent=2)

    if "stage_tree_searching" in script_args.stage:
        # 启发式边搜索边思考；via strong；
        heuristic_tree_search(script_args)

    if "stage_tree_annotation" in script_args.stage:
        # 获取树的信息，进行间接标注；via weak；
        heuristic_tree_annotation(script_args)
