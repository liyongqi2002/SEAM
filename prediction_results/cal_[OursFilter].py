import argparse
import json
import os

from transformers import AutoTokenizer

from utils import cal_HeldOut_metric

if __name__ == '__main__':
    harmlessness_list = [
        "Law-abiding",
        "No Risk Information",
        "Privacy Protection",
        "No NSFW Content",
        "Objective",
        "Fairness and Kindness",
    ]
    parser = argparse.ArgumentParser()
    # Model Name and Path
    parser.add_argument("--weak_model_name", default='Qwen2-1.5B-Instruct', type=str, help="Qwen2-1.5B-Instruct")
    parser.add_argument("--strong_model_name", default='Qwen2-7B', type=str, help="Qwen2-7B")
    script_args = parser.parse_args()
    weak_model_name=script_args.weak_model_name
    strong_model_name=script_args.strong_model_name

    # 需要将baseline的预测结果记录到 baseline_prediction_results 中；以方便后续实验
    dirs = ["baseline_prediction_results", "../metric_results/baseline_metric_results"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    dataset_names=['AHelpful', 'HelpSteer', 'AHarmless', 'CaiHarmless','AnthropicHH','SafeRLHF']
    # dataset_names=['AHelpful', 'AHarmless', 'AnthropicHH']


    for dataset_name in dataset_names:
        strong_pred_file_path = f"ordered_SPLIT[HeldOut]-M[{strong_model_name}]-D[{dataset_name}].json"
        with open(strong_pred_file_path, encoding='utf-8', mode='r') as f:
            raw_strong_pred_instances = json.load(f)

        file_path = f"judged_fast_and_frugal_trees-RUN[SPLIT[HeldOut]-WEAK[{weak_model_name}]-STRONG[{strong_model_name}]-DATA[{dataset_name}]-HeldOutSIZE[5000]].json"

        with open(file_path, encoding='utf-8', mode='r') as f:
            raw_instances = json.load(f)

        trees_info_score = []
        for ins_idx, instance in enumerate(raw_instances):
            levels_info_score = []
            for level_name in ["level1", "level2", "level3"]:
                level_enrichment_score = instance[level_name]["thought_information_enrichment_score"]
                level_info_score = instance[level_name]["info_score"]["info_score"]
                if level_enrichment_score > 0:
                    levels_info_score.append(level_info_score)
                else:
                    if level_name == "level1":
                        levels_info_score.append(level_info_score)
                    else:
                        break
            tree_info_score = sum(levels_info_score) / len(levels_info_score)
            trees_info_score.append(tree_info_score)

        sorted_trees_info_score = sorted(trees_info_score)
        n = len(trees_info_score)
        # 如果列表长度是奇数
        if n % 2 == 1:
            pi_median = sorted_trees_info_score[n // 2]
        # 如果列表长度是偶数
        else:
            mid1 = sorted_trees_info_score[n // 2 - 1]
            mid2 = sorted_trees_info_score[n // 2]
            pi_median = (mid1 + mid2) / 2


        instances = []
        strong_pred_instances = []
        for instance, tree_info_score, raw_strong_pred_instance in zip(raw_instances, trees_info_score,
                                                                       raw_strong_pred_instances):
            # 信息熵值越小，不确定性越低，越应该被保留
            if tree_info_score >= pi_median:
                instances.append(instance)
                strong_pred_instances.append(raw_strong_pred_instance)

        num_true_level1 = 0
        num_true_level2 = 0
        num_true_level3 = 0
        num_true_tree_pred = 0

        tree_depths = []
        num_type1_tree = 0
        num_type2_tree = 0

        annotated_instances=[]
        for ins_idx, instance in enumerate(instances):
            golden_preference = instance["golden_preference"]

            pred_level1 = "given_response_1" if (instance["level1"]["response1_score"] >
                                                 instance["level1"]["response2_score"]) else "given_response_2"
            pred_level2 = "given_response_1" if (instance["level2"]["response1_score"] >
                                                 instance["level2"]["response2_score"]) else "given_response_2"
            pred_level3 = "given_response_1" if (instance["level3"]["response1_score"] >
                                                 instance["level3"]["response2_score"]) else "given_response_2"

            # 只保留有信息增益的节点
            tree_depth = 0
            principle_chain = []
            for level_name in ["level1", "level2", "level3"]:
                level_enrichment_score = instance[level_name]["thought_information_enrichment_score"]
                level_principle_node = instance[level_name]["principle_node"]
                if level_enrichment_score > 0:
                    principle_chain.append(level_principle_node)
                    tree_depth += 1
                else:
                    if level_name == "level1":
                        # 特殊情况下，在第一层，即使信息增益很少，也增加到链条
                        principle_chain.append(level_principle_node)
                        tree_depth += 1
                    else:
                        break
            tree_depths.append(tree_depth)

            if tree_depth == 1:
                tree_pred = pred_level1
            elif tree_depth == 2:
                tree_pred = pred_level2
            elif tree_depth == 3:
                tree_pred = pred_level3

            #############################################################
            # 方案1：tree_pred 按照剪枝后的树做树判断
            flag = False
            for principle_node in principle_chain:
                if principle_node in harmlessness_list:
                    flag = True

            num_type1_tree += 1 if not flag else 0
            num_type2_tree += 1 if flag else 0


            #############################################################
            pred_response_strong = strong_pred_instances[ins_idx][
                strong_pred_instances[ins_idx]["reward_score_pred_preference"]]
            if pred_response_strong == instance["given_response_1"]:
                pred_strong = "given_response_1"
            else:
                pred_strong = "given_response_2"

            tree_pred = pred_strong if not flag else tree_pred
            pred_level1 = pred_strong if not flag else pred_level1
            pred_level2 = pred_strong if not flag else pred_level2
            pred_level3 = pred_strong if not flag else pred_level3

            num_true_tree_pred+=1 if tree_pred == golden_preference else 0
            num_true_level1+=1 if pred_level1 == golden_preference else 0
            num_true_level2+=1 if pred_level2 == golden_preference else 0
            num_true_level3+=1 if pred_level3 == golden_preference else 0


            annotated_instance = {
                "user_query": instance["user_query"],
                "given_response_1": instance["given_response_1"],
                "given_response_2": instance["given_response_2"],
                "golden_preference": instance["golden_preference"],
                "annotated_preference": tree_pred,
            }
            annotated_instances.append(annotated_instance)

        print(f"{dataset_name}")
        print(f"过滤前样本数目：{len(raw_instances)}")
        print(f"过滤后样本数目：{len(instances)}")
        print("num_true_level1", num_true_level1 / len(instances))
        print("num_true_level2", num_true_level2 / len(instances))
        print("num_true_level3", num_true_level3 / len(instances))
        print("---")
        print("num_true_tree_pred", num_true_tree_pred / len(instances))
        print("tree_depth avg", sum(tree_depths) / len(tree_depths))
        print("num_type1_tree", num_type1_tree / len(tree_depths))
        print("num_type2_tree", num_type2_tree / len(tree_depths))
        print('=============================================')

        HeldOut_metric=cal_HeldOut_metric(annotated_instances)
        metric_record_file_path = f"../metric_results/W2SHeldOut-Method[OursFilter]-W[{weak_model_name}]-S[{strong_model_name}].log"
        metric_record_file_path = open(metric_record_file_path, encoding='utf-8', mode='a')
        print("====================", file=metric_record_file_path)
        print("OursFilter HeldOut Results", file=metric_record_file_path)
        print(f"{dataset_name} ### {HeldOut_metric}\n", file=metric_record_file_path)

        target_filepath_annotation_results=f"./HeldOutAnnotation-Method[OursFilter]-WEAK[{weak_model_name}]-STRONG[{strong_model_name}]-DATA[{dataset_name}].json"
        with open(target_filepath_annotation_results, encoding='utf-8', mode='w') as f:
            json.dump(annotated_instances, f, indent=2)
