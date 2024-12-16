import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Name and Path
    parser.add_argument("--weak_model_name", default='Qwen2-1.5B-Instruct', type=str, help="Qwen2-1.5B-Instruct")
    parser.add_argument("--strong_model_name", default='Qwen2-7B', type=str, help="Qwen2-7B")
    script_args = parser.parse_args()
    weak_model_name=script_args.weak_model_name
    strong_model_name=script_args.strong_model_name

    # 需要将baseline的预测结果记录到 baseline_prediction_results 中；以方便后续实验
    dirs = ["dpo_data"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


    # methods=["Ours","OursFilter","weak","WeakSelf","StrongSelf","Ensemble","Burns","UFilter","WSConsistency","StrongCeiling",]
    methods=["RebuttalCotWithDefinition","RebuttalDebate"]
    for method in methods:

        dpo_instances=[]

        dataset_names=['AHelpful', 'AHarmless', 'AnthropicHH']

        for dataset_name in dataset_names:
            prediction_results_file = f'../prediction_results/W2SEval-RM_DataMode[{method}]-W[{weak_model_name}]-S[{strong_model_name}]-D[{dataset_name}]-HeldOutSIZE[{str(5000)}].json'

            with open(prediction_results_file, encoding='utf-8', mode='r') as f:
                eval_result_instances = json.load(f)
            for eval_result_instance in eval_result_instances:
                pred_key=f"{method}_pred_preference"
                pred=eval_result_instance[pred_key]

                user_query=eval_result_instance["user_query"]
                given_response_1=eval_result_instance["given_response_1"]
                given_response_2=eval_result_instance["given_response_2"]

                if pred=="given_response_1":
                    chosen_response=given_response_1
                    rejected_response=given_response_2
                else:
                    chosen_response=given_response_2
                    rejected_response=given_response_1

                if chosen_response == "":
                    chosen_response = "..."
                if rejected_response == "":
                    rejected_response = "..."


                dpo_instance={
                    "prompt": f"Question: {user_query}\n\nAnswer: ",
                    "chosen":f"{chosen_response}",
                    "rejected": f"{rejected_response}",
                }
                dpo_instances.append(dpo_instance)

        # print(len(dpo_instances))
        # print(dpo_instances[0])

        target_dpo_filepath=f"./dpo_data/DPOInstances-Method[{method}]-WEAK[{weak_model_name}]-STRONG[{strong_model_name}].json"
        with open(target_dpo_filepath, encoding='utf-8', mode='w') as f:
            json.dump(dpo_instances, f, indent=2)

