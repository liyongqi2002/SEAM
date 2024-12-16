import json
import random

from datasets import load_dataset
from tqdm import tqdm

if __name__ == '__main__':
    random.seed(42)
    dataset_name = "HelpSteer"

    if dataset_name == "HelpSteer":
        dataset_name_or_path = "raw_data/nvidia/HelpSteer"

        loaded_dataset = load_dataset(dataset_name_or_path)
        train_dataset = loaded_dataset["train"]
        test_dataset = loaded_dataset["validation"]

        loaded_dataset["train"] = train_dataset
        loaded_dataset["test"] = test_dataset

    for split in ["train", "test"]:
        formatted_data = []
        split_set = loaded_dataset[split]

        prompt2responses_dict = {}
        prompt2scores_dict = {}

        for sample in tqdm(split_set, total=len(split_set)):
            prompt = sample["prompt"]
            response = sample["response"]
            helpfulness_score = sample["helpfulness"]

            if prompt not in prompt2responses_dict.keys():
                prompt2responses_dict[prompt] = [response]
                prompt2scores_dict[prompt] = [helpfulness_score]

            else:
                prompt2responses_dict[prompt].append(response)
                prompt2scores_dict[prompt].append(helpfulness_score)

        for prompt in prompt2responses_dict.keys():
            if len(prompt2responses_dict[prompt]) < 2:
                # 对于只有一个回复的，不考虑构造偏好对
                continue
            responses = prompt2responses_dict[prompt]
            scores = prompt2scores_dict[prompt]

            max_score_index = scores.index(max(scores))
            min_score_index = scores.index(min(scores))
            if max(scores) == min(scores):
                # 对于最大值最小值相同的，不做考虑
                continue

            user_query = prompt
            chosen_response = responses[max_score_index]
            rejected_response = responses[min_score_index]

            # 特殊处理，如果两个一样，该样本舍去
            if chosen_response == rejected_response:
                continue

            # For evaluation, we do not know which order we should take.
            # Thus, we need to first calculate the confidence of each order.
            # For example, if model calculates preference scores [0.7, 0.3] in order1, [0.2, 0.8] in order2.
            # Since max([0.7, 0.3])<max([0.2, 0.8]), we need to take the order2 as the final predict result ('pred').
            # Then, we should compare if pred=="given_response_2".
            formatted_instance = {
                "order1": {
                    "user_query": user_query,
                    "given_response_1": chosen_response,
                    "given_response_2": rejected_response,
                    "golden_preference": "given_response_1"
                },
                "order2": {
                    "user_query": user_query,
                    "given_response_1": rejected_response,
                    "given_response_2": chosen_response,
                    "golden_preference": "given_response_2"
                }
            }
            formatted_data.append(formatted_instance)
        with open(f'./{dataset_name}_{split}.json', mode='w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2)
