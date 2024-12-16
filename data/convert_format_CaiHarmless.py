import json
import random

from datasets import load_dataset
from tqdm import tqdm

if __name__ == '__main__':
    random.seed(42)
    dataset_name = "CaiHarmless"

    if dataset_name == "CaiHarmless":
        dataset_name_or_path = "raw_data/HuggingFaceH4/cai-conversation-harmless"

        loaded_dataset = load_dataset(dataset_name_or_path)
        train_dataset = loaded_dataset["train"]
        test_dataset = loaded_dataset["test"]

        loaded_dataset["train"] = train_dataset
        loaded_dataset["test"] = test_dataset

    for split in ["train", "test"]:
        formatted_data = []
        split_set = loaded_dataset[split]

        for sample in tqdm(split_set, total=len(split_set)):
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            assert len(chosen)==2
            assert len(rejected)==2
            assert chosen[0]==rejected[0]
            assert chosen[0]["role"]=="user"
            assert rejected[0]["role"]=="user"
            assert chosen[1]["role"]=="assistant"
            assert rejected[1]["role"]=="assistant"

            user_query = chosen[0]["content"]
            chosen_response = chosen[1]["content"]
            rejected_response = rejected[1]["content"]

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
