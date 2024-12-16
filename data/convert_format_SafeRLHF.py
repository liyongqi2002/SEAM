import json
import random

from datasets import load_dataset
from tqdm import tqdm

if __name__ == '__main__':
    random.seed(42)
    dataset_name = "SafeRLHF"

    if dataset_name == "SafeRLHF":
        dataset_name_or_path = "./raw_data/PKU-Alignment/PKU-SafeRLHF-30K"

        loaded_dataset = load_dataset(dataset_name_or_path)
        train_dataset = loaded_dataset["train"]
        test_dataset = loaded_dataset["test"]

        loaded_dataset["train"] = train_dataset
        loaded_dataset["test"] = test_dataset

    formatted_data_SHarmless={"train":[],"test":[]}
    formatted_data_SHelpful={"train":[],"test":[]}

    for split in ["train", "test"]:
        formatted_data = []
        split_set = loaded_dataset[split]

        for sample in tqdm(split_set, total=len(split_set)):
            user_query = sample["prompt"]

            is_response_0_safe = sample["is_response_0_safe"]
            is_response_1_safe = sample["is_response_1_safe"]

            better_response_id = sample["better_response_id"]
            safer_response_id = sample["safer_response_id"]
            better_response = f"response_{better_response_id}"
            safer_response = f"response_{safer_response_id}"

            if is_response_0_safe and is_response_1_safe:
                # 两个回复均安全，才考虑有用性
                if better_response == "response_0":
                    chosen_response = sample["response_0"]
                    rejected_response = sample["response_1"]
                else:
                    chosen_response = sample["response_1"]
                    rejected_response = sample["response_0"]
            else:
                # 否则，考虑安全性排序
                if safer_response == "response_0":
                    chosen_response = sample["response_0"]
                    rejected_response = sample["response_1"]
                else:
                    chosen_response = sample["response_1"]
                    rejected_response = sample["response_0"]


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

            if is_response_0_safe and is_response_1_safe:
                # 两个回复均安全，将样本纳入有用性子集
                formatted_data_SHelpful[split].append(formatted_instance)
            else:
                # 否则，说明至少有一个不安全
                formatted_data_SHarmless[split].append(formatted_instance)


        with open(f'./{dataset_name}_{split}.json', mode='w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2)


    formatted_data_SHarmless_train=formatted_data_SHarmless["train"]
    formatted_data_SHarmless_test=formatted_data_SHarmless["test"]

    formatted_data_SHelpful_train=formatted_data_SHelpful["train"]
    formatted_data_SHelpful_test=formatted_data_SHelpful["test"]

    with open(f'./SHarmless_train.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_SHarmless_train, f, indent=2)
    with open(f'./SHarmless_test.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_SHarmless_test, f, indent=2)
    with open(f'./SHelpful_train.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_SHelpful_train, f, indent=2)
    with open(f'./SHelpful_test.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_SHelpful_test, f, indent=2)