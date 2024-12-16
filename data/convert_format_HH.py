import json
import random

from datasets import load_dataset
from tqdm import tqdm

if __name__ == '__main__':
    random.seed(42)
    dataset_name = "AnthropicHH"

    if dataset_name == "AnthropicHH":
        dataset_name_or_path = "raw_data/PKU-Alignment/processed-hh-rlhf"
        harmless_loaded_dataset = load_dataset(f"{dataset_name_or_path}/harmless-base")
        helpful_loaded_dataset = load_dataset(f"{dataset_name_or_path}/helpful-base")

    all_formatted_data=[]
    for dataset in [harmless_loaded_dataset, helpful_loaded_dataset]:
        for split in ["train", "test"]:
            split_set = dataset[split]

            formatted_data = []
            for sample in tqdm(split_set, total=len(split_set)):
                user_query = sample["context"][-1]["text"]
                chosen_response = sample["chosen"]["text"]
                rejected_response = sample["rejected"]["text"]

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
            all_formatted_data.append(formatted_data)

    formatted_data_AHarmless_train=all_formatted_data[0]
    formatted_data_AHarmless_test=all_formatted_data[1]
    formatted_data_AHelpful_train=all_formatted_data[2]
    formatted_data_AHelpful_test=all_formatted_data[3]

    with open(f'./AHarmless_train.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_AHarmless_train, f, indent=2)
    with open(f'./AHarmless_test.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_AHarmless_test, f, indent=2)
    with open(f'./AHelpful_train.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_AHelpful_train, f, indent=2)
    with open(f'./AHelpful_test.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_AHelpful_test, f, indent=2)


    formatted_data_AnthropicHH_train=formatted_data_AHarmless_train + formatted_data_AHelpful_train
    formatted_data_AnthropicHH_test=formatted_data_AHarmless_test + formatted_data_AHelpful_test
    random.shuffle(formatted_data_AnthropicHH_train)
    random.shuffle(formatted_data_AnthropicHH_test)

    with open(f'./AnthropicHH_train.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_AnthropicHH_train, f, indent=2)
    with open(f'./AnthropicHH_test.json', mode='w', encoding='utf-8') as f:
        json.dump(formatted_data_AnthropicHH_test, f, indent=2)