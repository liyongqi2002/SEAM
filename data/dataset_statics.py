import json

if __name__ == '__main__':
    dataset_names=[
        "AHelpful",
        "SHelpful",
        "HelpSteer",
        "AHarmless",
        "SHarmless",
        "CaiHarmless",
        "AnthropicHH",
        "SafeRLHF",
    ]

    for dataset_name in dataset_names:
        for split in ['train', 'test']:
            filepath=f"{dataset_name}_{split}.json"
            # print(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)
            print(f"数据集 {dataset_name} {split} 数目 {len(data)}")
            print("=======================")