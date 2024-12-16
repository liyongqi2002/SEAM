import json

if __name__ == '__main__':
    with open('../../data/SafeRLHF_test.json', encoding='utf-8', mode='r') as f:
        candidate_data=json.load(f)
    user_queries=[]
    for instance in candidate_data:
        user_queries.append(instance["order1"]["user_query"])

    print(len(user_queries))
    target_eval_prompts_filepath=f"./eval_prompts.json"
    with open(target_eval_prompts_filepath, encoding='utf-8', mode='w') as f:
            json.dump(user_queries, f, indent=2)
