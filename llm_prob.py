# encoding=utf-8

import torch
from tqdm import tqdm

def batch_compute_prob(script_args, tokenizer, model, all_principles_list,prompts):
    """Compute the probability of LLM generates a principle."""
    all_principles_vocab_id=[]
    for principles_list in all_principles_list:
        principles_vocab_id = []
        for principle in principles_list:
            vocab_id = tokenizer.encode(principle)[0]
            principles_vocab_id.append(vocab_id)
        all_principles_vocab_id.append(principles_vocab_id)

    def batchify_prompts(prompts, batch_size):
        all_batched_prompts=[]
        for i in range(0, len(prompts), batch_size):
            all_batched_prompts.append(prompts[i:i + batch_size])
        return all_batched_prompts

    def get_last_token_probs(prompts, batch_size):
        all_probs_list=[]
        all_batched_prompts=batchify_prompts(prompts, batch_size)
        for i, batched_prompts in tqdm(enumerate(all_batched_prompts),total=len(all_batched_prompts),desc="Calculating probs!"):
            inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(script_args.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            attention_mask = inputs['attention_mask']

            batch_size, seq_length, vocab_size = logits.size()
            probs_list = []

            for j in range(batch_size):
                # 找到序列中的真实长度（非填充部分）
                real_length = attention_mask[j].sum().item()
                last_token_logits = logits[j, real_length - 1, :]
                probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
                probs_list.append(probs)

            all_probs_list.extend(probs_list)

            # for k, probs in enumerate(probs_list):
            #     print(f"Batch {i}, Prompt {k}: {batched_prompts[k]}")
            #     print(f"Probability Distribution: {probs}\n")
            #
            #     next_token_id = torch.argmax(probs).item()
            #     next_token = tokenizer.decode(next_token_id)
            #     print(f"Next token ID: {next_token_id}")
            #     print(f"Next token: {next_token}")
            #     print(f"Probability: {probs[next_token_id].item()}\n")
        return all_probs_list
    # 将 prompts 分批次处理，每批次大小为4
    batch_size = 4
    all_probs_list = get_last_token_probs(prompts, batch_size)

    predefined_principle=script_args.dict_principles.keys()
    dict_principle2id={}
    for principle in predefined_principle:
        vocab_id = tokenizer.encode(principle)[0]
        dict_principle2id[principle]=vocab_id


    all_principles_score=[]
    for ins_idx in range(len(prompts)):
        principles_vocab_id=[dict_principle2id[principle] for principle in all_principles_list[ins_idx]]

        probs_list=all_probs_list[ins_idx]
        principles_score = probs_list[principles_vocab_id]

        # 对原则列表进行再一次的归一化
        # principles_score=principles_score.softmax(dim=-1)
        all_principles_score.append(principles_score)

    return all_principles_score

