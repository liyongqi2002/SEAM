import argparse
import json
import os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model Name and Path
    parser.add_argument("--weak_model_name", default='Qwen2-1.5B-Instruct', type=str, help="Qwen2-1.5B-Instruct")
    parser.add_argument("--strong_model_name", default='Qwen2-7B', type=str, help="Qwen2-7B")

    parser.add_argument("--method", default='SFT or Others', type=str, help="Qwen2-7B")



    script_args = parser.parse_args()
    script_args.run_name = f"Method[{script_args.method}]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}]"


    # 1、加载方法权重
    with open('../config/path_config.json', encoding='utf-8', mode='r') as f:
        path_config = json.loads(f.read())
        path_config_reversed = {}
        for key in path_config.keys():
            for name, path in zip(path_config[key].keys(), path_config[key].values()):
                path_config_reversed[path] = name
    if not os.path.exists(script_args.strong_model_name):
        script_args.strong_model_name_or_path = path_config["llm"][script_args.strong_model_name]

    if "SFT" in script_args.method:
        script_args.lora_ckpt_dir=f"./sft_ckpt/alpaca52k/M[Qwen2-7B]"
    else:
        script_args.lora_ckpt_dir = f"./dpo_ckpt/{script_args.run_name}"

    with open("eval_data/eval_prompts.json", encoding="utf-8", mode="r") as f:
        eval_prompts = json.load(f)


    llm = LLM(model=script_args.strong_model_name_or_path, gpu_memory_utilization=0.95,
              enable_lora=True,max_num_batched_tokens=65528,max_model_len=65528
              )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["\n\nQuestion"],
    )

    # 2、构造prompt
    template="Question: {}\n\nAnswer: "
    prompts = [
         template.format(eval_prompt) for eval_prompt in eval_prompts
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("alpaca52k_sft_adapter", 1, script_args.lora_ckpt_dir)
    )

    # 3、解析并存储生成结果
    eval_responses=[]
    for eval_idx in range(len(eval_prompts)):
        res_text=outputs[eval_idx].outputs[0].text

        eval_response={
            "prompt":eval_prompts[eval_idx],
            "response":res_text
        }
        eval_responses.append(eval_response)
    # print(eval_responses)

    target_eval_responses_filepath=f"./eval_data/eval_responses-{script_args.run_name}.json"
    with open(target_eval_responses_filepath, encoding='utf-8', mode='w') as f:
            json.dump(eval_responses, f, indent=2)
