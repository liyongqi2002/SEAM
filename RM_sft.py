# Fine-Tune Llama2-7b on SE paired dataset
import argparse
import copy
import json
import multiprocessing
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)

from utils import set_seed

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


def get_args():
    # 参数设置
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--output_dir", default='./reward_model_ckpt', type=str, help="Random seed")
    parser.add_argument("--report_to", default="wandb", type=str, help="Random seed")

    # LoraConfig
    parser.add_argument("--lora_alpha", default=16, type=float, help="the lora alpha parameter")
    parser.add_argument("--lora_dropout", default=0.05, type=float, help="the lora dropout parameter")
    parser.add_argument("--lora_r", default=8, type=int, help="the lora r parameter")

    # Training Config
    parser.add_argument("--streaming", default=False, type=bool, help="whether to stream the dataset")
    parser.add_argument("--shuffle_buffer", default=5000, type=int, help="the shuffle buffer size")
    parser.add_argument("--seq_length", default=1024, type=int, help="the max seq length")
    parser.add_argument("--num_workers", default=4, type=int, help="the num_workers")
    parser.add_argument("--packing", default=True, type=bool, help="whether to use packing for SFTTrainer")
    parser.add_argument("--use_bnb", default=True, type=bool, help="the use_bnb")

    parser.add_argument("--eval_steps", default=500, type=int, help="the evaluation frequency")

    parser.add_argument("--shuffle_mode", default='both', type=str, help="random or both; random may introduce randomness")


    # Model Name and Path
    parser.add_argument("--weak_model_name", default='Qwen2-1.5B-Instruct', type=str, help="Qwen2-1.5B-Instruct")
    parser.add_argument("--strong_model_name", default='Qwen2-7B', type=str, help="Qwen2-7B")

    # Config about SFT data
    parser.add_argument("--dataset_name", default='AnthropicHH', type=str, help="UltraFeedback AnthropicHH")
    parser.add_argument("--data_mode", default='StrongCeiling', type=str, help="StrongCeiling WeakSelf Ours")
    parser.add_argument("--held_out_sample_num", default=5000, type=int, help="选取样本数目")

    parser.add_argument("--time", default='07181648', type=str, help="time in bash.sh")

    script_args = parser.parse_args()

    with open(f'prompts/basic_prompt_preference.txt', mode='r') as f:
        basic_prompt_preference = f.read()
    script_args.basic_prompt_preference = basic_prompt_preference

    # 1）修改模型路径
    with open('./config/path_config.json', encoding='utf-8', mode='r') as f:
        path_config = json.loads(f.read())
        path_config_reversed = {}
        for key in path_config.keys():
            for name, path in zip(path_config[key].keys(), path_config[key].values()):
                path_config_reversed[path] = name

    if not os.path.exists(script_args.strong_model_name):
        script_args.strong_model_name_or_path = path_config["llm"][script_args.strong_model_name]
    if not os.path.exists(script_args.weak_model_name):
        script_args.weak_model_name_or_path = path_config["llm"][script_args.weak_model_name]

    # 2）修改所需数据集路径以及数据集整理
    if script_args.data_mode in ["Ours", "OursFilter"]:
        script_args.dataset_name_or_path = f'prediction_results/HeldOutAnnotation-Method[{script_args.data_mode}]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}]-DATA[{script_args.dataset_name}].json'
    elif script_args.data_mode in ["WeakSelf", "Burns", "Ensemble", "UFilter", "StrongCeiling", "WSConsistency", "StrongSelf"]:
        script_args.dataset_name_or_path = f'prediction_results/baseline_prediction_results/HeldOutAnnotation-Method[{script_args.data_mode}]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}]-DATA[{script_args.dataset_name}].json'

    elif script_args.data_mode in ["RebuttalCotWithDefinition", "RebuttalCot", "RebuttalCotNoPrinciple", "RebuttalDebate"]:
        script_args.dataset_name_or_path = f"ablation_prediction_results/HeldOutAnnotation-AblationOurs[{script_args.data_mode}]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}]-DATA[{script_args.dataset_name}].json"

    print(f"dataset for training path: [{script_args.dataset_name_or_path}]")

    # 3）设置运行名称并修改输出路径
    script_args.run_name = f"TIME[{script_args.time}]-DataMode[{script_args.data_mode}]-SPLIT[HeldOut]-W[{script_args.weak_model_name}]-S[{script_args.strong_model_name}]-D[{script_args.dataset_name}]-HeldOutSIZE[{str(script_args.held_out_sample_num)}]"

    script_args.output_dir = f"reward_model_ckpt/{script_args.run_name}"

    return script_args


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        # print(example)
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    prompt = example["prompt"]
    completion = example["completion"]

    text = f"{prompt}{completion}"
    return text


def create_datasets(script_args, tokenizer, seed=None):
    with open(script_args.dataset_name_or_path, encoding="utf-8", mode="r") as f:
        data = json.load(f)

    reformatted_data = []
    for instance in data:
        ins_pred = instance["annotated_preference"]

        prompt_order1 = script_args.basic_prompt_preference.format(
            user_query=instance["user_query"],
            given_response_1=instance["given_response_1"],
            given_response_2=instance["given_response_2"],
        )

        prompt_order2 = script_args.basic_prompt_preference.format(
            user_query=instance["user_query"],
            given_response_1=instance["given_response_2"],
            given_response_2=instance["given_response_1"],
        )
        if ins_pred == "given_response_1":
            completion_order1 = "<Chosen>Response 1 from AI</Chosen>\n<Rejected>Response 2 from AI</Rejected>"
            completion_order2 = "<Chosen>Response 2 from AI</Chosen>\n<Rejected>Response 1 from AI</Rejected>"
        else:
            completion_order1 = "<Chosen>Response 2 from AI</Chosen>\n<Rejected>Response 1 from AI</Rejected>"
            completion_order2 = "<Chosen>Response 1 from AI</Chosen>\n<Rejected>Response 2 from AI</Rejected>"

        reformatted_instance_order1 = {
            "prompt": prompt_order1,
            "completion": completion_order1,
        }
        reformatted_instance_order2 = {
            "prompt": prompt_order2,
            "completion": completion_order2,
        }

        if script_args.shuffle_mode=="random":
            # Following: "RLHF Workflow: From Reward Modeling to Online RLHF"
            random_number = random.random()
            if random_number > 0.5:
                reformatted_data.append(reformatted_instance_order1)
            else:
                reformatted_data.append(reformatted_instance_order2)
        elif script_args.shuffle_mode=="both":
            reformatted_data.append(reformatted_instance_order1)
            reformatted_data.append(reformatted_instance_order2)


    dataset = Dataset.from_list(reformatted_data)
    dataset = dataset.train_test_split(test_size=0.01)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    print(train_data)

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


if __name__ == '__main__':

    script_args = get_args()
    set_seed(script_args.seed)

    wandb.init(
        # set the wandb project where this run will be logged
        project=f"[H-W2SG]-SFT-[5k-HeldOut]",
        name=script_args.run_name,
        # track hyperparameters and run metadata
        config=script_args
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        output_dir=script_args.output_dir,

        max_steps=1000,
        # num_train_epochs=1,  # 所有数据过一遍(但是不建议用这个，因为似乎有bug，训练会提前停止)
        logging_steps=5,

        # 保存策略：1）每 100 steps尝试保存与在验证集验证；2）最后保存一个验证集上loss最小的
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,

        save_strategy="steps",
        save_steps=script_args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,

        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        group_by_length=False,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.05,
        optim="paged_adamw_32bit",
        bf16=True,
        remove_unused_columns=False,
        report_to=script_args.report_to
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if training_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    if training_args.gradient_checkpointing:
        raise ValueError("gradient_checkpointing not supported")

    bnb_config = None
    if script_args.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.strong_model_name_or_path,  # W2S实验只有训练 strong的设置，所以这里仅考虑 strong_model_name_or_path
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(script_args.strong_model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_dataset, eval_dataset = create_datasets(script_args, tokenizer, seed=None)
    print(train_dataset)
    print(eval_dataset)

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=script_args.packing,
        max_seq_length=script_args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    train_dataloader = trainer.get_train_dataloader()
    print('train_dataloader', len(train_dataloader))

    trainer.train()
    trainer.save_model(training_args.output_dir)
