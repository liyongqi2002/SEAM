# 0. imports
import json
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from trl import DPOTrainer,DPOConfig


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    base_model_name_or_path: Optional[str] = field(
        default="/data01/qty/liyongqi/llm_path/meta-llama/llama2_7b",
        metadata={"help": "the location of the SFT model name or path"},
    )

    base_adapter_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the SFT adapter model name or path"},
    )

    dataset_name_or_path: Optional[str] = field(
        default='',
        metadata={"help": "the location of the dataset name or path"},
    )

    method: Optional[str] = field(
        default='Ours',
        metadata={"help": '"Ours","OursFilter","weak","StrongSelf","Ensemble","Burns","UFilter","WSConsistency","StrongCeiling"'},
    )

    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the logging frequency"})
    # save_steps: Optional[int] = field(default=200, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=200, metadata={"help": "the evaluation frequency"})

    log_freq: Optional[int] = field(default=5, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
                    '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
                    'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
                    "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    # 由于只有一组弱强所以直接定义
    script_args.weak_model_name="Qwen2-1.5B-Instruct"
    script_args.strong_model_name="Qwen2-7B"

    # 1、定义输出路径
    script_args.run_name = f"Method[{script_args.method}]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}]"
    script_args.output_dir = f"./dpo_ckpt/{script_args.run_name}"

    # 2、定义SFT模型的加载
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    with open('../config/path_config.json', encoding='utf-8', mode='r') as f:
        path_config = json.loads(f.read())
        path_config_reversed = {}
        for key in path_config.keys():
            for name, path in zip(path_config[key].keys(), path_config[key].values()):
                path_config_reversed[path] = name
    if not os.path.exists(script_args.strong_model_name):
        script_args.strong_model_name_or_path = path_config["llm"][script_args.strong_model_name]

    script_args.base_model_name=script_args.strong_model_name
    script_args.base_model_name_or_path=script_args.strong_model_name_or_path
    script_args.base_adapter_model_name_or_path=f"sft_ckpt/alpaca52k/M[{script_args.base_model_name}]"

    model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            load_in_4bit=script_args.load_in_4bit,
            device_map={"": Accelerator().local_process_index},
            trust_remote_code=True
        )
    model = PeftModel.from_pretrained(model, script_args.base_adapter_model_name_or_path)



    model.config.use_cache = False
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.base_adapter_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # 3、加载dpo数据集
    script_args.dataset_name_or_path=f"./dpo_data/DPOInstances-Method[{script_args.method}]-WEAK[{script_args.weak_model_name}]-STRONG[{script_args.strong_model_name}].json"
    with open(script_args.dataset_name_or_path, encoding="utf-8", mode="r") as f:
        reformatted_data = json.load(f)

    dataset = Dataset.from_list(reformatted_data)
    dataset = dataset.train_test_split(test_size=0.01)


    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(eval_dataset)}")

    wandb.init(
        project=f"[H-W2SG]-DPO",
        config=script_args,
        name=script_args.run_name,
    )


    # 4、开始训练
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs=1,  # 所有数据过一遍
        logging_steps=script_args.logging_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,


        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,

        save_strategy="steps",
        save_steps=script_args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

