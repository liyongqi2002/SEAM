import os
import re

from tqdm import tqdm

import random
import numpy as np
from tqdm import trange, tqdm

import torch

from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import is_torch_available, is_torch_npu_available, is_torch_xpu_available, is_tf_available, \
    BitsAndBytesConfig

import numpy as np
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union

from accelerate import Accelerator
from tqdm import tqdm
from trl.trainer.utils import DPODataCollatorWithPadding

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import (
    LoraConfig,
    get_peft_model, PeftModel
)


def prepare_llm(llm_type, temperature=0, n=1):
    # the llm_type has been transformed into the path
    from vllm import LLM

    # 定义采样参数，temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = prepare_sampling_params(temperature=temperature, n=n)

    gpu_memory_utilization = 0.9
    llm = LLM(model=f"{llm_type}",gpu_memory_utilization=gpu_memory_utilization, )

    return llm, sampling_params


def prepare_sampling_params(temperature, n):
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=temperature,
                                     top_p=1,
                                     max_tokens=512,
                                     n=n,
                                     stop=["</Instance>", "<Instance>", "</Explanation>"],
                                     include_stop_str_in_output=True,
                                     logprobs=1
                                     )

    return sampling_params


# Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L531
def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


class DPOInference:
    def __init__(self, model, ref_model, tokenizer, accelerator, ref_free_norm="norm"):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.model.eval().requires_grad_(False)
        if ref_model is not None:
            self.ref_model.eval().requires_grad_(False)
            self.ref_free_norm = "none"
        else:
            if ref_free_norm not in ["norm", "avg", "sum"]:
                raise ValueError(f"Unknown ref_free_norm: {ref_free_norm}")
            self.ref_free_norm = ref_free_norm

        # for internals from TRL
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.label_pad_token_id = -100
        self.padding_value = tokenizer.pad_token_id
        self.truncation_mode = "keep_end"
        self.max_prompt_length = 128
        self.max_length = 2048

    def tokenize_row(self, feature) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["text_chosen"]  # modified from source
        rejected = feature["text_rejected"]  # modified from source

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337
            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # add BOS token to head of prompt
            prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
            chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
            rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                                                                                             self.label_pad_token_id
                                                                                         ] * len(
                chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                                                                                                 self.label_pad_token_id
                                                                                             ] * len(
                rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

            # print(batch)
            # print(batch.keys())

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

        return batch

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    # modified / new code for multiple DPO reward functions
    def inference_step(self, batch, ref_free: bool = False) -> list:
        """
        Uses TRL inference batched logprob computation to compute chosen + rejected
        logprobs then compute rewards and win rate.
        """
        with torch.no_grad():
            (
                policy_chosen_logps,
                policy_rejected_logps,
                _,  # policy_chosen_logits,
                _,  # policy_rejected_logits,
            ) = self.concatenated_forward(self.model, batch)

            # optionally compute reward without normalizing via reference model
            if not ref_free:
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    _,  # ref_chosen_logits,
                    _,  # ref_rejected_logits,
                ) = self.concatenated_forward(self.ref_model, batch)
                chosen_logratios = policy_chosen_logps.detach().cpu() - ref_chosen_logps.detach().cpu()
                rejected_logratios = policy_rejected_logps.detach().cpu() - ref_rejected_logps.detach().cpu()
            else:
                chosen_logratios = policy_chosen_logps.detach().cpu()
                rejected_logratios = policy_rejected_logps.detach().cpu()

        return chosen_logratios, rejected_logratios

    def concatenated_forward(
            self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        # with open('txt.txt', encoding='utf-8', mode='w') as f:
        #     for key, value in concatenated_batch.items():
        #         print(key, file=f)
        #         print(value.cpu().numpy(), file=f)
        #         print(key)
        #         print(value.cpu().numpy())
        # print(concatenated_batch,)

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits

        # set in init
        if self.ref_free_norm == "norm":
            average_log_prob = False
            norm_log_prob = True
        elif self.ref_free_norm == "avg":
            average_log_prob = True
            norm_log_prob = False
        elif self.ref_free_norm == "sum":
            average_log_prob = False
            norm_log_prob = False
        elif self.ref_free_norm == "none":
            average_log_prob = False
            norm_log_prob = False

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=average_log_prob,
            norm_log_prob=norm_log_prob,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    @staticmethod
    def get_batch_logps(
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            average_log_prob: bool = False,
            norm_log_prob: bool = False,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of
                label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token.
                Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            norm_log_prob: If True, return the normalized log probability per (non-masked) token.
                Note, only one of average_log_prob and norm_log_prob can be True.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities
                of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        elif norm_log_prob:
            return -torch.norm((per_token_logps * loss_mask), p=2, dim=-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    @staticmethod
    def concatenated_inputs(
            batch: Dict[str, Union[List, torch.LongTensor]],
            is_encoder_decoder: bool = False,
            label_pad_token_id: int = -100,
            padding_value: int = 0,
            device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids',
                which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch


def compute_AI_feedback(script_args, model_name_or_path, datasets):
    script_args.ref_free_type = "avg"
    tokenizer = script_args.tokenizer_builder(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. Qwen models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = script_args.model_builder(
            model_name_or_path,
            trust_remote_code=True,
            **script_args.model_kwargs,
    )

    first_order_dataset = datasets[0]
    reversed_order_dataset = datasets[1]

    dpo = DPOInference(
        model,
        ref_model=None,
        tokenizer=tokenizer,
        accelerator=Accelerator(),
        ref_free_norm='avg',
        # norm is norm, avg is average, sum is sum
    )

    BATCH_SIZE = 4

    def _compute_AI_feedback(dpo, dataset):
        # tokenize dataset
        column_names = list(dataset.features)
        tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=dpo.tokenizer.pad_token_id,
                label_pad_token_id=dpo.label_pad_token_id,
                is_encoder_decoder=dpo.is_encoder_decoder,
            ),
            # collate_fn = lambda x: x, # fix weird batching error
            shuffle=False,
            drop_last=False,
        )
        scores_chosen = []
        scores_rejected = []

        ref_free = True
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            rewards_chosen, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free)

            # for each item in batch, record 1 if chosen > rejected
            # extra score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards_chosen[0], dict):
                scores_chosen_batch = [result["score"] for result in rewards_chosen]
                scores_rejected_batch = [result["score"] for result in rewards_rejected]
            # for classes that directly output scores (custom code)
            else:
                # print('----------------')
                # print(rewards_chosen)
                # print(rewards_rejected)
                rewards = torch.stack([rewards_chosen, rewards_rejected])
                # print(rewards)
                rewards_softmax = torch.softmax(rewards, dim=0)
                # print(rewards_softmax)
                rewards_chosen = rewards_softmax[0]
                rewards_rejected = rewards_softmax[1]
                # print(rewards_chosen)
                # print(rewards_rejected)

                scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
                scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()

            scores_chosen += scores_chosen_batch
            scores_rejected += scores_rejected_batch

        return scores_chosen, scores_rejected

    first_order_chosen_scores, first_order_rejected_scores = _compute_AI_feedback(dpo=dpo,
                                                                                  dataset=first_order_dataset)
    reversed_order_chosen_scores, reversed_order_rejected_scores = _compute_AI_feedback(dpo=dpo,
                                                                                        dataset=reversed_order_dataset)

    return first_order_chosen_scores, first_order_rejected_scores, reversed_order_chosen_scores, reversed_order_rejected_scores


def compute_AI_feedback_single(script_args, model_name_or_path, dataset):
    script_args.ref_free_type = "avg"
    tokenizer = script_args.tokenizer_builder(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. Qwen models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    BATCH_SIZE = 4



    model = script_args.model_builder(
            model_name_or_path,
            trust_remote_code=True,
            **script_args.model_kwargs,
    )


    dpo = DPOInference(
        model,
        ref_model=None,
        tokenizer=tokenizer,
        accelerator=Accelerator(),
        ref_free_norm='avg',
        # norm is norm, avg is average, sum is sum
    )


    def _compute_AI_feedback(dpo, dataset):
        # tokenize dataset
        column_names = list(dataset.features)
        tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=dpo.tokenizer.pad_token_id,
                label_pad_token_id=dpo.label_pad_token_id,
                is_encoder_decoder=dpo.is_encoder_decoder,
            ),
            # collate_fn = lambda x: x, # fix weird batching error
            shuffle=False,
            drop_last=False,
        )
        scores_chosen = []
        scores_rejected = []

        ref_free = True
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            rewards_chosen, rewards_rejected = dpo.inference_step(batch, ref_free=ref_free)

            # for each item in batch, record 1 if chosen > rejected
            # extra score from dict within batched results (e.g. logits)
            # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
            if isinstance(rewards_chosen[0], dict):
                scores_chosen_batch = [result["score"] for result in rewards_chosen]
                scores_rejected_batch = [result["score"] for result in rewards_rejected]
            # for classes that directly output scores (custom code)
            else:
                rewards = torch.stack([rewards_chosen, rewards_rejected])
                rewards_softmax = torch.softmax(rewards, dim=0)
                rewards_chosen = rewards_softmax[0]
                rewards_rejected = rewards_softmax[1]

                scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
                scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()

            scores_chosen += scores_chosen_batch
            scores_rejected += scores_rejected_batch

        return scores_chosen, scores_rejected

    chosen_scores, rejected_scores = _compute_AI_feedback(dpo=dpo, dataset=dataset)

    return chosen_scores, rejected_scores


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
    if is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


def cal_metric(pred_instances):
    num_true = 0
    reward_score_num_true = 0

    for instance in pred_instances:

        pred_preference = instance["pred_preference"]
        reward_score_pred_preference = instance["reward_score_pred_preference"]

        golden_preference = instance["golden_preference"]

        if pred_preference == golden_preference:
            num_true += 1
        if reward_score_pred_preference == golden_preference:
            reward_score_num_true += 1
    acc = num_true / len(pred_instances)
    reward_score_acc = reward_score_num_true / len(pred_instances)

    metric = {"acc": acc, "reward_score_acc": reward_score_acc, "sample_num": len(pred_instances)}
    return metric


def parse_preference(generated_text_preference):
    try:
        chosen_text = re.search(r'<Chosen>(.*?)</Chosen>', generated_text_preference, re.DOTALL).group(1).strip()
        if "1" in chosen_text:
            pred_preference = "Response 1 from AI"
        else:
            pred_preference = "Response 2 from AI"

    except:
        pred_preference = "Response 1 from AI"

    if pred_preference == "Response 1 from AI":
        pred_preference = "given_response_1"
    else:
        pred_preference = "given_response_2"

    return pred_preference


def parse_explanation(generated_text_preference):
    try:
        pred_explanation = re.search(r'<Explanation>(.*?)</Explanation>', generated_text_preference, re.DOTALL).group(
            1).strip()
    except:
        if "<Explanation>" in generated_text_preference:
            pred_explanation = generated_text_preference.split('<Explanation>')[1]
            # pred_explanation = ""
        else:
            pred_explanation = ""
    return pred_explanation


def merge_lora_to_base_model(base_model_path, adapter_name_or_path, merged_save_path):
    # 大模型

    model_name_or_path = base_model_path
    # 保存路径
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={'': 'cpu'},
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(merged_save_path)
    model.save_pretrained(merged_save_path)
