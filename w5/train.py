import logging
import os
import sys
sys.path.append(os.getcwd())

import math
from dotenv import load_dotenv

import hydra
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.data.data_collator import default_data_collator
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from datasets import Dataset, load_dataset

import wandb

from config import BaseConfig
from utils import Preprocessor2

logger = logging.getLogger(__name__)

# CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29500 w5/train

class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # training perplexity
        if "loss" in logs:
            try:
                ppl = math.exp(logs["loss"])
                logs["perplexity"] = ppl
            except OverflowError:
                logs["perplexity"] = float("inf")

        # evaluation perplexity
        if "eval_loss" in logs:
            try:
                eval_ppl = math.exp(logs["eval_loss"])
                logs["eval_perplexity"] = eval_ppl
            except OverflowError:
                logs["eval_perplexity"] = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return
        try:
            metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["eval_perplexity"] = float("inf")

@hydra.main(version_base=None, config_name="basic")
def main(config: BaseConfig):
    load_dotenv()

    wandb.init(
        project=config.project_name,
        name=config.run_name,
        config={
            **dict(config), # type: ignore
            "exec_file": __file__,
        },
    ) if os.environ.get("WANDB_API_KEY") else wandb.init(mode="disabled")

    # Model and Tokenizer setup
    model_id = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="kernels-community/vllm-flash-attn3"
    )
    # FSDP에서는 얘를 활성화 해야함. TODO 왜지??
    model.config.use_cache = False

    # LoRA setup matching acc.py
    if config.peft == "LoRA":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules=[
                "embed_tokens",
                "q_proj",
                "v_proj",
                "o_proj",
                "k_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ]
        )
        model = get_peft_model(model, peft_config)
        logger.info(f"Adopted LoRA - {model.print_trainable_parameters()}")

    preprocessor = Preprocessor2(tokenizer, max_length=512)

    full_dataset: Dataset = load_dataset("selimaktas/maze-curriculum-dataset", split="train")  # type: ignore
    dataset = full_dataset.train_test_split(test_size=0.1)
    column_names = dataset["train"].column_names
    dataset = dataset["train"].shuffle(config.seed)
    if config.num_samples:
        dataset = dataset.select(range(config.num_samples))
    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = dataset["train"].map(
        preprocessor.prepare_train_features,
        batched=True,
        remove_columns=column_names,
        desc="Processing train dataset",
    )
    eval_dataset = dataset["test"].map(
        preprocessor.prepare_train_features,
        batched=True,
        remove_columns=column_names,
        desc="Processing eval dataset",
    )

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        run_name=config.run_name,
        learning_rate=config.learning_rate,
        completion_only_loss=True,
        lr_scheduler_type=config.lr_scheduler_type,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accmulation_steps, # using the typo name from BaseConfig
        weight_decay=config.weight_decay,
        warmup_steps=config.num_warm_up_steps,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.checkpointing_steps,
        seed=config.seed,
        bf16=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        max_length=512,
        remove_unused_columns=False, # Preprocessor already handled column removal
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model, # type: ignore
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[PerplexityCallback()],
    )

    logger.info("***** Running Training with SFTTrainer *****")
    
    # Check the first batch - TODO completion_only_loss가 안되어 있는 것 같음. 어느 시점에서 되는지 확인하고 로깅 다시 해야함
    logger.info("***** Checking first batch*****")
    first_batch = next(iter(trainer.get_train_dataloader()))
    logger.info(f"First data input_ids: {first_batch['input_ids'][0]}")
    logger.info(f"First data attention_mask: {first_batch['attention_mask'][0]}")
    logger.info(f"First data labels: {first_batch['labels'][0]}")
    
    trainer.train()

    wandb.finish()

    # Save model and tokenizer
    logger.info("Saving Model...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()
