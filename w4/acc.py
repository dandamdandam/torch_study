from datetime import timedelta
import os
import math
from dotenv import load_dotenv
import hydra
import torch

from peft import LoraConfig, TaskType, get_peft_model

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from tqdm import tqdm

from transformers.trainer_pt_utils import get_parameter_names
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_scheduler

from config import BaseConfig
from utils import Preprocessor

logger = get_logger(__name__, log_level="INFO")

# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --config_file acc_config.yaml acc.py

def evaluate(model, eval_dataloader, accelerator: Accelerator):
    model.eval()
    total_loss = 0
    total_eval_steps = 0

    with torch.inference_mode():
        for batch in tqdm(
            eval_dataloader, desc="Evaluating", leave=False, disable=not accelerator.is_main_process, mininterval=1
        ):
            eval_outputs = model(**batch)
            eval_loss = eval_outputs.loss
            total_loss += eval_loss.item()
            total_eval_steps += 1

    avg_eval_loss = total_loss / total_eval_steps
    ppl = math.exp(avg_eval_loss)
    return avg_eval_loss, ppl

@hydra.main(version_base=None, config_name="basic")
def main(config: BaseConfig):
    load_dotenv()

    accelerator = Accelerator(
        log_with="wandb",
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(minutes=120))
        ]
    )

    logger.info(accelerator.state)
    if os.environ.get("WANDB_API_KEY"):
        accelerator.init_trackers(
            project_name=config.project_name, config=dict(config), init_kwargs={"wandb": {"name": config.run_name}} # type: ignore
        )
        logger.info(f"WandB initialized with project: {config.project_name}, run: {config.run_name}")
    else:
        logger.warning("WANDB_API_KEY not found or not configured in .env file. Disabling WandB logging.")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    if config.peft == "LoRA":
        """
        r: low-rank matrices의 dimension
        lora_alpha: low-rank matrices의 scaling factor
        lora_dropout: LoRA layer의 droupout 확률
        """
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

    preprocessor = Preprocessor(tokenizer, max_length=512)
    train_dataloader, eval_dataloader = preprocessor.get_dataset("train", accelerator, config.per_device_train_batch_size, config.seed, config.num_samples) # type: ignore

    forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    decay_params = set(get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns))
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_params],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_params],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accmulation_steps
    )
    max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warm_up_steps,
        num_training_steps=max_train_steps,
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)

    model.config.use_cache = False 

    logger.info("***** Running Training *****")
    model.train()
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(config.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)  
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)

            # Logging
            if step % config.logging_steps == 0 and accelerator.is_main_process:
                lr = lr_scheduler.get_last_lr()[0]
                loss = loss.item()
                ppl = math.exp(loss)
                progress_bar.write(f"Epoch {epoch}, Step {step}/{max_train_steps} Loss: {loss:.4f}, PPL: {ppl:.4f}, LR: {lr:.2e}")
                accelerator.log({"train/loss": loss, "train/epoch": epoch, "train/learning_rate": lr, "train/ppl": ppl}, step=step)

            # eval
            if step > 0 and step % config.eval_steps == 0:
                eval_loss, eval_ppl = evaluate(model, eval_dataloader, accelerator)
                model.train()
                
                if accelerator.is_main_process:
                    progress_bar.write(f"eval loss : {eval_loss}, eval ppl : {eval_ppl}")
                    accelerator.log({ "eval/loss": eval_loss, "eval/ppl": eval_ppl }, step=step)

            if isinstance(config.checkpointing_steps, int):
                if step % config.checkpointing_steps == 0 and accelerator.sync_gradients:
                    # TODO checkpoint 구현
                    pass
                    # out = os.path.join(config.output_dir, f"step_{completed_steps}")
                    # accelerator.save_state(out)

        eval_loss, eval_ppl = evaluate(model, eval_dataloader, accelerator)
        logger.info(f"Final Evaluation Loss: {eval_loss:.4f}, Final Evaluation PPL: {eval_ppl:.4f}")
        accelerator.log({ "eval/loss": eval_loss, "eval/ppl": eval_ppl }, step=step)

    # save 
    logger.info("Train End, Waiting Assumble...")
    accelerator.wait_for_everyone()
    logger.info("Saving Model...")
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save_model(unwrapped_model, config.output_dir)
        # unwrapped_model.save_pretrained(
        #     config.output_dir,
        #     is_main_process=accelerator.is_main_process,
        #     save_function=accelerator.save,
        #     state_dict=accelerator.get_state_dict(model)
        # )
        tokenizer.save_pretrained(config.output_dir)
    
if __name__ == "__main__":
    main()
