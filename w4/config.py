from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

@dataclass
class BaseConfig:
    project_name: str = "accelerator"
    run_name: str = "train"

    output_dir: str = "results"
    checkpointing_steps: int = 300
    logging_steps: int = 20
    eval_steps: int = 50
    num_samples: int | None = 10000
    seed: int = 42

    learning_rate: float = 3e-4
    lr_scheduler_type: str = "linear"
    num_warm_up_steps: int = 0
    weight_decay: float = 0.01
    
    per_device_train_batch_size: int = 16
    gradient_accmulation_steps: int  = 1
    num_train_epochs: int = 1

    peft: str = "none"

@dataclass
class TestConfig(BaseConfig):
    run_name: str = "test"

    num_samples: int | None = 128
    logging_steps: int = 3

@dataclass
class LoRAConfig(BaseConfig):
    run_name: str = "use_lora"

    num_samples: int | None = 128
    logging_steps: int = 3
    peft: str = "LoRA"

cs = ConfigStore.instance()
cs.store(
    name="basic",
    node=BaseConfig
)
cs.store(
    name="test",
    node=TestConfig
)
cs.store(
    name="lora",
    node=LoRAConfig
)