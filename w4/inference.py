from accelerate.logging import get_logger

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from utils import Preprocessor, generate_answers, compute_exact_match

logger = get_logger(__name__)

if __name__ == "__main__":
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True)

    preprocessor = Preprocessor(tokenizer, max_length=512)

    test_dataloader = preprocessor.get_dataset(
        "test",
        accelerator,
        per_device_train_batch_size=24,
        seed=42
    )
    
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    preds, golds = generate_answers(accelerator, model, tokenizer, test_dataloader)

    if accelerator.is_main_process:
        test_accuracy = compute_exact_match(preds, golds)
        logger.info(f"Final Test Accuracy: {test_accuracy:.4f}")

        logger.info(f"Sample {0}:")
        logger.info(f"  Predict: {preds[0]}")
        logger.info(f"  Ground Truth: {golds[0]}")
