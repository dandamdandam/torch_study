import random
from accelerate.logging import get_logger

from datasets import load_dataset
import torch
from transformers.data.data_collator import default_data_collator
from torch.utils.data import DataLoader

logger = get_logger(__name__)

class Preprocessor:
    question_column_name = "prompt"
    context_column_name = "maze"
    answer_column_name = "solution"

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def build_prompt(context: str, question: str) -> str:
        return f"{question}\n\nASCII maze:\n{context}\n\nAnswer:\n"

    def prepare_train_features(self, examples):
        questions = [q.lstrip() for q in examples[self.question_column_name]]
        contexts = examples[self.context_column_name]
        answers = examples[self.answer_column_name] 

        prompts = [self.build_prompt(c, q) for c, q in zip(contexts, questions)]
        full_texts = [p + a + self.tokenizer.eos_token for p, a in zip(prompts, answers)]

        tokenized_full = self.tokenizer(
            full_texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        # label 처리
        # prompt와 padding은 -100로 처리해서 학습에 제외시킴
        labels = []
        for p, full_ids in zip(prompts, tokenized_full["input_ids"]):
            p_ids = self.tokenizer(
                p,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )["input_ids"]

            lab = full_ids.copy()
            prompt_len = min(len(p_ids), len(lab))
            for i in range(prompt_len):
                lab[i] = -100

            pad_id = self.tokenizer.pad_token_id
            for i in range(len(lab)):
                if lab[i] == pad_id:
                    lab[i] = -100

            labels.append(lab)

        tokenized_full["labels"] = labels
        return tokenized_full

    def prepare_test_features(self, examples):
        questions = [q.lstrip() for q in examples[self.question_column_name]]
        contexts = examples[self.context_column_name]
        prompts = [self.build_prompt(c, q) for c, q in zip(contexts, questions)]

        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            # padding_side="left"
        )
        tokenized["maze"] = examples[self.context_column_name]
        tokenized["gold"] = examples[self.answer_column_name]
        return tokenized

    def get_dataset(self, purpose, accelerator, per_device_train_batch_size, seed, num_samples = None):
        dataset = load_dataset("selimaktas/maze-curriculum-dataset", split="train").train_test_split(test_size=0.1)
        column_names = dataset["train"].column_names
        if purpose == "train":
            dataset = dataset["train"].shuffle(seed)

            if num_samples:
                dataset = dataset.select(range(num_samples))
            dataset = dataset.train_test_split(test_size=0.1)
            examples, eval_examples = dataset["train"], dataset["test"]

            with accelerator.main_process_first():
                train_dataset = examples.map(
                    self.prepare_train_features,
                    batched=True,
                    remove_columns=column_names,
                )
                eval_dataset = eval_examples.map(
                    self.prepare_train_features,
                    batched=True,
                    remove_columns=column_names,
                )

            index = random.sample(range(len(train_dataset)), 1)[0]
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            index = random.sample(range(len(eval_dataset)), 1)[0]
            logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")

            train_dataloader = DataLoader(train_dataset, batch_size=per_device_train_batch_size, shuffle=False, collate_fn=default_data_collator, drop_last=True) # type: ignore
            eval_dataloader = DataLoader(eval_dataset, batch_size=per_device_train_batch_size, shuffle=False, collate_fn=default_data_collator, drop_last=True) # type: ignore

            return train_dataloader, eval_dataloader
        else:
            dataset = dataset["test"]
            if num_samples:
                dataset = dataset.select(range(num_samples))
            with accelerator.main_process_first():
                test_dataset = dataset.map(
                    self.prepare_train_features,
                    batched=True,
                    remove_columns=column_names,
                )
            index = random.sample(range(len(test_dataset)), 1)[0]
            logger.info(f"Sample {index} of the eval set: {test_dataset[index]}.")
            test_dataloader = DataLoader(test_dataset, batch_size=per_device_train_batch_size, shuffle=False, collate_fn=default_data_collator, drop_last=True) # type: ignore

            return test_dataloader

def normalize_answer(s: str) -> str:
    # squad 스타일 정규화 (아주 간단 버전)
    return " ".join(s.strip().lower().split())


@torch.no_grad()
def generate_answers(accelerator, model, tokenizer, dataloader, max_new_tokens=32):
    model.eval()
    preds = []
    golds = []
    mazes = []

    for batch in dataloader:
        gen_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
        )

        prompt_len = batch["input_ids"].shape[1]
        answer_ids = gen_ids[:, prompt_len:]

        gathered_ids = accelerator.gather_for_metrics(answer_ids)
        gathered_golds = accelerator.gather(batch["gold"])
        gathered_mazes = accelerator.gather(batch["maze"])
        decoded_preds = tokenizer.batch_decode(gathered_ids, skip_special_tokens=True)
        
        preds.extend([p.strip() for p in decoded_preds])
        golds.extend(gathered_golds)
        mazes.extend(gathered_mazes)

    return preds, golds, mazes


def compute_exact_match(preds, golds):
    em = 0
    for p, g in zip(preds, golds):
        em += int(normalize_answer(p) == normalize_answer(g))
    em /= max(1, len(preds))
    return em
