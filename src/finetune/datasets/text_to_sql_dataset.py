import copy
import json
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

PROMPT_TEMPLATE = (
    "Given the following database schema and natural language question, write the SQL query that answers the question.\n\n"
    "### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:"
)


class TextToSQLDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.dataset = []
        raw_data = load_dataset(dataset_config.dataset)[
            partition
        ]  # Hugging Face dataset
        for item in raw_data:
            # Adjust keys if needed
            self.dataset.append({"input": item["input"], "output": item["output"]})
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100

        ann = self.dataset[index]
        schema, question = ann["input"].split(" -- -- ", 1)
        prompt = PROMPT_TEMPLATE.format(
            schema=schema.strip(), question=question.strip()
        )
        full_text = prompt + ann["output"]

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        full_ids.append(self.tokenizer.eos_token_id)

        input_ids = torch.tensor(full_ids, dtype=torch.int64)
        labels = copy.deepcopy(input_ids)

        # Mask labels for prompt
        labels[: len(prompt_ids)] = IGNORE_INDEX

        attention_mask = input_ids.ge(0)
        label_mask = labels.ge(0)

        input_ids[~attention_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": input_ids.tolist(),
            "labels": labels.tolist(),
            "attention_mask": attention_mask.tolist(),
        }


def get_custom_dataset(dataset_config, tokenizer, split):
    return TextToSQLDataset(dataset_config, tokenizer, split)
