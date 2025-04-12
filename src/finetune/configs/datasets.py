from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_cookbook/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = (
        "src/llama_cookbook/datasets/grammar_dataset/grammar_validation.csv"
    )


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_cookbook/datasets/alpaca_data.json"


@dataclass
class custom_dataset:
    dataset: str = "lamini/text_to_sql_finetune"
    file: str = "datasets/text_to_sql_dataset.py:get_custom_dataset"
    train_split: str = "train"
    test_split: str = "train"


@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"
