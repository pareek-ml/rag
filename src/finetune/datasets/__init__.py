from functools import partial


from datasets.custom_dataset import get_custom_dataset, get_data_collator


DATASET_PREPROC = {
    "custom_dataset": get_custom_dataset,
}
DATALOADER_COLLATE_FUNC = {"custom_dataset": get_data_collator}
