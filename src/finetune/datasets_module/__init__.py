from datasets_module.custom_dataset import get_custom_dataset as get_custom_dataset_loader
from datasets_module.custom_dataset import get_data_collator as get_custom_data_collator

from datasets_module.text_to_sql_dataset import get_custom_dataset as get_text_to_sql_dataset

DATASET_PREPROC = {
    "custom_dataset": get_text_to_sql_dataset,
    "text_to_sql_dataset": get_text_to_sql_dataset,
    "lamini/text_to_sql_finetune": get_text_to_sql_dataset,
}
DATALOADER_COLLATE_FUNC = {
    "custom_dataset": get_custom_data_collator,
}
