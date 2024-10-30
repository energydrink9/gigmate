
from clearml import Dataset
from gigmate.data_preprocessing.constants import CLEARML_DATASET_TRAINING_NAME, CLEARML_DATASET_TRAINING_VERSION
from gigmate.utils.constants import get_clearml_project_name


def get_remote_dataset_by_id(id: str):
    dataset = Dataset.get(
        dataset_id=id,
        only_completed=True, 
        only_published=False, 
    )
    return dataset.get_local_copy()


def get_remote_dataset_by_tag(tag: str):
    dataset = Dataset.get(
        dataset_project=get_clearml_project_name(),
        dataset_name=CLEARML_DATASET_TRAINING_NAME,
        dateset_version=CLEARML_DATASET_TRAINING_VERSION,
        dataset_tags=[tag],
        only_completed=False,  # True 
        only_published=False, 
    )
    return dataset.get_local_copy()