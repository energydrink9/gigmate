from clearml import Dataset

from gigmate.data_preprocessing.constants import CLEARML_DATASET_TRAINING_NAME
from gigmate.utils.constants import get_clearml_project_name


def upload_dataset(path: str, version: str, tags: list[str] = [], dataset_set=None):
    print(f'Creating dataset (set: {dataset_set}, tags: {tags})')
    tags = [f'{dataset_set}-set'] + tags if dataset_set is not None else tags
    dataset = Dataset.create(
        dataset_project=get_clearml_project_name(), 
        dataset_name=CLEARML_DATASET_TRAINING_NAME,
        dataset_version=version,
        dataset_tags=tags,
    )
    print('Adding files')
    dataset.add_files(path=path)
    print('Uploading')
    dataset.upload(show_progress=True, preview=False)
    print('Finalizing')
    dataset.finalize()