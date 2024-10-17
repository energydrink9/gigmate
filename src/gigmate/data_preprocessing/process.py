from clearml import Dataset
from gigmate.utils.constants import get_clearml_dataset_name, get_clearml_dataset_training_name, get_clearml_dataset_version, get_clearml_project_name


def upload_dataset(path: str, version: str, tags: list[str] = [], dataset_set=None):
    print(f'Creating dataset (set: {dataset_set}, tags: {tags})')
    tags = [f'{dataset_set}-set'] + tags if dataset_set is not None else tags
    dataset = Dataset.create(
        dataset_name=get_clearml_dataset_training_name(),
        dataset_project=get_clearml_project_name(), 
        dataset_version=version,
        dataset_tags=tags
    )
    print('Adding files')
    dataset.add_files(path=path)
    print('Uploading')
    dataset.upload(show_progress=True, preview=False)
    print('Finalizing')
    dataset.finalize()


def get_remote_dataset(tags: list[str]):
    dataset = Dataset.get(
        dataset_project=get_clearml_project_name(),
        dataset_name=get_clearml_dataset_name(),
        dataset_version=get_clearml_dataset_version(),
        dataset_tags=tags,
        only_completed=False, 
        only_published=False, 
    )
    return dataset.get_local_copy()


def get_remote_dataset_by_id(id: str):
    dataset = Dataset.get(
        dataset_id=id,
        only_completed=False, 
        only_published=False, 
    )
    return dataset.get_local_copy()
