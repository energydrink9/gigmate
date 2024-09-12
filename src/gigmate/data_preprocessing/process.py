from zipfile import ZipFile
from clearml import Dataset
from pathlib import Path
from gigmate.utils.constants import get_clearml_dataset_name, get_clearml_dataset_version, get_clearml_project_name
import time

def measure_time(method_name: str, func):
    start_time = time.time()
    ret = func()
    end_time = time.time()
    print(f'{method_name} took {end_time - start_time} seconds')
    return ret

def extract_dataset(filename: str, out: str):
    with ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(out)

def get_files_in_directory(directory: str, extension: str = '.mid'):
    absolute_path = Path(directory).resolve()
    iterator = Path(absolute_path).glob(f'**/*{extension}')
    return [p for p in iterator]

def directory_has_files(directory: str, file_extension: str):
    return len(get_files_in_directory(directory, file_extension)) > 0

def upload_dataset(path: str, version: str, tags: list[str] = [], dataset = None):
    print(f'Creating dataset (set: {dataset}, tags: {tags})')
    dataset = Dataset.create(
        dataset_name=get_clearml_dataset_name(),
        dataset_project=get_clearml_project_name(), 
        dataset_version=version,
        dataset_tags=[f'{dataset}-set'] + tags
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