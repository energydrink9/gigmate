import os
from typing import List
from clearml import PipelineDecorator
from gigmate.data_preprocessing.constants import CLEARML_DATASET_TRAINING_VERSION, DATASET_TAGS
from gigmate.data_preprocessing.dataset import get_remote_dataset_by_tag
from gigmate.data_preprocessing.steps.augment import augment_all
from gigmate.data_preprocessing.steps.convert_to_ogg import convert_to_ogg
from gigmate.data_preprocessing.steps.encode import encode_all
from gigmate.data_preprocessing.steps.merge import assort_and_merge_all
from gigmate.data_preprocessing.steps.split import split_all
from gigmate.data_preprocessing.steps.uncompress import uncompress_files
from gigmate.data_preprocessing.steps.upload import upload
from gigmate.data_preprocessing.utils import upload_dataset
from gigmate.utils.constants import get_clearml_project_name
from gigmate.data_preprocessing.steps.distort import distort_all

BASE_DIR = '../dataset'
ORIGINAL_FILES_DIR = os.path.join(BASE_DIR, 'original')
MERGED_FILES_DIR = os.path.join(BASE_DIR, 'merged')
AUGMENTED_FILES_DIR = os.path.join(BASE_DIR, 'augmented')
DISTORTED_FILES_DIR = os.path.join(BASE_DIR, 'distorted')
ENCODED_FILES_DIR = os.path.join(BASE_DIR, 'encoded')
SPLIT_FILES_DIR = os.path.join(BASE_DIR, 'split')


@PipelineDecorator.component(return_values=['uncompressed_dir'], cache=False)
def uncompress_step(source_dir: str):
    output_dir = uncompress_files(source_dir)
    return output_dir


@PipelineDecorator.component(return_values=['converted_to_ogg_dir'], cache=False)
def convert_to_ogg_step(source_dir):
    output_dir = convert_to_ogg(source_dir)
    return output_dir


@PipelineDecorator.component(return_values=['merged_dir'], cache=False)
def assort_and_merge_step(merged_dir, stem_name, tags: List[str]):
    print('Creating assortment and merging')
    source_dir = get_remote_dataset_by_tag('original')
    output_dir = assort_and_merge_all(source_dir, merged_dir, stem_name)
    upload_dataset(path=output_dir, version=CLEARML_DATASET_TRAINING_VERSION, tags=tags + ['merged'], dataset_set=set)
    return output_dir


@PipelineDecorator.component(return_values=['augmented_dir'], cache=False)
def augment_step(output_dir: str, tags: List[str]) -> str:
    print('Augmenting dataset')
    source_dir = get_remote_dataset_by_tag('merged')
    output_dir = augment_all(source_dir, output_dir)
    upload_dataset(path=output_dir, version=CLEARML_DATASET_TRAINING_VERSION, tags=tags + ['augmented'], dataset_set=set)
    return output_dir


@PipelineDecorator.component(return_values=['distorted_dir'], cache=False)
def distort_step(output_dir: str, tags: List[str]) -> str:
    print('Distorting dataset')
    source_dir = get_remote_dataset_by_tag('augmented')
    output_dir = distort_all(source_dir, output_dir)
    upload_dataset(path=output_dir, version=CLEARML_DATASET_TRAINING_VERSION, tags=tags + ['distorted'], dataset_set=set)
    return output_dir


@PipelineDecorator.component(return_values=['encoded_dir'], cache=False)
def encode_step(output_dir: str, tags: List[str]) -> str:
    print('Encoding dataset')
    source_dir = get_remote_dataset_by_tag('distorted')
    output_dir = encode_all(source_dir, output_dir)
    upload_dataset(path=output_dir, version=CLEARML_DATASET_TRAINING_VERSION, tags=tags + ['encoded'], dataset_set=set)
    return output_dir


@PipelineDecorator.component(return_values=['split_dir'], cache=False)
def split_step(output_dir: str, tags: List[str]) -> List[str]:

    print('Splitting dataset')
    source_dir = get_remote_dataset_by_tag('encoded')
    split_dirs = split_all(source_dir, output_dir)

    return split_dirs


@PipelineDecorator.component(return_values=[], cache=False)
def upload_step(input_dirs: List[str], tags: List[str]):

    upload(input_dirs, tags)


@PipelineDecorator.pipeline(
    name='Dataset preparation pipeline',
    project=get_clearml_project_name(),
    version=CLEARML_DATASET_TRAINING_VERSION,
)
def dataset_preparation_pipeline(source_dir: str):

    print(f'Preparing dataset. Source dir: {source_dir}')

    # uncompressed_dir = uncompress_step(source_dir)
    # converted_to_ogg_dir = convert_to_ogg_step(uncompressed_dir)
    tags = DATASET_TAGS + ['original']

    print('Uploading prepared dataset')
    upload_dataset(path=source_dir, version=CLEARML_DATASET_TRAINING_VERSION, tags=tags, dataset_set=None)


@PipelineDecorator.pipeline(
    name='Dataset creation pipeline',
    project=get_clearml_project_name(),
    version=CLEARML_DATASET_TRAINING_VERSION,
)
def dataset_creation_pipeline(stem_name: str, merged_dir: str, augmented_dir: str, distorted_dir: str, encoded_dir: str, split_dir: str):
    
    tags = DATASET_TAGS + [f'stem-{stem_name}']

    assort_and_merge_step(merged_dir, stem_name, tags)
    augment_step(augmented_dir, tags)
    distort_step(distorted_dir, tags)
    encode_step(encoded_dir, tags)
    split_dirs = split_step(split_dir, tags)
    upload_step(split_dirs, tags)
