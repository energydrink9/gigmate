import os
from typing import List
from clearml import PipelineDecorator

from gigmate.data_preprocessing.process import get_remote_dataset_by_id, upload_dataset
from gigmate.data_preprocessing.steps.augment import augment_all
from gigmate.data_preprocessing.steps.convert_to_ogg import convert_to_ogg
from gigmate.data_preprocessing.steps.encode import encode_all
from gigmate.data_preprocessing.steps.merge import assort_and_merge_all
from gigmate.data_preprocessing.steps.split import split_all
from gigmate.data_preprocessing.steps.uncompress import uncompress_files
from gigmate.utils.constants import get_clearml_dataset_version, get_clearml_project_name
from gigmate.data_preprocessing.steps.distort import distort_all

BASE_DIR = '/dataset'
SOURCE_DIR = os.path.join(BASE_DIR, 'original')
MERGED_FILES_DIR = os.path.join(BASE_DIR, 'merged')
AUGMENTED_FILES_DIR = os.path.join(BASE_DIR, 'augmented')
DISTORTED_FILES_DIR = get_remote_dataset_by_id('a19b2949bf984f029d2deb2c892defea')
ENCODED_FILES_DIR = os.path.join(BASE_DIR, 'encoded')
SPLIT_FILES_DIR = os.path.join(BASE_DIR, 'split')

STEM_NAME = 'guitar'
RANDOM_ASSORTMENTS_PER_SONG = 1
DATASET_TAGS = ['medium', 'soundstripe']


@PipelineDecorator.component(return_values=['uncompressed_dir'], cache=False)
def uncompress_step(source_dir: str):
    output_dir = uncompress_files(source_dir)
    return output_dir


@PipelineDecorator.component(return_values=['converted_to_ogg_dir'], cache=False)
def convert_to_ogg_step(split_output):
    output_dir = convert_to_ogg(split_output)
    return output_dir


@PipelineDecorator.component(return_values=['merged_dir'], cache=False)
def assort_and_merge_step(converted_to_ogg_dir, merged_dir, stem_name, random_assortments_per_song):
    print(f'Creating assortment and merging {converted_to_ogg_dir}')
    output_dir = assort_and_merge_all(converted_to_ogg_dir, merged_dir, stem_name, random_assortments_per_song)
    return output_dir


@PipelineDecorator.component(return_values=['augmented_dir'], cache=False)
def augment_step(source_dir: str, output_dir: str) -> str:
    print('Augmenting dataset')
    return augment_all(source_dir, output_dir)


@PipelineDecorator.component(return_values=['distorted_dir'], cache=False)
def distort_step(source_dir: str, output_dir: str) -> str:
    print('Distorting dataset')
    return distort_all(source_dir, output_dir)


@PipelineDecorator.component(return_values=['encoded_dir'], cache=False)
def encode_step(source_dir: str, output_dir: str) -> str:
    print('Encoding dataset')
    return encode_all(source_dir, output_dir)


@PipelineDecorator.component(return_values=['split_dir'], cache=False)
def split_step(source_dir: str, output_dir: str) -> List[str]:
    print('Splitting dataset')
    return split_all(source_dir, output_dir)


@PipelineDecorator.pipeline(
    name='Dataset creation pipeline',
    project=get_clearml_project_name(),
    version='1.0'
)
def dataset_creation_pipeline(stem_name: str, random_assortments_per_song: int, source_dir: str, merged_dir: str, augmented_dir: str, distorted_dir: str, encoded_dir: str, split_dir: str):
    # uncompressed_dir = uncompress_step(source_dir)
    # converted_to_ogg_dir = convert_to_ogg_step(uncompressed_dir)
    # merged_dir = assort_and_merge_step(converted_to_ogg_dir, merged_dir, stem_name, random_assortments_per_song)
    # augmented_dir = augment_step(merged_dir, augmented_dir)
    # distorted_dir = distort_step(augmented_dir, distorted_dir)
    encoded_dir = encode_step(distorted_dir, encoded_dir)
    split_dirs = split_step(encoded_dir, split_dir)
    tags = DATASET_TAGS + [f'stem-{stem_name}']

    for split_dir in split_dirs:
        set = os.path.split(split_dir)[1]
        print(f'Uploading {set} dataset')
        upload_dataset(path=split_dir, version=get_clearml_dataset_version(), tags=tags, dataset_set=set)


if __name__ == '__main__':
    PipelineDecorator.run_locally()
    dataset_creation_pipeline(
        STEM_NAME,
        RANDOM_ASSORTMENTS_PER_SONG,
        source_dir=SOURCE_DIR,
        merged_dir=MERGED_FILES_DIR,
        augmented_dir=AUGMENTED_FILES_DIR,
        distorted_dir=DISTORTED_FILES_DIR,
        encoded_dir=ENCODED_FILES_DIR,
        split_dir=SPLIT_FILES_DIR,
    )
    print('Pipeline completed')