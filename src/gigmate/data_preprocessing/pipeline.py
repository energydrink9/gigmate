from typing import List
from clearml import PipelineDecorator

from gigmate.data_preprocessing.process import upload_dataset
from gigmate.data_preprocessing.steps.convert_to_ogg import convert_to_ogg
from gigmate.data_preprocessing.steps.merge import assort_and_merge_all
from gigmate.data_preprocessing.steps.uncompress import uncompress_files
from gigmate.utils.constants import get_clearml_dataset_version, get_clearml_project_name
from gigmate.utils.constants import get_clearml_project_name
from gigmate.data_preprocessing.steps.augment import augment_all

SOURCE_DIR = '/Users/michele/Music/soundstripe/original'
MERGED_FILES_DIR = '/Users/michele/Music/soundstripe/merged'
STEM_NAME = ['guitar', 'gtr']
BASIC_STEM_NAMES = ['guitar', 'drums', 'bass', 'perc']
RANDOM_ASSORTMENTS_PER_SONG = 1
DATASET_TAGS = ['small']

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
def augment_step(merged_dir: str) -> str:
    print(f'Augmenting dataset')
    output_dir = augment_all(merged_dir)
    return output_dir

@PipelineDecorator.pipeline(
    name='Dataset creation pipeline',
    project=get_clearml_project_name(),
    version='1.0'
)
def dataset_creation_pipeline(source_dir: str, merged_dir: str, stem_name: List[str], random_assortments_per_song: int):
    uncompressed_dir = uncompress_step(source_dir)
    converted_to_ogg_dir = convert_to_ogg_step(uncompressed_dir)
    merged_dir = assort_and_merge_step(converted_to_ogg_dir, merged_dir, stem_name, random_assortments_per_song)
    augmented_dir = augment_step(merged_dir)
    upload_dataset(path=augmented_dir, version=get_clearml_dataset_version(), tags=DATASET_TAGS)

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    dataset_creation_pipeline(SOURCE_DIR, MERGED_FILES_DIR, STEM_NAME, RANDOM_ASSORTMENTS_PER_SONG)
    print('Pipeline completed')