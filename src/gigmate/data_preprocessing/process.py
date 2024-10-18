from clearml import Dataset, PipelineDecorator
import argparse

from gigmate.data_preprocessing.pipeline import AUGMENTED_FILES_DIR, DISTORTED_FILES_DIR, ENCODED_FILES_DIR, MERGED_FILES_DIR, RANDOM_ASSORTMENTS_PER_SONG, SPLIT_FILES_DIR, STEM_NAME, dataset_creation_pipeline


def get_remote_dataset_by_id(id: str):
    dataset = Dataset.get(
        dataset_id=id,
        only_completed=False, 
        only_published=False, 
    )
    return dataset.get_local_copy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create a dataset from an already pre-processed dataset")
    parser.add_argument("dataset_id", help="Id of the pre-processed dataset.", type=str)
    args = parser.parse_args()
    PipelineDecorator.run_locally()
    dataset_creation_pipeline(
        STEM_NAME,
        RANDOM_ASSORTMENTS_PER_SONG,
        source_dir=get_remote_dataset_by_id(args.dataset_id),
        merged_dir=MERGED_FILES_DIR,
        augmented_dir=AUGMENTED_FILES_DIR,
        distorted_dir=DISTORTED_FILES_DIR,
        encoded_dir=ENCODED_FILES_DIR,
        split_dir=SPLIT_FILES_DIR,
    )
    print('Pipeline completed')
