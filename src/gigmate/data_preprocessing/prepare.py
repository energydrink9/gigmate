import argparse
from clearml import PipelineDecorator

from gigmate.data_preprocessing.pipeline import dataset_preparation_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pre-process and compress audio stems")
    parser.add_argument("source_dir", help="Path to the directory containing the original compressed audio stems.", type=str)
    args = parser.parse_args()
    PipelineDecorator.run_locally()
    dataset_preparation_pipeline(
        source_dir=args.source_dir,
    )
    print('Pipeline completed')