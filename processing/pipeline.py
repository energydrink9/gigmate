from clearml import PipelineDecorator
from gigmate.constants import get_clearml_project_name

from gigmate.constants import get_clearml_project_name
from gigmate.processing.steps.split import split_midi_files
from gigmate.processing.steps.augment import augment_midi_files
from gigmate.processing.steps.tokenize import tokenize_midi_files

@PipelineDecorator.component(return_values=['split_output_dir'], cache=True)
def split_midi():
    split_output_dir = split_midi_files()
    return split_output_dir

@PipelineDecorator.component(return_values=['augmented_output_dir'], cache=True)
def augment_midi(split_output):
    augmented_output_dir = augment_midi_files(split_output)
    return augmented_output_dir

@PipelineDecorator.component(return_values=['augmented_output_dir'], cache=True)
def tokenize_midi(augmented_output_dir):
    tokenized_output = tokenize_midi_files(augmented_output_dir)
    return tokenized_output

@PipelineDecorator.pipeline(
    name='MIDI Processing Pipeline',
    project=get_clearml_project_name(),
    version='1.0'
)
def midi_processing_pipeline():
    split_output = split_midi()
    augmented_output = augment_midi(split_output)
    tokenize_midi(augmented_output)

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    midi_processing_pipeline()
    print('Pipeline completed')