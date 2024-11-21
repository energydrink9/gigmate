
from gigmate.data_preprocessing.steps.augment import augment_pitch_and_tempo


TEST_FILE_PATH_ALL = 'resources/all.ogg'
TEST_FILE_PATH_STEM = 'resources/stem.ogg'


def check_augment_pitch_and_tempo() -> None:
    
    augment_pitch_and_tempo([(TEST_FILE_PATH_ALL, f'{TEST_FILE_PATH_ALL}-augmented.ogg')])
    augment_pitch_and_tempo([(TEST_FILE_PATH_STEM, f'{TEST_FILE_PATH_STEM}-augmented.ogg')])

    print('Augmented files generated successfully')


if __name__ == '__main__':
    check_augment_pitch_and_tempo()