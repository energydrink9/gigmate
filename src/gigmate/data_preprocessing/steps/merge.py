from dataclasses import dataclass
import io
import os
import random
from typing import FrozenSet, List, Optional, Tuple, cast, Set
import librosa
from pydub import AudioSegment
from dask.distributed import progress, Client
from s3fs.core import S3FileSystem

from gigmate.data_preprocessing.cluster import get_client
from gigmate.data_preprocessing.constants import STEM_NAME
from gigmate.utils.constants import get_random_seed

SOURCE_FILES_PATH = 's3://soundstripe-dataset-original/original'
OUTPUT_FILES_DIR = 's3://soundstripe-dataset-original/merged'
STEM_NAMES = ['guitar', 'drums', 'bass', 'perc', 'fx', 'vocals', 'piano', 'synth', 'winds', 'strings']
BASIC_STEM_NAMES = ['guitar', 'drums', 'bass', 'perc', 'gtr', 'drm']
MAX_RANDOM_FULL_ASSORTMENTS_PER_SONG = 3
MAX_BASIC_STEM_RANDOM_ASSORTMENTS_PER_SONG = 4
MIN_PERCENTAGE_OF_AUDIO_IN_NON_SILENT_FILES = 0.5

# Set this flag to True to run locally (i.e. not on Coiled)
RUN_LOCALLY = False

ADDITIONAL_STEM_NAMES = {
    'guitar': ['guitars', 'gtr'],
    'drums': ['drum', 'drm'],
    'piano': ['keys'],
    'vocals': ['vocal', 'vox'],
}


@dataclass
class StemFile:
    file_path: str
    is_mostly_silent: bool


def get_ogg_file_paths(fs: S3FileSystem, dir: str) -> List[str]:
    return ['s3://' + path for path in cast(List[str], fs.glob(os.path.join(dir, '*.ogg')))]


def get_directories_containing_ogg_files(fs: S3FileSystem, dir: str) -> FrozenSet[str]:
    ogg_files = cast(List, fs.glob(os.path.join(dir, '**/*.ogg')))
    directories = frozenset({'s3://' + os.path.dirname(ogg_file) for ogg_file in ogg_files})
    return directories


def get_current_stem_files(stems: List[StemFile], stem_name: str) -> List[str]:

    additional_stem_names = ADDITIONAL_STEM_NAMES.get(stem_name, [])
    current_stem_names = [stem_name] + additional_stem_names

    current_stem_files: List[str] = []

    for stem in stems:
        for name in current_stem_names:
            if name.lower() in os.path.basename(stem.file_path).lower():
                if not stem.is_mostly_silent:
                    current_stem_files.append(stem.file_path)
                    break

    return current_stem_files


def get_basic_stems(stems: FrozenSet[str], basic_stem_names: List[str]) -> FrozenSet[str]:
    return frozenset({stem for stem in stems if any([stem_name.lower() in os.path.basename(stem).lower() for stem_name in basic_stem_names])})


def get_random_stem(stems: FrozenSet[str], exclude: FrozenSet[str] = frozenset({})) -> Optional[str]:

    if len(stems) == 0:
        return None
    
    return random.choice(list(stems.difference(exclude)))


def get_assortment(other_stems: FrozenSet[str], current_stem_file: str) -> Tuple[str, FrozenSet[str]]:
    return current_stem_file, other_stems


def get_stem_files_paths(stems: List[StemFile]) -> FrozenSet[str]:
    return frozenset({stem.file_path for stem in stems})


def create_stems_assortments(other_stems: List[StemFile], current_stem_file: str) -> List[Tuple[str, FrozenSet[str]]]:
    other_stems_paths = get_stem_files_paths(other_stems)
    non_silent_stems_paths = get_stem_files_paths([stem for stem in other_stems if not stem.is_mostly_silent])
    non_silent_basic_stems_paths = get_basic_stems(non_silent_stems_paths, BASIC_STEM_NAMES)
    non_basic_stem_paths = other_stems_paths.difference(non_silent_basic_stems_paths)

    assortments: Set[FrozenSet[str]] = set()

    # 1. all stems assortment
    other_stems_files = get_stem_files_paths(other_stems)
    all_stems_assortment = other_stems_files
    assortments.add(all_stems_assortment)

    if len(non_silent_basic_stems_paths) > 0:

        exclude: FrozenSet[str] = frozenset()  # Keep track of already seen stems in order to exclude them from sampling

        # 2. random basic stem assortments
        for _ in range(MAX_BASIC_STEM_RANDOM_ASSORTMENTS_PER_SONG):
            random_basic_stem = get_random_stem(non_silent_basic_stems_paths, exclude)
            if random_basic_stem is not None:
                assortments.add(frozenset({random_basic_stem}))
                exclude = exclude.union(random_basic_stem)
        
        # 3. random full assortments
        if len(non_basic_stem_paths) > 0:

            for _ in range(MAX_RANDOM_FULL_ASSORTMENTS_PER_SONG):
                number_of_basic_stems = random.randint(1, len(non_silent_basic_stems_paths))
                number_of_non_basic_stems = random.randint(1, len(non_basic_stem_paths))

                full_assortment_paths: Set[str] = set()
                for _ in range(number_of_basic_stems):
                    random_basic_stem = get_random_stem(non_silent_basic_stems_paths)
                    if random_basic_stem is not None:
                        full_assortment_paths.add(random_basic_stem)

                for _ in range(number_of_non_basic_stems):
                    random_non_basic_stem = get_random_stem(non_basic_stem_paths)
                    if random_non_basic_stem is not None:
                        full_assortment_paths.add(random_non_basic_stem)

                assortments.add(frozenset(full_assortment_paths))

    return [(current_stem_file, assortment) for assortment in assortments]


def is_mostly_silent(fs: S3FileSystem, file_path: str) -> bool:
    with fs.open(file_path, 'rb') as file:
        
        audio, sr = librosa.load(file)  # type: ignore
        no_of_samples = audio.shape[-1]
        splits = librosa.effects.split(audio, top_db=60)
        non_silent_samples = sum([end - start for (start, end) in splits])
        return non_silent_samples / no_of_samples < MIN_PERCENTAGE_OF_AUDIO_IN_NON_SILENT_FILES


def get_stem(file_path: str, silent: bool) -> StemFile:
    return StemFile(file_path=file_path, is_mostly_silent=silent)


def get_stems(fs: S3FileSystem, paths: List[str]) -> List[StemFile]:
    return [get_stem(path, is_mostly_silent(fs, path)) for path in paths]


def assort(fs: S3FileSystem, directory: str, stem_name: str) -> List[List[Tuple[str, FrozenSet[str]]]]:
    stems = get_stems(fs, get_ogg_file_paths(fs, directory))
    current_stem_files = get_current_stem_files(stems, stem_name)

    assortments = []

    for stem_file in current_stem_files:
        other_stems = [stem for stem in stems if stem.file_path != stem_file]
        assortments.append(create_stems_assortments(other_stems, stem_file))
    
    return assortments


def merge_stems(fs: S3FileSystem, ogg_files: List[str], output_file: str):
    # Load the first stem as the base track
    with fs.open(ogg_files[0], 'rb') as first_file:
        bytes_io = io.BytesIO(first_file.read())  # type: ignore
        merged_track = AudioSegment.from_file(bytes_io, format="ogg", codec='libopus')  # type: ignore
    
    # Load and overlay the rest of the stems
    for ogg_file in ogg_files[1:]:
        with fs.open(ogg_file, 'rb') as file:
            bytes_io = io.BytesIO(file.read())  # type: ignore
            stem = AudioSegment.from_file(bytes_io, format="ogg", codec='libopus')  # type: ignore
            merged_track = merged_track.overlay(stem)
    
    # Export the final merged track to a single .ogg file
    with fs.open(output_file, 'wb') as file:
        bytes_io = io.BytesIO()
        merged_track.export(bytes_io, format='ogg', codec='libopus')  # type: ignore
        file.write(bytes_io.getvalue())  # type: ignore


def assort_directory(params: Tuple[S3FileSystem, str, str, str, str]) -> None:

    fs, source_directory, output_directory, directory, stem_name = params
    assortments = assort(fs, directory, stem_name)

    # It is possible to have multiple stems for a stem name (e.g. "vocals" and "vocals_2")
    for i, stem_assortments in enumerate(assortments):
        relative_path = os.path.relpath(directory, source_directory)
            
        for j, assortment in enumerate(stem_assortments):
            song_directory = os.path.join(output_directory, relative_path + f'-inst{i}-assort{j}')
            stem, stems_to_merge = assortment
            if not fs.exists(song_directory):
                fs.makedirs(song_directory, exist_ok=True)

            output_path = os.path.join(song_directory, "all.ogg")
            if not fs.exists(output_path):
                merge_stems(fs, list(stems_to_merge) + [stem], output_file=output_path)

            stem_output_file_path = os.path.join(song_directory, "stem.ogg")
            if not fs.exists(stem_output_file_path):
                fs.copy(stem, stem_output_file_path)


def assort_and_merge_all(source_directory: str, output_directory: str, stem_name: str):

    client = cast(Client, get_client(RUN_LOCALLY))
    fs = S3FileSystem()

    dirs = get_directories_containing_ogg_files(fs, source_directory)
    
    params_list: List[Tuple[S3FileSystem, str, str, str, str]] = [(fs, source_directory, output_directory, directory, stem_name) for directory in dirs]
    
    print('Assorting and merging audio tracks')
    futures = progress(client.map(assort_directory, params_list))
    client.gather(futures)

    return output_directory


if __name__ == '__main__':
    random.seed(get_random_seed())
    assort_and_merge_all(SOURCE_FILES_PATH, OUTPUT_FILES_DIR, STEM_NAME)