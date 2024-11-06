from dataclasses import dataclass
import io
import itertools
import os
import random
from typing import List, Optional, Set, Tuple, cast
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
BASIC_STEM_NAMES = ['guitar', 'drums', 'bass', 'perc']
RANDOM_ASSORTMENTS_PER_SONG = 3
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


def get_directories_containing_ogg_files(fs: S3FileSystem, dir: str) -> Set[str]:
    ogg_files = cast(List, fs.glob(os.path.join(dir, '**/*.ogg')))
    directories = {'s3://' + os.path.dirname(ogg_file) for ogg_file in ogg_files}
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


def get_permutation_subset(permutation: list[str]) -> list[str]:
    permutation_length = len(permutation)

    # get a random number between 2 (included) and permutation_length - 1 (included)
    subset_length = random.randint(2, permutation_length - 1)

    return permutation[:subset_length]


def get_random_basic_stem(stems: List[StemFile], basic_stem_names: List[str]) -> Optional[str]:

    basic_stems = [stem.file_path for stem in stems if not stem.is_mostly_silent and any([stem_name.lower() in os.path.basename(stem.file_path).lower() for stem_name in basic_stem_names])]

    if len(basic_stems) == 0:
        return None
    
    return random.choice(basic_stems)


def get_assortment(other_stems: List[str], current_stem_file: str) -> Tuple[str, List[str]]:
    return current_stem_file, other_stems


def get_random_elements(lst: List[str]) -> List[str]:
    if len(lst) == 0:
        return []
    
    # get a random number between 0 (included) and len(lst) - 1 (included)
    n = random.randint(0, len(lst) - 1)
    return random.sample(lst, n)


def get_random_assortments(non_silent_stems: List[str], mostly_silent_stems: List[str], stem_file: str, random_assortments_per_song: int) -> List[Tuple[str, List[str]]]:
    permutations = [list(permutation) for permutation in itertools.permutations(non_silent_stems, len(non_silent_stems))]
    permutations_indices = random.sample(range(len(permutations)), min(random_assortments_per_song, len(permutations)))
    return [get_assortment(get_permutation_subset(permutation) + get_random_elements(mostly_silent_stems), stem_file) for i, permutation in enumerate(permutations) if i in permutations_indices]


def create_stems_assortments(other_stems: List[StemFile], current_stem_file: str, random_assortments_per_song: int) -> List[Tuple[str, List[str]]]:
    non_mostly_silent_stems_count = len([stem for stem in other_stems if not stem.is_mostly_silent])
    assortments = []

    # 1. all stems assortment
    if non_mostly_silent_stems_count >= 1:
        other_stems_files = [stem.file_path for stem in other_stems]
        all_stems_assortment = get_assortment(other_stems_files, current_stem_file)
        assortments.append(all_stems_assortment)

    # 2. random basic stem assortment
    if non_mostly_silent_stems_count >= 2:
        random_basic_stem = get_random_basic_stem(other_stems, BASIC_STEM_NAMES)
        if random_basic_stem is not None:
            basic_stem_assortment = get_assortment([random_basic_stem], current_stem_file)
            assortments.append(basic_stem_assortment)
    
    # 3. random assortments
    if non_mostly_silent_stems_count >= 3:
        non_silent_other_stems_files = [stem.file_path for stem in other_stems if not stem.is_mostly_silent]
        silent_other_stems_files = [stem.file_path for stem in other_stems if stem.is_mostly_silent]
        random_assortments = get_random_assortments(non_silent_other_stems_files, silent_other_stems_files, current_stem_file, random_assortments_per_song)
        assortments += random_assortments

    return assortments


def is_mostly_silent(fs: S3FileSystem, file_path: str) -> bool:
    with fs.open(file_path, 'rb') as file:
        
        audio, sr = librosa.load(file)  # type: ignore
        no_of_samples = audio.shape[-1]
        splits = librosa.effects.split(audio, top_db=60)
        non_silent_samples = sum([end - start for (start, end) in splits])
        return non_silent_samples / no_of_samples < MIN_PERCENTAGE_OF_AUDIO_IN_NON_SILENT_FILES


def get_stems(fs: S3FileSystem, paths: List[str]) -> List[StemFile]:
    return [StemFile(file_path=path, is_mostly_silent=is_mostly_silent(fs, path)) for path in paths]


def assort(fs: S3FileSystem, directory: str, stem_name: str, random_assortments_per_song: int) -> List[List[Tuple[str, List[str]]]]:
    stems = get_stems(fs, get_ogg_file_paths(fs, directory))
    current_stem_files = get_current_stem_files(stems, stem_name)

    assortments = []

    for stem_file in current_stem_files:
        other_stems = [stem for stem in stems if stem.file_path != stem_file]
        assortments.append(create_stems_assortments(other_stems, stem_file, random_assortments_per_song))
    
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


def assort_directory(params: Tuple[S3FileSystem, str, str, str, str, int]) -> None:

    fs, source_directory, output_directory, directory, stem_name, random_assortments_per_song = params
    assortments = assort(fs, directory, stem_name, random_assortments_per_song)

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
                merge_stems(fs, stems_to_merge + [stem], output_file=output_path)

            stem_output_file_path = os.path.join(song_directory, "stem.ogg")
            if not fs.exists(stem_output_file_path):
                fs.copy(stem, stem_output_file_path)


def assort_and_merge_all(source_directory: str, output_directory: str, stem_name: str, random_assortments_per_song: int):

    client = cast(Client, get_client(RUN_LOCALLY))
    fs = S3FileSystem()

    dirs = get_directories_containing_ogg_files(fs, source_directory)
    
    params_list: List[Tuple[S3FileSystem, str, str, str, str, int]] = [(fs, source_directory, output_directory, directory, stem_name, random_assortments_per_song) for directory in dirs]
    
    print('Assorting and merging audio tracks')
    futures = progress(client.map(assort_directory, params_list))
    client.gather(futures)

    return output_directory


if __name__ == '__main__':
    random.seed(get_random_seed())
    assort_and_merge_all(SOURCE_FILES_PATH, OUTPUT_FILES_DIR, STEM_NAME, RANDOM_ASSORTMENTS_PER_SONG)