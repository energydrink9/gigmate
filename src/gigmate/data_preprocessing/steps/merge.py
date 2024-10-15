import glob
import itertools
import os
import random
import shutil
from typing import List, Optional, Set, Tuple
from pydub import AudioSegment
from tqdm import tqdm

from gigmate.utils.constants import get_random_seed

SOURCE_FILES_DIR = '/Users/michele/Music/soundstripe/original'
OUTPUT_FILES_DIR = '/Users/michele/Music/soundstripe/merged'
STEM_NAMES = ['guitar', 'drums', 'bass', 'perc', 'fx', 'vocals', 'piano', 'synth', 'winds', 'strings']
BASIC_STEM_NAMES = ['guitar', 'drums', 'bass', 'perc']
STEM_NAME = 'guitar'
RANDOM_ASSORTMENTS_PER_SONG = 3

ADDITIONAL_STEM_NAMES = {
    'guitar': ['guitars', 'gtr'],
    'drums': ['drum', 'drm'],
    'piano': ['keys'],
    'vocals': ['vocal', 'vox'],
}


def get_ogg_file_paths(dir: str):
    return glob.glob(os.path.join(dir, '*.ogg'))


def get_directories_containing_ogg_files(dir: str) -> Set[str]:
    ogg_files = glob.glob(os.path.join(dir, '**/*.ogg'), recursive=True)
    directories = {os.path.dirname(ogg_file) for ogg_file in ogg_files}
    
    return directories


def get_stem_files(file_paths: list, stem_name: str) -> List[str]:

    additional_stem_names = ADDITIONAL_STEM_NAMES.get(stem_name, [])
    current_stem_names = [stem_name] + additional_stem_names

    stem_files = []

    for file_path in file_paths:
        for name in current_stem_names:
            if name.lower() in os.path.basename(file_path).lower():
                stem_files.append(file_path)
                break

    return stem_files


def get_permutation_subset(permutation: list[str]) -> list[str]:
    permutation_length = len(permutation)

    # get a random number between 2 (included) and permutation_length - 1 (included)
    subset_length = random.randint(2, permutation_length - 1)

    return permutation[:subset_length]


def get_random_basic_stem(audio_files: List[str], basic_stem_names: List[str]) -> Optional[str]:

    basic_stems = [stem_file_name for stem_file_name in audio_files if any([stem_name.lower() in os.path.basename(stem_file_name).lower() for stem_name in basic_stem_names])]

    if len(basic_stems) == 0:
        return None
    return random.choice(basic_stems)


def get_assortment(files: List[str], stem_file: str) -> Tuple[str, List[str]]:
    return stem_file, files


def get_random_assortments(audio_files: List[str], stem_file: str, random_assortments_per_song: int) -> List[Tuple[str, List[str]]]:
    permutations = [list(permutation) for permutation in itertools.permutations(audio_files, len(audio_files))]
    permutations_indices = random.sample(range(len(permutations)), min(random_assortments_per_song, len(permutations)))
    return [get_assortment(get_permutation_subset(permutation), stem_file) for i, permutation in enumerate(permutations) if i in permutations_indices]


def create_stems_assortments(audio_files: list[str], stem_file: str, random_assortments_per_song: int) -> List[Tuple[str, List[str]]]:
    stems_count = len(audio_files)
    assortments = []

    if stems_count >= 1:
        all_stems_assortment = get_assortment(audio_files, stem_file)
        assortments.append(all_stems_assortment)

    if stems_count >= 2:
        random_basic_stem = get_random_basic_stem(audio_files, BASIC_STEM_NAMES)
        if random_basic_stem is not None:
            basic_stem_assortment = get_assortment([random_basic_stem], stem_file)
            assortments.append(basic_stem_assortment)
    
    if stems_count >= 3:
        random_assortments = get_random_assortments(audio_files, stem_file, random_assortments_per_song)
        assortments += random_assortments

    return assortments


def assort(directory: str, stem_name: str, random_assortments_per_song: int) -> List[List[Tuple[str, List[str]]]]:
    audio_files = get_ogg_file_paths(directory)
    stem_files = get_stem_files(audio_files, stem_name)

    assortments = []

    for stem_file in stem_files:
        extra_stem_files = audio_files.copy()
        extra_stem_files.remove(stem_file)
        assortments.append(create_stems_assortments(extra_stem_files, stem_file, random_assortments_per_song))
    
    return assortments


def merge_stems(ogg_files, output_file):
    # Load the first stem as the base track
    merged_track = AudioSegment.from_file(ogg_files[0], format="ogg")
    
    # Load and overlay the rest of the stems
    for ogg_file in ogg_files[1:]:
        stem = AudioSegment.from_file(ogg_file, format="ogg")
        merged_track = merged_track.overlay(stem)
    
    # Export the final merged track to a single .ogg file
    merged_track.export(output_file, format="ogg")


def assort_and_merge_all(source_directory: str, output_directory: str, stem_name: str, random_assortments_per_song: int):
    dirs = get_directories_containing_ogg_files(source_directory)
    
    for directory in tqdm(dirs, "Merging audio files"):
        assortments = assort(directory, stem_name, random_assortments_per_song)

        # It is possible to have multiple stems for a stem name (e.g. "vocals" and "vocals_2")
        for i, stem_assortments in enumerate(assortments):
            relative_path = os.path.relpath(directory, source_directory)
                
            for j, assortment in enumerate(stem_assortments):
                song_directory = os.path.join(output_directory, relative_path + f'-inst{i}-assort{j}')
                stem, stems_to_merge = assortment
                output_path = os.path.join(song_directory, "all.ogg")
                if not os.path.exists(output_path):
                    os.makedirs(song_directory, exist_ok=True)
                    merge_stems(stems_to_merge + [stem], output_file=output_path)

                stem_output_file_path = os.path.join(song_directory, "stem.ogg")
                if not os.path.exists(stem_output_file_path):
                    os.makedirs(song_directory, exist_ok=True)
                    shutil.copy(stem, stem_output_file_path)

    return output_directory


if __name__ == '__main__':
    random.seed(get_random_seed())
    assort_and_merge_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR, STEM_NAME, RANDOM_ASSORTMENTS_PER_SONG)