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
STEM_NAME = ['guitar', 'gtr']
BASIC_STEM_NAMES = ['guitar', 'drums', 'bass', 'perc']
RANDOM_ASSORTMENTS_PER_SONG = 1

def get_ogg_file_paths(dir: str):
    return glob.glob(os.path.join(dir, '*.ogg'))


def get_directories_containing_ogg_files(dir: str) -> Set[str]:
    ogg_files = glob.glob(os.path.join(dir, '**/*.ogg'), recursive=True)
    directories = {os.path.dirname(ogg_file) for ogg_file in ogg_files}
    
    return directories


def get_stem_file(file_paths: list, stem_name: List[str]) -> Optional[str]:

    for file_path in file_paths:
        for name in stem_name:
            if name.lower() in os.path.basename(file_path).lower():
                return file_path

    return None


def get_permutation_subset(permutation: list[str], index: int) -> list[str]:
    permutation_length = len(permutation)
    if index == 0:
        return permutation # Take all the tracks once

    # get a random number between 2 and permutation_length - 1
    subset_length = random.randint(2, permutation_length - 1)

    return permutation[:subset_length]

def get_random_basic_stem_assortment(audio_files: List[str], basic_stem_names: List[str]) -> List[str]:

    basic_stems = [stem_file_name for stem_file_name in audio_files if any([stem_name.lower() in os.path.basename(stem_file_name).lower() for stem_name in basic_stem_names])]
    return [random.choice(basic_stems)]


def get_assortment(files: List[str], stem_file: str) -> Tuple[str, List[str]]:
    return stem_file, files


def get_rndom_assortments(audio_files: List[str], stem_file: str, random_assortments_per_song: int) -> List[Tuple[str, List[str]]]:
    permutations = [list(combination) for combination in itertools.permutations(audio_files, len(audio_files))]
    permutations_indices = random.sample(range(len(permutations)), min(random_assortments_per_song, len(permutations)))
    return [get_assortment(get_permutation_subset(permutation, i), stem_file) for i, permutation in enumerate(permutations) if i in permutations_indices]


def create_stems_assortments(audio_files: list[str], stem_file: str, random_assortments_per_song: int) -> List[Tuple[str, List[str]]]:
    all_stems_assortment = get_assortment(audio_files, stem_file)
    basic_stem_assortment = get_assortment(get_random_basic_stem_assortment(audio_files, BASIC_STEM_NAMES), stem_file)
    random_assortments = get_rndom_assortments(audio_files, stem_file, random_assortments_per_song)

    stems_count = len(audio_files)

    if stems_count == 1:
        return [all_stems_assortment]
    if stems_count == 2:
        return [all_stems_assortment, basic_stem_assortment]
    
    return [all_stems_assortment, basic_stem_assortment] + random_assortments


def assort(directory: str, stem_name: List[str], random_assortments_per_song: int) -> Optional[List[Tuple[str, List[str]]]]:
    audio_files = get_ogg_file_paths(directory)
    stem_file = get_stem_file(audio_files, stem_name)
    
    if stem_file is None:
        return None

    extra_stem_files = audio_files.copy()
    extra_stem_files.remove(stem_file)
    return create_stems_assortments(extra_stem_files, stem_file, random_assortments_per_song)
    

def merge_stems(ogg_files, output_file):
    # Load the first stem as the base track
    merged_track = AudioSegment.from_file(ogg_files[0], format="ogg")
    
    # Load and overlay the rest of the stems
    for ogg_file in ogg_files[1:]:
        stem = AudioSegment.from_file(ogg_file, format="ogg")
        merged_track = merged_track.overlay(stem)
    
    # Export the final merged track to a single .ogg file
    merged_track.export(output_file, format="ogg")


def merge(stems_to_merge: List[str], stem: str, output_directory: str, index: int) -> None:
    os.makedirs(output_directory, exist_ok=True)
    merge_stems(stems_to_merge + [stem], os.path.join(output_directory, f"all-{index}.ogg"))

def assort_and_merge_all(source_directory: str, output_directory: str, stem_name: List[str], random_assortments_per_song: int):
    dirs = get_directories_containing_ogg_files(source_directory)
    
    for directory in tqdm(dirs, "Merging audio files"):
        assortments = assort(directory, stem_name, random_assortments_per_song)

        if assortments is not None:
            for i, assortment in enumerate(assortments):
                stem, stems_to_merge = assortment
                relative_path = os.path.relpath(directory, source_directory)
                song_directory = os.path.join(output_directory, relative_path)
                merge(stems_to_merge, stem, song_directory, i)
                shutil.copy(stem, os.path.join(song_directory, "stem.ogg"))

    return output_directory


if __name__ == '__main__':
    random.seed(get_random_seed())
    assort_and_merge_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR, STEM_NAME, RANDOM_ASSORTMENTS_PER_SONG)