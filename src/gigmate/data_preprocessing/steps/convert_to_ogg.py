import glob
import os
from pydub import AudioSegment
from tqdm import tqdm

WAV_FILES_DIR = '/Users/michele/Music/soundstripe/original'
BITRATE = '160k'


def get_wav_files(dir: str):
    return glob.glob(os.path.join(dir, '**/*.wav'), recursive=True)


def convert_to_ogg(directory: str) -> str:
    files = get_wav_files(directory)
    
    for filename in tqdm(files, "Converting files to Ogg Vorbis format"):
        # Load the WAV file
        audio = AudioSegment.from_wav(filename)

        # Export the file as OGG
        ogg_file = os.path.join(os.path.dirname(filename), os.path.basename(filename).replace('.wav', '.ogg'))
        audio.export(ogg_file, format="ogg", codec="libvorbis", bitrate=BITRATE)
        os.remove(filename)
    
    return directory


if __name__ == '__main__':
    convert_to_ogg(WAV_FILES_DIR)