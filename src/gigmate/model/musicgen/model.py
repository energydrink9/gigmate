import torchaudio
from audiocraft.models.musicgen import MusicGen
from audiocraft.data.audio import audio_read, audio_write


def generate_model_medium():
    model_medium = MusicGen.get_pretrained('facebook/musicgen-melody')
    model_medium.set_generation_params(duration=5)  # generate 5 seconds.
    descriptions: list[str] = ['unplugged']
    prompt, prompt_sample_rate = audio_read('output/nirvana_the_man_who_sold_the_world_started.ogg', 0, 3)
    wav = model_medium.generate_continuation(prompt, prompt_sample_rate, descriptions=descriptions, progress=True)  # generates 3 samples.

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'output/musicgen_{idx}', one_wav.cpu(), model_medium.sample_rate, strategy="loudness", loudness_compressor=True)

def generate_model_small():
    model_small = MusicGen.get_pretrained('facebook/musicgen-small')
    model_small.set_generation_params(duration=5)  # generate 5 seconds.
    descriptions: list[str] = ['unplugged live rock music']
    wav = model_small.generate(descriptions=descriptions, progress=True)  # generates 3 samples.

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'output/musicgen_{idx}', one_wav.cpu(), model_small.sample_rate, strategy="loudness", loudness_compressor=True)

if __name__ == '__main__':
    generate_model_small()