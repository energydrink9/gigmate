from gigmate.__fixtures__.midi_fixtures import get_midi_bass_fixture_1
from gigmate.midi_conversion import convert_midi_to_wav, convert_wav_to_midi
import pretty_midi
from pydub import AudioSegment

OUTPUT_SAMPLE_RATE = 22050

def test_midi_conversion_returns_sequence_of_correct_length():
    fixture = get_midi_bass_fixture_1()
    fixture_length = fixture.get_end_time()

    wav_file = convert_midi_to_wav(fixture, OUTPUT_SAMPLE_RATE)
    fixture_length_wav = len(AudioSegment.from_wav(wav_file)) / 1000

    back_to_midi = convert_wav_to_midi(wav_file)
    back_to_midi_length = back_to_midi.get_end_time()

    back_to_midi_wav_file = convert_midi_to_wav(back_to_midi, OUTPUT_SAMPLE_RATE)
    back_to_midi_wav_length = len(AudioSegment.from_wav(back_to_midi_wav_file)) / 1000

    print(f"MIDI events in original fixture:")
    for instrument in fixture.instruments:
        for note in instrument.notes:
            print(f"Note: {pretty_midi.note_number_to_name(note.pitch)}, Start: {note.start}, End: {note.end}")
    
    print("\nMIDI events after conversion:")
    for instrument in back_to_midi.instruments:
        for note in instrument.notes:
            print(f"Note: {pretty_midi.note_number_to_name(note.pitch)}, Start: {note.start}, End: {note.end}")

    print(f"Fixture length midi: {fixture_length}. Back to midi midi length: {back_to_midi_length}.")
    print(f"Fixture length wav: {fixture_length_wav}. Back to midi wav length: {back_to_midi_wav_length}.")

    assert abs(fixture_length - back_to_midi_length) < 0.1
    assert abs(fixture_length_wav - back_to_midi_wav_length) < 0.1
    