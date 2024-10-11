from pretty_midi import PrettyMIDI, Instrument
import pretty_midi


def get_midi_bass_fixture_1():
    fixture = PrettyMIDI()
    bass_program = pretty_midi.instrument_name_to_program('Electric Bass (pick)')
    bass = Instrument(program=bass_program)

    for i in range(10):
        for j, note_name in enumerate(['C5', 'E5', 'G5']):
            note_number = pretty_midi.note_name_to_number(note_name)
            start = (i * 3) + j
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=start, end=start + 1)
            bass.notes.append(note)

    fixture.instruments.append(bass)
    # fixture.write('output/midi_bass_fixture_1.mid')
    return fixture