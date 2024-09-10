
import random
from symusic import Score


def generate_random_filename(prefix: str = 'temp_', extension: str = '.wav') -> str:
    return f'output/{prefix}{random.randint(0, 1000000)}{extension}'

def calculate_score_length_in_seconds(score: Score) -> float:
    score_end = score.end()
    score_tpq = score.tpq
    tempos = score.tempos

    total_seconds = 0
    current_tick = 0
    current_tempo_index = 0

    while current_tick < score_end and current_tempo_index < len(tempos):
        current_tempo = tempos[current_tempo_index]
        next_tempo_tick = score_end if current_tempo_index == len(tempos) - 1 else tempos[current_tempo_index + 1].time
        
        # Calculate ticks in this tempo section
        section_ticks = min(next_tempo_tick, score_end) - current_tick
        
        # Calculate duration of this section
        bpm = current_tempo.qpm
        seconds_per_beat = 60 / bpm
        quarter_notes = section_ticks / score_tpq
        section_seconds = quarter_notes * seconds_per_beat
        
        total_seconds += section_seconds
        
        current_tick = next_tempo_tick
        current_tempo_index += 1

    return total_seconds