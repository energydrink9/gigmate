import sounddevice as sd
import numpy as np
import time
from scipy.signal import correlate, find_peaks


def generate_tone(duration, frequency, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def measure_audio_delay(duration=1, frequency=1000, sample_rate=44100):
    tone = generate_tone(duration, frequency, sample_rate)
    
    # Prepare the output stream
    output_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype=np.float32, blocksize=0, latency='low')
    output_stream.start()

    # Prepare the input stream
    input_stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32, blocksize=0, latency='low')
    input_stream.start()

    # Play the tone
    start_time = time.perf_counter()
    output_stream.write(tone)
    
    # Record audio
    recorded_audio, _ = input_stream.read(int(sample_rate * (duration + 1)))
    end_time = time.perf_counter()

    # Stop streams
    output_stream.stop()
    input_stream.stop()

    recorded_audio = recorded_audio.flatten()
    
    # Find the start of the tone in the recorded audio
    correlation = correlate(recorded_audio, tone)
    peaks, _ = find_peaks(correlation, height=0.88 * np.max(correlation))
    if len(peaks) > 0:
        start_sample = peaks[0]
    else:
        raise ValueError("Could not detect the tone in the recorded audio")
    
    # Calculate delay
    playback_start_time = start_sample / sample_rate
    total_delay = end_time - start_time

    return playback_start_time, total_delay


# Measure the delay multiple times and take the average
num_measurements = 5
playback_delays, total_delays = [], []

for i in range(num_measurements):
    try:
        playback_delay, total_delay = measure_audio_delay()
        if 0 <= playback_delay <= total_delay:
            playback_delays.append(playback_delay)
            total_delays.append(total_delay)
            print(f"Measurement {i+1}: Playback delay = {playback_delay:.3f}s, Total delay = {total_delay:.3f}s")
        else:
            print(f"Measurement {i+1}: Invalid results, skipping")
    except Exception as e:
        print(f"Error in measurement {i+1}: {str(e)}")

if playback_delays and total_delays:
    avg_playback_delay = sum(playback_delays) / len(playback_delays)
    avg_total_delay = sum(total_delays) / len(total_delays)
    print(f"\nAverage playback delay: {avg_playback_delay:.3f} seconds")
    print(f"Average total delay: {avg_total_delay:.3f} seconds")
else:
    print("No valid measurements were obtained.")
