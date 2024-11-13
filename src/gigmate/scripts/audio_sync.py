import sounddevice as sd
import numpy as np
from scipy import signal


class AudioSyncTester:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = 'float32'
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 2

        self.stream = sd.OutputStream(samplerate=self.RATE, channels=self.CHANNELS, dtype=self.FORMAT, latency='low')
        self.recording_stream = sd.InputStream(samplerate=self.RATE, channels=self.CHANNELS, dtype=self.FORMAT, latency='low')

    def generate_test_tone(self, duration, frequency):
        t = np.linspace(0, duration, int(self.RATE * duration), False)
        return np.sin(2 * np.pi * frequency * t).astype(np.float32)

    def play_and_record(self):
        print("Playing test tone and recording from microphone...")
        test_tone = self.generate_test_tone(self.RECORD_SECONDS, 440)  # 440 Hz tone
        recorded_audio = np.array([], dtype=np.float32)

        self.stream.start()
        self.recording_stream.start()

        for i in range(0, len(test_tone), self.CHUNK):
            chunk = test_tone[i: i + self.CHUNK]
            self.stream.write(chunk)
            recorded_chunk = self.recording_stream.read(self.CHUNK)[0].flatten()  # Flatten the recorded chunk
            recorded_audio = np.concatenate((recorded_audio, recorded_chunk))

        self.stream.stop()
        self.recording_stream.stop()

        return test_tone, recorded_audio

    def calculate_delay(self, original, recorded):
        correlation = signal.correlate(recorded, original, mode='full')
        delay_samples = np.argmax(correlation) - (len(original) - 1)
        delay_seconds = delay_samples / self.RATE
        return delay_seconds

    def test_sync(self):
        original, recorded = self.play_and_record()
        delay = self.calculate_delay(original, recorded)
        print(f"Estimated delay: {delay:.3f} seconds")

        if abs(delay) < 0.1:  # Allow for 100ms delay
            print("Audio synchronization test PASSED")
        else:
            print("Audio synchronization test FAILED")

    def close(self):
        self.stream.stop()
        self.stream.close()


if __name__ == "__main__":
    tester = AudioSyncTester()
    try:
        tester.test_sync()
    finally:
        tester.close()