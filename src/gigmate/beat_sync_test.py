import pyaudio
import numpy as np
import time

class MicrophoneSyncPlayer:
    def __init__(self):
        self.CHUNK = 4096
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                 channels=self.CHANNELS,
                                 rate=self.RATE,
                                 input=True,
                                 output=True,
                                 frames_per_buffer=self.CHUNK)

        # Load your sample audio
        self.sample = self.load_sample("output/loop.wav")
        self.sample_pos = 0

    def load_sample(self, file_path):
        """Load a WAV file as a numpy array"""
        return np.fromfile(file_path, dtype=np.float32)

    def play_sync(self):
        print("Playing sample in sync with microphone input...")

        while True:
            # Read audio chunk from microphone
            mic_data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.float32)

            # Play a chunk of the sample audio
            sample_chunk = self.sample[self.sample_pos:self.sample_pos+self.CHUNK]
            self.stream.write(sample_chunk.tobytes())
            self.sample_pos = (self.sample_pos + self.CHUNK) % len(self.sample)

            # Add any beat detection logic here to trigger the sample playback

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    player = MicrophoneSyncPlayer()
    try:
        player.play_sync()
    finally:
        player.close()