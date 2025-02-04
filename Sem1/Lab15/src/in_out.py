import sounddevice as sd
from scipy.io import wavfile

class InOut:
    @staticmethod
    def readWAV(filename):
        rate, data = wavfile.read(filename)
        N = len(data)
        return data, rate, N

    @staticmethod
    def writeWAV(filename, data, rate):
        wavfile.write(filename, rate, data.astype('int16'))

    @staticmethod
    def record_audio(filename, duration=1, rate=22050):
        audio_data = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='int16')
        sd.wait()
        wavfile.write(filename, rate, audio_data)
        return audio_data.flatten(), rate, len(audio_data)
