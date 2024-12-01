from scipy.io import wavfile

class InOut:
    @staticmethod
    def readWAV(filename):
        rate, data = wavfile.read(filename)
        N = len(data)
        return data, rate, N

    @staticmethod
    def writeWAV(filename, data, rate):
        wavfile.write(filename, rate, data)