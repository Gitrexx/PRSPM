from scipy.io import wavfile
import librosa

data, sample_rate = librosa.load('test_materials/higher_volume.wav',sr=None)
data = data * 4
wavfile.write('louder.wav',sample_rate,data)
