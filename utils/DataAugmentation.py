import numpy as np
import librosa
from numpy.random._examples.cffi.extending import rng
import soundfile as sf
import random


def volume_augment(samples, min_gain_dBFS=-10, max_gain_dBFS=10):

    samples = samples.copy()
    data_type = samples[0].dtype
    gain = rng.uniform(min_gain_dBFS, max_gain_dBFS)
    gain = 10. ** (gain / 20.)
    samples = samples * gain
    samples = samples.astype(data_type)
    return samples

def speed_librosa(samples, min_speed=0.9, max_speed=1.1):

    samples = samples.copy()
    data_type = samples[0].dtype

    speed = rng.uniform(min_speed, max_speed)
    samples = samples.astype(np.float)
    samples = librosa.effects.time_stretch(samples, speed)
    samples = samples.astype(data_type)
    return samples

def time_shift_numpy(samples, max_ratio=0.05):

    samples = samples.copy()
    data_type = samples[0].dtype
    frame_num = samples.shape[0]
    max_shifts = frame_num * max_ratio
    nb_shifts = np.random.randint(-max_shifts, max_shifts)
    samples = np.roll(samples, nb_shifts, axis=0)
    samples = samples.astype(data_type)
    return samples

def gaussian_white_noise_numpy(samples, min_snr = 30, max_snr=40):

    snr = random.randint(min_snr,max_snr)
    P_signal = np.mean(samples**2)
    k = np.sqrt(P_signal / 10 ** (snr / 10.0))
    return samples + np.random.randn(len(samples)) * k

# THIS FUNCTION TO RANDOM AUGMENT ON AUDIO
def randomAugment(sample):
    choice = random.randint(1,10)
    if choice < 5:
        return sample

    choice = random.randint(1,10)
    if choice > 5:
        sample = volume_augment(sample)

    choice = random.randint(1,10)
    if choice > 5:
        sample = speed_librosa(sample)

    choice = random.randint(1,10)
    if choice > 5:
        sample = time_shift_numpy(sample)

    choice = random.randint(1,10)
    if choice > 5:
        sample = gaussian_white_noise_numpy(sample)
    return sample



if __name__ == "__main__":
    wave, sr = sf.read("noise_test.flac")
    voice = randomAugment(wave)
    sf.write("./aug/finalfuk.flac", voice, sr)
    # voice = volume_augment(wave)
    # speed = speed_librosa(wave)
    # timeshift = time_shift_numpy(wave)
    # noise = gaussian_white_noise_numpy(wave)
    #
    # sf.write("./aug/voice.flac",voice,sr)
    # sf.write("./aug/speed.flac", speed, sr)
    # sf.write("./aug/time.flac", timeshift, sr)
    # sf.write("./aug/noise.flac", noise, sr)

