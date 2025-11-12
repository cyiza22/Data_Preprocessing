import librosa
import numpy as np

def pitch_shift(data, sampling_rate, n_steps=4):
    return librosa.effects.pitch_shift(data, sampling_rate, n_steps=n_steps)

def time_stretch(data, rate=1.25):
    return librosa.effects.time_stretch(data, rate)

def add_background_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data
