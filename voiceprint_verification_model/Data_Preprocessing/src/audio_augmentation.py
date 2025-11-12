import librosa
import numpy as np

def pitch_shift(data, sampling_rate, n_steps=4):
    """Shift the pitch of the audio data."""
    return librosa.effects.pitch_shift(data, sampling_rate, n_steps=n_steps)

def time_stretch(data, rate=1.25):
    """Speed up or slow down the audio without changing pitch."""
    return librosa.effects.time_stretch(data, rate)

def add_background_noise(data, noise_factor=0.005):
    """Add random noise to audio data to simulate background noise."""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data
