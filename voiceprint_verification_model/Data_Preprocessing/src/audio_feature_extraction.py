import pandas as pd
import librosa
import os
import numpy as np

def extract_features(audio_path):
    """
    Extract MFCCs (13 coefficients), spectral rolloff, and energy from an audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    energy = np.sum(y**2) / len(y)
    features = np.hstack((mfccs_mean, spectral_rolloff, energy))
    return features

def process_audio_folder(folder_path, output_csv):
    """
    Process all audio files in folder, extract features, and save to CSV.
    Assumes filenames hold user labels before first underscore.
    """
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.wav', '.opus')):
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path)
            label = filename.split('_')[0]  # Extract label from filename
            data.append(features)
            labels.append(label)
    columns = [f"mfcc_{i+1}" for i in range(13)] + ['spectral_rolloff', 'energy']
    df = pd.DataFrame(data, columns=columns)
    df['label'] = labels
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")
