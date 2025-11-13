import pandas as pd
import librosa
import os
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    energy = np.sum(y**2) / len(y)
    features = np.hstack((mfccs_mean, spectral_rolloff, energy))
    return features

def process_audio_folder(folder_path, output_csv):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav') or filename.endswith('.opus'):
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path)
            label = filename.split('_')[0]  # Customize label extraction as needed
            row = np.hstack((features, label))
            data.append(row)
    columns = [f"mfcc_{i+1}" for i in range(13)] + ['spectral_rolloff', 'energy', 'label']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")
