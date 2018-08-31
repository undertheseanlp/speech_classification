import librosa
import numpy as np

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


def extract_features(filepath, bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []

    sound_clip, s = librosa.load(filepath)
    for start, end in windows(sound_clip, window_size):
        if len(sound_clip[start:end]) == window_size:
            signal = sound_clip[start:end]
            melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
            logspec = librosa.logamplitude(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    return features