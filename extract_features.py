import os
import numpy as np
import librosa

SAMPLE_RATE = 16000
N_MFCC = 13
FIXED_FRAMES = 63  # every sample will be padded/trimmed to this many time-windows

def extract_mfcc(filepath, n_mfcc=N_MFCC, sample_rate=SAMPLE_RATE, is_training=False):
    audio, sr = librosa.load(filepath, sr=sample_rate)
    audio, _ = librosa.effects.trim(audio, top_db=25)  # strip leading/trailing silence
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if not is_training:
        return mfccs

    if mfccs.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode="constant")
    else:
        best_energy = -1
        best_start = 0
        for i in range(mfccs.shape[1] - FIXED_FRAMES + 1):
            window = mfccs[:, i:i+FIXED_FRAMES]
            energy = np.sum(window**2)
            if energy > best_energy:
                best_energy = energy
                best_start = i
        mfccs = mfccs[:, best_start:best_start+FIXED_FRAMES]

    return mfccs

def main():
    features, labels = [], []

    for label, class_id in [("positive", 1), ("negative", 0)]:
        folder = os.path.join("data", label)
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(folder, fname)
            mfcc = extract_mfcc(path, is_training=True)
            features.append(mfcc)
            labels.append(class_id)
            print(f"Processed {path} -> shape {mfcc.shape}")

    features = np.array(features)
    labels = np.array(labels)

    print(f"\nTotal samples: {len(labels)}")
    print(f"Positive: {sum(labels == 1)}  Negative: {sum(labels == 0)}")

    np.savez("features.npz", features=features, labels=labels)
    print("Saved features.npz")

if __name__ == "__main__":
    main()