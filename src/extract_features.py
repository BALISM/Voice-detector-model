import os
import numpy as np
import librosa

import warnings

SAMPLE_RATE = 16000
N_MFCC = 13
FIXED_FRAMES = 63  # every sample will be padded/trimmed to this many time-windows

def extract_mfcc(filepath, n_mfcc=N_MFCC, sample_rate=SAMPLE_RATE, is_training=False, augment=False):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: '{filepath}'")
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sr = librosa.load(filepath, sr=sample_rate)
    
    if augment:
        # Add random noise
        noise = np.random.randn(len(audio))
        audio = audio + 0.005 * noise
        # Random pitch shift
        n_steps = np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        # Random time stretch (speed up or slow down slightly)
        rate = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)
        # Random volume change
        volume_factor = np.random.uniform(0.7, 1.3)
        audio = audio * volume_factor
        
    audio, _ = librosa.effects.trim(audio, top_db=25)  # strip leading/trailing silence
    
    # Extract MFCCs + delta + delta-delta (39 features instead of 13)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    features = np.concatenate([mfccs, delta, delta2], axis=0)  # shape: (39, T)
    
    # Normalize each feature to zero mean, unit variance
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True) + 1e-8
    features = (features - mean) / std

    if not is_training:
        return features

    if features.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    else:
        best_energy = -1
        best_start = 0
        for i in range(features.shape[1] - FIXED_FRAMES + 1):
            window = features[:, i:i+FIXED_FRAMES]
            energy = np.sum(window**2)
            if energy > best_energy:
                best_energy = energy
                best_start = i
        features = features[:, best_start:best_start+FIXED_FRAMES]

    return features

def main():
    all_features, labels = [], []

    for label, class_id in [("positive", 1), ("negative", 0)]:
        folder = os.path.join("data", "dataset", label)
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(folder, fname)
            # Always add original
            feat = extract_mfcc(path, is_training=True, augment=False)
            all_features.append(feat)
            labels.append(class_id)
            
            # Add augmented versions
            n_aug = 4 if class_id == 1 else 2
            for _ in range(n_aug):
                feat_aug = extract_mfcc(path, is_training=True, augment=True)
                all_features.append(feat_aug)
                labels.append(class_id)
                
            print(f"Processed {path} (with {n_aug} augmentations)")

    all_features = np.array(all_features)
    labels = np.array(labels)

    print(f"\nTotal samples: {len(labels)}")
    print(f"Positive: {sum(labels == 1)}  Negative: {sum(labels == 0)}")
    print(f"Feature shape per sample: {all_features[0].shape}")

    np.savez(os.path.join("data", "features", "features.npz"), features=all_features, labels=labels)
    print("Saved data/features/features.npz")

if __name__ == "__main__":
    main()