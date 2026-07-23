"""
Batch test all .wav files in the tests/ directory.
Usage: py src/test_all.py
"""
import os
import sys
import torch
import numpy as np

# Add src to path so imports work when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from model import WakeWordCNN
from extract_features import extract_mfcc

THRESHOLD = 0.85
FIXED_FRAMES = 63

def predict_one(model, filepath):
    features = extract_mfcc(filepath, is_training=False)
    
    if features.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")

    best_confidence = 0.0
    
    with torch.no_grad():
        for i in range(features.shape[1] - FIXED_FRAMES + 1):
            window = features[:, i:i+FIXED_FRAMES]
            tensor_input = torch.tensor(window).unsqueeze(0).unsqueeze(0).float()
            output = model(tensor_input)
            probabilities = torch.softmax(output, dim=1)
            
            confidence = probabilities[0][1].item()
            if confidence > best_confidence:
                best_confidence = confidence

    return best_confidence

def main():
    model = WakeWordCNN(n_mfcc=39, n_frames=63)
    model.load_state_dict(torch.load(os.path.join("models", "wake_word_model.pt"), weights_only=True))
    model.eval()

    test_dir = "tests"
    if not os.path.isdir(test_dir):
        print(f"No '{test_dir}' directory found. Place .wav files there.")
        sys.exit(1)

    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".wav")])
    if not files:
        print(f"No .wav files found in '{test_dir}/'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  BATCH TEST — {len(files)} clips  (threshold: {THRESHOLD*100:.0f}%)")
    print(f"{'='*60}\n")
    print(f"  {'File':<25} {'Result':<22} {'Confidence':>10}")
    print(f"  {'-'*25} {'-'*22} {'-'*10}")

    for fname in files:
        filepath = os.path.join(test_dir, fname)
        conf = predict_one(model, filepath)
        label = "WAKE WORD DETECTED" if conf > THRESHOLD else "not wake word"
        marker = "✓" if conf > THRESHOLD else "✗"
        print(f"  {fname:<25} {marker} {label:<20} {conf*100:>8.1f}%")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
