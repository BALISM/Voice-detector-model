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
    model_path = os.path.join("models", "wake_word_model.pt")
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Please train the model first.")
        sys.exit(1)

    model = WakeWordCNN(n_mfcc=39, n_frames=63)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_dir = "tests"
    if not os.path.isdir(test_dir):
        print(f"No '{test_dir}' directory found. Place .wav files there.")
        sys.exit(1)

    all_test_files = os.listdir(test_dir)
    wav_files = sorted([f for f in all_test_files if f.lower().endswith(".wav")])
    non_wav_files = sorted([f for f in all_test_files if f.lower().endswith((".ogg", ".mp3", ".m4a", ".flac", ".aac", ".wma"))])

    if non_wav_files:
        print(f"\n[NOTICE] Found {len(non_wav_files)} non-WAV audio clip(s) in '{test_dir}/': {', '.join(non_wav_files)}")
        print("To convert them to .wav, run this PowerShell command:")
        print("  Get-ChildItem tests\\*.ogg | ForEach-Object { ffmpeg -i $_.FullName -ar 16000 -ac 1 \"tests\\$($_.BaseName).wav\" }\n")

    if not wav_files:
        print(f"No .wav files found in '{test_dir}/'. Please convert your test clips to .wav format first.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  BATCH TEST -- {len(wav_files)} clips  (threshold: {THRESHOLD*100:.0f}%)")
    print(f"{'='*65}\n")
    print(f"  {'File':<25} {'Marker':<6} {'Result':<22} {'Confidence':>10}")
    print(f"  {'-'*25} {'-'*6} {'-'*22} {'-'*10}")

    for fname in wav_files:
        filepath = os.path.join(test_dir, fname)
        try:
            conf = predict_one(model, filepath)
            label = "WAKE WORD DETECTED" if conf > THRESHOLD else "not wake word"
            marker = "[OK]" if conf > THRESHOLD else "[X]"
            print(f"  {fname:<25} {marker:<6} {label:<22} {conf*100:>8.1f}%")
        except Exception as e:
            print(f"  {fname:<25} [ERR] Error processing file: {e}")

    print(f"\n{'='*65}\n")

if __name__ == "__main__":
    main()
