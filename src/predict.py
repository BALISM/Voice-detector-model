import sys
import os
import torch
import numpy as np
from model import WakeWordCNN
from extract_features import extract_mfcc

def predict(filepath):
    model = WakeWordCNN(n_mfcc=13, n_frames=63)
    model.load_state_dict(torch.load(os.path.join("models", "wake_word_model.pt"), weights_only=True))
    model.eval()

    mfccs = extract_mfcc(filepath, is_training=False)
    
    FIXED_FRAMES = 63
    if mfccs.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode="constant")

    best_confidence = 0.0
    
    with torch.no_grad():
        for i in range(mfccs.shape[1] - FIXED_FRAMES + 1):
            window = mfccs[:, i:i+FIXED_FRAMES]
            tensor_input = torch.tensor(window).unsqueeze(0).unsqueeze(0).float()
            output = model(tensor_input)
            probabilities = torch.softmax(output, dim=1)
            
            confidence = probabilities[0][1].item()
            if confidence > best_confidence:
                best_confidence = confidence

    label = "WAKE WORD DETECTED" if best_confidence > 0.5 else "not wake word"
    print(f"\nFile: {filepath}")
    print(f"Result: {label}")
    print(f"Confidence: {best_confidence*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: py predict.py path\\to\\your\\file.wav")
        sys.exit(1)
    predict(sys.argv[1])