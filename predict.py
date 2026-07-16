import sys
import torch
import numpy as np
from model import WakeWordCNN
from extract_features import extract_mfcc

def predict(filepath):
    model = WakeWordCNN(n_mfcc=13, n_frames=63)
    model.load_state_dict(torch.load("wake_word_model.pt"))
    model.eval()

    mfcc = extract_mfcc(filepath)
    tensor_input = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        output = model(tensor_input)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probabilities[0][prediction].item()

    label = "WAKE WORD DETECTED" if prediction == 1 else "not wake word"
    print(f"\nFile: {filepath}")
    print(f"Result: {label}")
    print(f"Confidence: {confidence*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: py predict.py path\\to\\your\\file.wav")
        sys.exit(1)
    predict(sys.argv[1])