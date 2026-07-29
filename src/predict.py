import sys
import os
import torch
import numpy as np
from model import WakeWordCNN
from extract_features import extract_mfcc

THRESHOLD = 0.85

def find_file_suggestion(filepath):
    """Search for the file basename across common project directories."""
    basename = os.path.basename(filepath)
    search_dirs = [
        os.path.join("data", "dataset", "positive"),
        os.path.join("data", "dataset", "negative"),
        "tests"
    ]
    for d in search_dirs:
        possible_path = os.path.join(d, basename)
        if os.path.exists(possible_path):
            return possible_path
    return None

def predict(filepath):
    if not os.path.exists(filepath):
        print(f"\n[ERROR] File not found: '{filepath}'")
        suggestion = find_file_suggestion(filepath)
        if suggestion:
            print(f"Did you mean: {suggestion} ?")
            print(f"\nTry running:")
            print(f"  py src\\predict.py {suggestion}")
        else:
            print("Please verify the file path or copy the file into the 'tests\\' directory.")
        sys.exit(1)

    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".wav":
        print(f"\n[WARNING] '{filepath}' is a {ext} file. For best accuracy, convert to .wav format first:")
        output_wav = os.path.splitext(filepath)[0] + ".wav"
        print(f'  ffmpeg -i "{filepath}" -ar 16000 -ac 1 "{output_wav}"')

    model_path = os.path.join("models", "wake_word_model.pt")
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Trained model file not found at '{model_path}'.")
        print("Please train the model first by running:")
        print("  py src\\extract_features.py")
        print("  py src\\train.py")
        sys.exit(1)

    model = WakeWordCNN(n_mfcc=39, n_frames=63)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    try:
        features = extract_mfcc(filepath, is_training=False)
    except Exception as e:
        print(f"\n[ERROR] Could not extract audio features from '{filepath}': {e}")
        sys.exit(1)
    
    FIXED_FRAMES = 63
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

    label = "WAKE WORD DETECTED" if best_confidence > THRESHOLD else "not wake word"
    print(f"\nFile: {filepath}")
    print(f"Result: {label}")
    print(f"Confidence: {best_confidence*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: py src\\predict.py path\\to\\your\\file.wav")
        sys.exit(1)
    predict(sys.argv[1])