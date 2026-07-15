import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from model import WakeWordCNN

EPOCHS = 30
LR = 0.001

def main():
    data = np.load("features.npz")
    X, y = data["features"], data["labels"]

    X = torch.tensor(X).unsqueeze(1).float()
    y = torch.tensor(y).long()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    model = WakeWordCNN(n_mfcc=X.shape[2], n_frames=X.shape[3])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        print(f"Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).argmax(dim=1)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    print(f"\nTest results -> Precision: {precision:.2f}  Recall: {recall:.2f}  F1: {f1:.2f}")

    torch.save(model.state_dict(), "wake_word_model.pt")
    print("Saved wake_word_model.pt")

if __name__ == "__main__":
    main()