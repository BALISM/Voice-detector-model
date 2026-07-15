import torch.nn as nn

class WakeWordCNN(nn.Module):
    def __init__(self, n_mfcc=13, n_frames=63):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        pooled_h = n_mfcc // 2 // 2
        pooled_w = n_frames // 2 // 2
        self.fc1 = nn.Linear(16 * pooled_h * pooled_w, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x