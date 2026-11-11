import torch
from torch import nn

class DigitCNN(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten())
        self.classifier=nn.Sequential(nn.Linear(64*7*7,128),nn.ReLU(),nn.Dropout(0.25),nn.Linear(128,n_classes))
    def forward(self,x): return self.classifier(self.features(x))
