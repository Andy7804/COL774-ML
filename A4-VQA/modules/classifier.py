# modules/classifier.py
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=500, num_classes=28):  # Assume 28 classes (you'll set actual value later)
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
