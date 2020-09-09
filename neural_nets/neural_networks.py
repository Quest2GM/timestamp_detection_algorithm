# ================================================================
#
#  Filename:     neural_networks.py
#  Author:       Siddarth Narasimhan
#  Description:  Contains the architecture for an autoencoder
#                for image denoising and a multi-layer perceptron
#                for digit classification.
#
# ================================================================


import torch.nn as nn
import torch.nn.functional as F


# AutoEncoder class used for de-noising images
class DigitDenoiseV3(nn.Module):
    def __init__(self):
        super(DigitDenoiseV3, self).__init__()
        self.encoder = nn.Sequential(                              # Input Image Size: 32 x 32
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # Output: 16 x 16
            nn.Conv2d(16, 48, 3, stride=2, padding=1), nn.ReLU(),  # Output: 8 x 8
            nn.Conv2d(48, 64, 3, stride=1, padding=0), nn.ReLU(),  # Output: 6 x 6
            nn.Conv2d(64, 80, 3, stride=1, padding=0)              # Output: 4 x 4
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(80, 64, 3, stride=1, padding=0), nn.ReLU(),                    # Output: 6 x 6
            nn.ConvTranspose2d(64, 48, 3, stride=1, padding=0), nn.ReLU(),                    # Output: 8 x 8
            nn.ConvTranspose2d(48, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # Output: 16 x 16
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),              # Output: 32 x 32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Neural network for classifying digit outputs of the AutoEncoder
class ANNDigitDetect(nn.Module):
    def __init__(self):
        super(ANNDigitDetect, self).__init__()
        self.fc1 = nn.Linear(32 * 32, 120)
        self.fc2 = nn.Linear(120, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x