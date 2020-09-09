# ================================================================
#
#  Filename:     augment.py
#  Author:       Siddarth Narasimhan
#  Description:  Used to perform data augmentation and increase
#                size of a dataset for training a neural network.
#
# ================================================================

# Import PyTorch
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

import os
from configs_main import PATH_DIR

# Include all transformations to be applied here. A complete list of transformations available are found at
# https://pytorch.org/docs/stable/torchvision/transforms.html
trans = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.RandomAffine(degrees=(0, -0), translate=(0.35, 0.35), fillcolor=(255, 255, 255)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomApply([transforms.RandomResizedCrop((120, 120), scale=(0.5, 1.0), ratio=(0.75, 1.333),
                                                         interpolation=2)], p=0.02),
    transforms.ToTensor()
])

# The root folder where the data is located
root = os.path.join(PATH_DIR, "time_stamp_detection", "yolov3", "time_stamps", "separated")

# Load the data
data = torchvision.datasets.ImageFolder(root=root, transform=trans)
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

# Modify the number of iterations based on requirement
# Specify path to save images in the save_image function
n = 0
iterations = 1
for _ in range(iterations):
    for img, label in iter(data_loader):
        save_image(img, os.path.join(PATH_DIR, "img" + str(n) + ".jpg"))
        n += 1
