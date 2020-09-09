# ================================================================
#
#  Filename:     denoiser_train.py
#  Author:       Siddarth Narasimhan
#  Description:  Used to train an autoencoder to denoise images
#                of digits.
#
# ================================================================

# Import PyTorch
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# Import PIL and cv2 for image preprocessing
from PIL import Image, ImageOps
import os
from configs_main import PATH_DIR
import matplotlib.pyplot as plt

# Import Neural Network for de-noising digits
from neural_nets.neural_networks import DigitDenoiseV3


# Function used to train the autoencoder
def train_model(model, train_loader, epochs=125, batch_size=1, learning_rate=0.5e-3):

    # Specify manual seed for consistency
    # torch.manual_seed(50)

    # Load data and specify loss function and optimizer
    data_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # Iterate through all the images, keeping track of loss and iterations
    iters, losses = [], []
    i = 0
    for epoch in range(epochs):
        for data in data_loader:
            img_noisy, label = data

            # Check if GPU is available
            if torch.cuda.is_available():
                img_noisy, label = img_noisy.cuda(), label.cuda()

            # The true image will be mapped to its corresponding denoised image
            # Note that the label does not always correspond to the actual digit in the image due some
            # strange array ordering in PyTorch's DataLoader
            if label.item() == 0:
                img = img_0
            elif label.item() == 1:
                img = img_1
            elif label.item() == 2:
                img = img_10
            elif label.item() == 3:
                img = img_11
            elif label.item() == 4:
                img = img_2
            elif label.item() == 5:
                img = img_3
            elif label.item() == 6:
                img = img_4
            elif label.item() == 7:
                img = img_5
            elif label.item() == 8:
                img = img_6
            elif label.item() == 9:
                img = img_7
            elif label.item() == 10:
                img = img_8
            elif label.item() == 11:
                img = img_9

            # Check if GPU is available
            if torch.cuda.is_available():
                img = img.cuda()

            recon = model(img_noisy)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('Epoch:{}, Iterations:{}, Loss:{:.4f}'.format(epoch, i, float(loss)))
            i += 1

        iters.append(epoch)
        losses.append(loss)

        print('Epoch:{}, Loss:{:.4f}'.format(epoch, float(losses[-1])))

    # Save model
    torch.save(model.state_dict(), os.path.join(PATH_DIR, "neural_nets", "auto_denoise_new.pb"))

    # Plotting Accuracy
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.legend(loc='best')
    plt.show()


# Directory to folders of images where the denoised images are
path = os.path.join(PATH_DIR, "neural_nets", "net_classes_autoenc")

# Specify all of the directories to find the corresponding numbers. Transform to PyTorch tensor for augmentation
img_0 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\0\0.jpg")))
img_1 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\1\1.jpg")))
img_2 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\2\2.jpg")))
img_3 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\3\3.jpg")))
img_4 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\4\4.jpg")))
img_5 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\5\5.jpg")))
img_6 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\6\6.jpg")))
img_7 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\7\7.jpg")))
img_8 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\8\8.jpg")))
img_9 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\9\9.jpg")))
img_10 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\10\10.jpg")))
img_11 = transforms.ToTensor()(ImageOps.grayscale(Image.open(path + r"\11\11.jpg")))

# Load the training data and specify transformations
trans = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

# Specify folder where the training images are found, separated by class
root = os.path.join(PATH_DIR, "time_stamp_detection", "yolov3", "time_stamps", "augmented")
data = torchvision.datasets.ImageFolder(root=root, transform=trans)

model = DigitDenoiseV3()

# Check if GPU is available
if torch.cuda.is_available():
    model = model.cuda()

# Training will only work with batch_size = 1 at the moment
train_model(model, data, epochs=125, batch_size=1, learning_rate=0.5e-3)
