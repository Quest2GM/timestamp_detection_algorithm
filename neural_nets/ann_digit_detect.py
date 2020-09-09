# ================================================================
#
#  Filename:     ann_digit_detect.py
#  Author:       Siddarth Narasimhan
#  Description:  Used to help train a multi-layer perceptron (ANN)
#                to recognize digits. This program can be used for
#                experimentation or training a new ANN model.
#
# ================================================================


# Import PyTorch and matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

import os
from configs_main import PATH_DIR
from neural_nets.neural_networks import ANNDigitDetect


def train_model(model, train_data, epochs, batch_size, learning_rate=0.001, momentum=0.9):

    # Load data with DataLoader, define loss and optimizer
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    iters, train_acc = [], []
    n = 0

    for epoch in range(epochs):
        for imgs, labels in iter(train_loader):

            # Learn parameters
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save the current training information
            iters.append(n)
            train_acc.append(get_accuracy(model, train_loader))
            print("N:", n, "Accuracy:", train_acc[-1])

            # Updates to parameters
            n += 1

        print("Epoch:", epoch, " | Training Accuracy:", train_acc[-1])

    # Save model
    torch.save(model.state_dict(), os.path.join(PATH_DIR, "neural_nets", "ANN_Model_New.pb"))

    # Plotting Accuracy
    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))


# Function used to test the accuracy of the model
def get_accuracy(model, data):
    correct = 0
    total = 0
    for imgs, labels in data:
        output = model(imgs)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


# Load the training data and transform by resizing and converting to grayscale
trans = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

# The root is the directory where image classes are found
root = os.path.join(PATH_DIR, "neural_nets", "net_classes_ann")
data = torchvision.datasets.ImageFolder(root=root, transform=trans)

# Load and train the model
model = ANNDigitDetect()
train_model(model, data, epochs=150, batch_size=1)
