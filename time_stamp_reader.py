# ================================================================
#
#  Filename:     time_stamp_reader.py
#  Author:       Siddarth Narasimhan
#  Description:  Script used to perform timestamp detection.
#  Other Notes:  The predictions in most cases are only off by a digit or two.
#                While digit recognition has to be performed, we need to find a
#                way to use the final output array to separate the year, date
#                and time, which is a harder task since not all time stamp formats
#                are the same.
#
# ================================================================

# Import PyTorch, OpenCV, Image, Counter
import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import cv2
import numpy as np
from collections import Counter
import time
import glob

import os
from configs_main import PATH_DIR

# Import neural networks for digit recognition and de-noising
from neural_nets.neural_networks import ANNDigitDetect, DigitDenoiseV3


# Calculate mode of input dataset (Counter is more flexible than scipy's mode function)
def mode_arr(x):
    rem_dash = [i for i in x if i != "-" and i != ":"]
    data = Counter(rem_dash)
    freq = data.most_common(2)
    if freq[0][1] > freq[1][1]:
        return freq[0][0]
    elif freq[0][1] == freq[1][1]:
        return freq[1][0]


# Load ANN and Convolutional Autoencoder
detect = ANNDigitDetect()
denoise = DigitDenoiseV3()
detect.load_state_dict(torch.load('neural_nets/ANN_Digit_Detect.pb', map_location=torch.device('cpu')))
denoise.load_state_dict(torch.load('neural_nets/auto_denoise.pb', map_location=torch.device('cpu')))

path_in = os.path.join(PATH_DIR, "time_stamp_detection", "yolov3", "time_stamps", "full", "cat1")

# Loop through all time stamps in specified path
for path in glob.glob(path_in + "\*.jpg"):

    # Keep track of the time to see calculation length per iteration
    start = time.time()

    print("___________________________________________________________")

    # Convert image to grayscale
    img_orig = cv2.imread(path)
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    # Find window width by applying bilateral filtering and canny edge detection
    # This is to find an average window size to use for that time stamp
    # _________________

    image_bifilter = cv2.bilateralFilter(img, 20, 100, 150)
    image_canny = cv2.Canny(image_bifilter, 15, 200)
    contours, hier = cv2.findContours(image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    widths = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        widths.append(w)

    # Eliminate outliers (include ones only within the 35th to 75th percentile)
    if len(widths) >= 5:
        q1 = np.quantile(widths, 0.35)
        q3 = np.quantile(widths, 0.75)
        print("Quartiles:", q1, q3)
        indices = np.where((widths >= q1) & (widths <= q3))[0]
        final = np.take(widths, indices)
        window_width = round(sum(final) / len(final)) + 9
    elif len(widths) == 0:
        cv2.imshow("orig", img_orig)
        cv2.waitKey(0)
        print("Default Width Used")
        window_width = 20
    else:
        window_width = int(round(sum(widths) / len(widths))) + 9

    # Correct for too large or too small window sizes - default to 20
    if window_width > 40 or window_width < 12:
        window_width = 20

    print("Window Width:", window_width)

    # _________________

    # Rescale the height of the image (maintaining aspect ratio) so that it is 32 pixels
    # 32 pixels since this is the height size required for the auto encoder
    img = Image.fromarray(img)
    h_percent = 32 / float(img.size[1])
    w_size = int(float(img.size[0]) * float(h_percent))
    img = img.resize((w_size, 32), PIL.Image.ANTIALIAS)

    i = 0
    date_str = ""
    pred_arr, prob_arr, c = [], [], []

    # Loop through timestamp by cropping by window size and perform digit recognition at each step
    # Increment the window size by a pixel each time until the window reaches the end of the image
    while i < int(w_size - window_width / 2):
        im = img.crop((i, 0, i + window_width, 32))
        im1 = im.resize((32, 32))

        im2 = transforms.ToTensor()(im1).unsqueeze_(0)

        recon = denoise(im2)        # Denoise the imag
        output = detect(recon)      # Predict digit
        prob = F.softmax(output)    # Find confidence probability of prediction

        pred = prob.max(1, keepdim=True)[1]
        val_prob = prob.max(1, keepdim=True)[0]

        # If the probability is large enough, keep track of the prediction
        if val_prob > 0.63 and pred.item() < 10:
            pred_arr.append(pred.item())
            prob_arr.append(val_prob.item())
        elif pred.item() == 11:
            pred_arr.append("-")
        else:
            pred_arr.append("-")
            prob_arr.append(0)

        # Reset for next iteration
        i += 1
        c.append(i)
        recon = recon.detach().numpy()

        # View image if needed
        # cv2.imshow("x", np.array(im1))
        # cv2.imshow("y", recon[0][0])
        # cv2.waitKey(250)
        # cv2.destroyAllWindows()

    # Initialization of variables
    z = [i % 10 for i in range(len(pred_arr))]
    pred_arr = ["-"] + pred_arr     # Prediction arrays should start with "-" otherwise for loop below doesn't work
    new_arr = [pred_arr[0]]
    index_arr, index_dash = [], []

    print(pred_arr)
    print(z)

    # Loop through prediction array and predict the time_stamp based on closely grouped detected digits
    # To do: Repetition in the following for loop needs be fixed for efficiency
    for i, val in enumerate(pred_arr):

        # If window size < 20 then you want to look ahead 3 values, if not 4 values (condition was found to work best)
        # Essentially what we are doing is finding three or four consecutive "-"s in the prediction array (indicating
        # blank spaces in the timestamp), and then trying to identify when a new digit has begun (so that all
        # duplicates are removed)
        if window_width < 20:
            if i == len(pred_arr) - 3:
                break

            if pred_arr[i] == "-" and pred_arr[i + 1] == "-" and pred_arr[i + 2] == "-" and pred_arr[i + 3] == "-":
                if new_arr[-1] != "-":
                    new_arr.append("-")
                    index_arr.append(i)
                    index_dash.append(i)

            elif val != "-":
                if new_arr[-1] != val:
                    new_arr.append(val)
                    index_arr.append(i)

        else:
            if i == len(pred_arr) - 4:
                break

            if pred_arr[i] == "-" and pred_arr[i + 1] == "-" and pred_arr[i + 2] == "-" and pred_arr[i + 3] == "-" \
                and pred_arr[i + 4] == "-":

                if new_arr[-1] != "-":
                    new_arr.append("-")
                    index_arr.append(i)
                    index_dash.append(i)

            elif val != "-":
                if new_arr[-1] != val:
                    new_arr.append(val)
                    index_arr.append(i)

    print(new_arr)
    print(index_arr)
    print(index_dash)

    # Re-initialization for final detection step
    index_arr = [0] + index_arr
    new_list_2 = [index_arr.index(i) for i in index_dash]
    new_list_2 = [0] + new_list_2
    fin_list = []

    print(new_list_2)

    # Predict final timestamp by finding modes for closely connected groups of digits
    for i in range(len(new_list_2) - 1):
        if new_list_2[i + 1] - new_list_2[i] > 2:
            index_1 = index_arr[new_list_2[i]]
            index_2 = index_arr[new_list_2[i + 1]]
            x = mode_arr(pred_arr[index_1:index_2])
            fin_list.append(x)
        else:
            val = new_arr[new_list_2[i] + 1]
            if val != "-":
                fin_list.append(val)

    end = time.time()
    print("Total Detection Time:", end - start)

    # Display prediction and image of time stamp
    print("Final TimeStamp Prediction:", fin_list)
    cv2.imshow("orig", img_orig)
    cv2.waitKey(0)