# ================================================================
#
#  Filename:     canny_edge_detect.py
#  Author:       Siddarth Narasimhan
#  Description:  Script is mostly used to experiment with
#                canny edge detection in images. Parts of
#                this script are used for time stamp detection in
#                time_stamp_reader.py.
#
# ================================================================


import cv2
import numpy as np
import glob
import os
from configs_main import PATH_DIR

path_in = os.path.join(PATH_DIR, "time_stamp_detection", "yolov3", "time_stamps", "full", "cat1")

# Loop through all images in directory, applying a bilateral filter and then canny ede detection
for i, filename in enumerate(glob.glob(path_in + "\*.jpg")):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filt_image = cv2.bilateralFilter(image, 10, 100, 150)
    canny_image = cv2.Canny(filt_image, 15, 200)

    contours, hier = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("______________")
    print("File:", i)
    print(filename)
    widths = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        widths.append(w)
        print("Box", x, y, w, h)
        cv2.rectangle(canny_image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    if len(widths) == 0:
        cv2.imshow("bilateral", filt_image)
        cv2.imshow("contours", canny_image)
        cv2.imshow("original", image)
        cv2.waitKey(0)
        print("No text detected")
        continue

    # Eliminate detection outliers by only including box sizes within the 35% - 75% percentile
    if len(widths) >= 5:
        q1 = np.quantile(widths, 0.35)
        q3 = np.quantile(widths, 0.75)
        print("Quartiles:", q1, q3)
        indices = np.where((widths >= q1) & (widths <= q3))[0]
        final = np.take(widths, indices)
        final = round(sum(final) / len(final))
    else:
        final = int(round(sum(widths) / len(widths)))

    # Display predicted average bounding box size and display filtered images
    print("Out Width:", final)
    cv2.drawContours(canny_image, contours, -1, (255, 0, 0), 1)
    cv2.imshow("contours", canny_image)
    cv2.imshow("bilateral", filt_image)
    cv2.imshow("original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
