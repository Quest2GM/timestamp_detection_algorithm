# ================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-05-18
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
# ================================================================

# Load files
from yolov3 import Create_Yolov3
from utils import detect_image
from configs import *

# Image Imports
from PIL import Image
import glob
import os
from configs_main import PATH_DIR

input_size = YOLO_INPUT_SIZE
path_in = os.path.join(PATH_DIR, "Images", "401-2")
save_to = os.path.join(PATH_DIR, "Images", "crop.jpg")

# Load Keras weights
yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
yolo.load_weights("weights/checkpoints2/yolov3_custom")

# Detect bounding boxes in image
for i, filename in enumerate(glob.glob(path_in + r"\*.jpg")):
    print(filename)
    orig, img, locs = detect_image(yolo, filename, save_to, input_size=input_size, show=True, CLASSES=TRAIN_CLASSES,
                                   rectangle_colors=(255, 0, 0))
    for j, crops in enumerate(locs):
        new_img = orig[crops[1]:crops[3],crops[0]:crops[2],:3]
        new_img = Image.fromarray(new_img)

        w_size = int(float(new_img.size[0]) * float(32 / float(new_img.size[1])))
        new_img = new_img.resize((w_size, 32), Image.ANTIALIAS)

        # path_ext = str(i+1200) + "-" + str(j) + ".jpg"
        # new_img.save(os.path.join(r"C:\Users\sid_a\PycharmProjects\MTO_Highway_Analysis\time_stamps", path_ext))

    print(i)

