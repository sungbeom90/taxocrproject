from model.detection import detection

import cv2
import os
import tensorflow as tf

enco_path = "/data/image/resized"
original_file_list = os.listdir(enco_path)
num_enco_files = len(original_file_list)

deco_path = "data/image/hitmap"
heatmap_file_list = os.listdir(deco_path)
num_deco_f = len(heatmap_file_list)

X = []
Y = []

for i in range(num_enco_files):
    img_path = enco_path + "/after_" + str(i) + ".jpg"
    print(img_path)
    img = cv2.imread(img_path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    X.append(img)
    # Y.append(label)


for i in range(num_deco_f):
    img_path = deco_path + "/heatmap_" + str(i) + ".jpg"
    print(img_path)
    img = cv2.imread(img_path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    Y.append(img)


demo = detection.Detection_model()
demo.summary()
