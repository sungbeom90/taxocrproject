import cv2
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():

    enco_path = "/data/image/resized"
    original_file_list = os.listdir(enco_path)
    num_enco_files = len(original_file_list)

    deco_path = "/data/image/hitmap"
    heatmap_file_list = os.listdir(deco_path)
    num_deco_f = len(heatmap_file_list)

    X = None
    Y = None

    for i in range(num_enco_files):
        img_path = enco_path + "/after_" + str(i) + ".jpg"
        print(img_path)
        img = cv2.imread(img_path)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = np.reshape(img, ((1,) + img.shape))  # 차원추가
        if i == 0:
            X = img
        else:
            X = np.concatenate((X, img), axis=0)  # 추가된 차원(4차원) 방향으로 이미지 연결

    for i in range(num_deco_f):
        img_path = deco_path + "/heatmap_" + str(i) + ".jpg"
        print(img_path)
        img = cv2.imread(img_path)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = np.reshape(img, ((1,) + img.shape))  # 차원추가
        if i == 0:
            Y = img
        else:
            Y = np.concatenate((Y, img), axis=0)  # 추가된 차원(4차원) 방향으로 이미지 연결

    train_x, train_y, test_x, test_y = train_test_split(X, Y, test_size=0.3)
    val_x = X
    val_y = Y

    return train_x, train_y, val_x, val_y, test_x, test_y
