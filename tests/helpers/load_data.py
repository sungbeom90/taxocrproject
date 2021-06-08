import cv2
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split

# 변수로 주어진 경로 하위의 .jpg 파일명 가져오는 함수
def load_images(file_path):
    file_list = os.listdir(file_path)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
    return file_list_jpg


def load_data2():

    enco_path = "./data/image/origin_image"
    original_file_list = os.listdir(enco_path)
    original_file_list_jpg = [
        file for file in original_file_list if file.endswith(".jpg")
    ]
    num_enco_files = len(original_file_list)

    deco_path = "./data/image/hitmap"
    heatmap_file_list = os.listdir(deco_path)
    num_deco_f = len(heatmap_file_list)

    X = None
    Y = None

    for i in range(num_enco_files):
        img_path = enco_path + "/after_" + str(i) + ".jpg"
        print(img_path)
        img = cv2.imread(img_path, 0)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        img = img / 255
        img = np.reshape(img, ((1,) + img.shape + (1,)))  # 차원추가
        if i == 0:
            X = img
        else:
            X = np.concatenate((X, img), axis=0)  # 추가된 차원(4차원) 방향으로 이미지 연결

    for i in range(num_deco_f):
        img_path = deco_path + "/heatmap_" + str(i) + ".jpg"
        print(img_path)
        img = cv2.imread(img_path, 0)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        img = img / 255
        img = np.reshape(img, ((1,) + img.shape + (1,)))  # 차원추가
        if i == 0:
            Y = img
        else:
            Y = np.concatenate((Y, img), axis=0)  # 추가된 차원(4차원) 방향으로 이미지 연결

    train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.3)

    return X, X, Y, Y
