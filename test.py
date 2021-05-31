from model.detection import detection
from model.detection.lossFuntion import loss_region
import cv2
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np

enco_path = "data/image/resized"
original_file_list = os.listdir(enco_path)
num_enco_files = len(original_file_list)

deco_path = "data/image/hitmap"
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
        Y = np.concatenate((X, img), axis=0)  # 추가된 차원(4차원) 방향으로 이미지 연결
print("X shape : {}".format(X.shape))
print("Y shape : {}".format(Y.shape))


a = np.ones((3, 3, 1))
print(a)
b = np.ones((3, 3, 3)) * 3
print(b)

c = b - a
print(c)

140
3
R, B, G
200, 20, 100
200 - 140, 20 - 140, 100 - 140

106 - 140


# IOU check
# IOU 활용법
# IOU >= 0.8 일시, 해당 case는 TP(True Positive)
# precision == 전체 예측 갯수 중 TP의 갯수
# recall == 전체 정답지 갯수 중 TP의 갯수
# 비교방법 == 예측치 하나와 전체 정답지 갯수를 비교한 뒤, IOU가 0.8 이상이 나오면 break 하고 TP로 설정


def IoU(predict, answer):
    # box = (x1, y1, x2, y2)
    predict_area = (predict["xmax"] - predict["xmin"] + 1) * (
        predict["ymax"] - predict["ymin"] + 1
    )
    answer_area = (predict["xmax"] - predict["xmin"] + 1) * (
        predict["ymax"] - predict["ymin"] + 1
    )

    # obtain x1, y1, x2, y2 of the intersection
    inter_xmin = max(predict["xmin"], answer["xmin"])
    inter_ymin = max(predict["ymin"], answer["ymin"])
    inter_xmax = min(predict["xmax"], answer["xmax"])
    inter_ymax = min(predict["ymax"], answer["ymax"])

    # compute the width and height of the intersection
    inter_w = max(0, inter_xmax - inter_xmin + 1)
    inter_h = max(0, inter_ymax - inter_ymin + 1)

    inter_area = inter_w * inter_h
    iou = inter_area / (predict_area + answer_area - inter_area)
    return iou


def TP_check(predict_list, answer_list):
    # 예측 박스와 전체 정답 박스를 비교 하되, iou가 0.8 이상이 나오면 break 후 TP 갯수로 출력
    tp_case = 0
    for predict in predict_list:
        for answer in answer_list:
            iou = IoU(predict, answer)
            if iou >= 0.8:
                tp_case += 1
                break
            else:
                pass
    precision = tp_case / len(predict_list)
    recall = tp_case / len(answer_list)
    return precision, recall


# 모델을 5epoch 마다 가중치 저장 - path 설정 및 추가
#

checkpoint_path = "./model/dt_model_save/cp.ckpt"
checkpoint_dir = os.path.dirname(
    checkpoint_path
)  # Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)  # Train the model with the new callback

model.detection_model()

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback],
)  # Pass callback to training# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
