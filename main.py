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

X = []
Y = []

for i in range(num_enco_files):
    img_path = enco_path + "/after_" + str(i) + ".jpg"
    print(img_path)
    img = cv2.imread(img_path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = np.reshape(img, ((1,) + img.shape))
    X.append(img)
    # Y.append(label)


for i in range(num_deco_f):
    img_path = deco_path + "/heatmap_" + str(i) + ".jpg"
    print(img_path)
    img = cv2.imread(img_path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = np.reshape(img, ((1,) + img.shape))
    Y.append(img)

print("model 생성시작...")
# model
demo = detection.Detection_model()
demo.summary()
print("model 생성완료")


# callback
callbacks_list = [
    EarlyStopping(
        monitor="val_acc",
        patience=1,
    ),
    ModelCheckpoint(
        filepath="model.hr",  # 모델파일 경로,
        monitor="val_loss",
        save_best_only=True,
    ),
]

# compile
demo.compile(optimizer="adam", loss=loss_region, metrics=["accuracy"])  # acc

print("model 학습 시작...")

# fit
demo.fit(X[0], Y[0], batch_size=1, epochs=1, verbose=True, callbacks=callbacks_list)

print("model 학습 종료")
