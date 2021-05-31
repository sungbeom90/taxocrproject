from tests.detection_t import detection_tests
from model.detection.lossFuntion import loss_region
from tests.detection_t import load_data
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tests.detection_t.detection_tests import Detection_callback

import cv2
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

# 훈련 데이터 불러오기
train_x, val_x, train_y, val_y = load_data.load_data()


print("model 생성시작...")
# model
model = detection.Detection_model()
model.summary()
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
    Detection_callback(
        train_x=train_x, val_x=val_x, train_y=train_y, val_y=val_y
    )
]

# compile
model.compile(optimizer="adam", loss=loss_region, metrics=["accuracy"])  # acc

print("model 학습 시작...")

# fit
model.fit(
    train_x, train_y, batch_size=1, epochs=100, verbose=True, callbacks=callbacks_list
)

print("model 학습 종료")
