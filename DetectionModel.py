# import necessary layers
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate
import pickle
import cv2
import tensorflow
import numpy as np
import tensorflow.keras.backend as K

# 나열한 모델입니다.
import cv2

inputs = Input(shape=(1024, 1024, 3))
type(inputs)
# Encoding_stage_1 (64 filters로 2개층)
x = Conv2D(filters=64, kernel_size=3, padding="same")(inputs)
x1 = BatchNormalization()(x)
x1 = Activation("relu")(x1)
x1 = Conv2D(filters=64, kernel_size=3, padding="same")(x1)
x1 = BatchNormalization()(x1)
x1 = Activation("relu")(x1)
x_1 = MaxPool2D(2)(x1)
# Encoding_stage_2 (128 filters로 2개층)
x2 = Conv2D(filters=128, kernel_size=3, padding="same")(x_1)
x2 = BatchNormalization()(x2)
x2 = Activation("relu")(x2)
x2 = Conv2D(filters=128, kernel_size=3, padding="same")(x2)
x2 = BatchNormalization()(x2)
x2 = Activation("relu")(x2)
x_2 = MaxPool2D(2)(x2)
# Encoding_stage_3 (256 filters로 2개층)
x3 = Conv2D(filters=256, kernel_size=3, padding="same")(x_2)
x3 = BatchNormalization()(x3)
x3 = Activation("relu")(x3)
x3 = Conv2D(filters=256, kernel_size=3, padding="same")(x3)
x3 = BatchNormalization()(x3)
x3 = Activation("relu")(x3)
x_3 = MaxPool2D(2)(x3)
# Encoding_stage_4 (512 filters로 2개층)
x4 = Conv2D(filters=512, kernel_size=3, padding="same")(x_3)
x4 = BatchNormalization()(x4)
x4 = Activation("relu")(x4)
x4 = Conv2D(filters=512, kernel_size=3, padding="same")(x4)
x4 = BatchNormalization()(x4)
x4 = Activation("relu")(x4)
x_4 = MaxPool2D(2)(x4)
# Encoding_stage_5 (512 filters 및 concatenate)
x5 = Conv2D(filters=512, kernel_size=3, padding="same")(x_4)
x5 = Concatenate(axis=-1)([x5, x_4])
x5 = BatchNormalization()(x5)
x5 = Conv2D(filters=1024, kernel_size=3, padding="same")(x5)
# Decoding_stage_1 (512 filters 및 concatenate)
t = Conv2DTranspose(512, 2, strides=(2, 2), padding="same")(x5)
t = Concatenate(axis=-1)([t, x4])
t1 = Conv2D(512, 3, padding="same")(t)
t1 = BatchNormalization()(t1)
t1 = Activation("relu")(t1)
t1 = Conv2D(256, 3, padding="same")(t1)
t1 = BatchNormalization()(t1)
t1 = Activation("relu")(t1)
# Decoding_stage_2 (256 filters 및 concatenate)
t2 = Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(t1)
t2 = Concatenate(axis=-1)([t2, x3])
t2 = Conv2D(256, 3, padding="same")(t2)
t2 = BatchNormalization()(t2)
t2 = Activation("relu")(t2)
t2 = Conv2D(128, 3, padding="same")(t2)
t2 = BatchNormalization()(t2)
t2 = Activation("relu")(t2)
# Decoding_stage_3 (128 filters 및 concatenate)
t3 = Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(t2)
t3 = Concatenate(axis=-1)([t3, x2])
t3 = Conv2D(128, 3, padding="same")(t3)
t3 = BatchNormalization()(t3)
t3 = Activation("relu")(t3)
t3 = Conv2D(64, 3, padding="same")(t3)
t3 = BatchNormalization()(t3)
t3 = Activation("relu")(t3)
# Decoding_stage_4 (164 filters 및 concatenate)
t4 = Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(t3)
t4 = Concatenate(axis=-1)([t4, x1])
t4 = Conv2D(64, 3, padding="same")(t4)
t4 = BatchNormalization()(t4)
t4 = Activation("relu")(t4)
t4 = Conv2D(64, 3, padding="same")(t4)
t4 = BatchNormalization()(t4)
t4 = Activation("relu")(t4)
decoder_output = Conv2D(3, 3, padding="same")(t4)
# Output 기반 Model 생성 검증
decoder = Model(inputs, decoder_output, name="decoder")
decoder.summary()

# 손실함수 - 기존 numpy.array 에서 tensor type으로 바꿈
def loss_region(y_true, y_pred):
    result = K.sum(K.square(y_true - y_pred), axis=-1)
    return result


# compile
decoder.compile(optimizer="adam", loss=loss_region, metrics=["accuracy"])  # acc

# fit
x = cv2.imread("./detection_data/test/after_0000000.jpg", cv2.IMREAD_COLOR)
with open("./detection_data/test/heatmap_0", "rb") as MyFile:
    y = pickle.load(MyFile)
x = np.reshape(x, ((1,) + x.shape))
print(x.shape)
x = K.constant(x)
print(type(x))
y = np.reshape(y, ((1,) + y.shape))
y = np.float32(y)
print(y.shape)
y = K.constant(y)
print(type(y))
decoder.fit(x, y, epochs=1, verbose=1)  # 배치사이즈 등 추가 필요
