from tensorflow.keras.layers import (
    Dense,
    LSTM,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    Lambda,
    Bidirectional,
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def act_model_load_LSTM(char_list, inputs):
    conv_1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)

    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool_2)

    conv_4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv_3)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool_4)

    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3, 3), activation="relu", padding="same")(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2, 2), activation="relu")(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

    outputs = Dense(len(char_list) + 1, activation="softmax")(blstm_2)

    act_model = Model(inputs, outputs, name="load_model")

    return inputs, outputs, act_model
