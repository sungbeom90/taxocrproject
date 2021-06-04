from abc import ABC

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import VersionAwareLayers

layers = VersionAwareLayers()

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
session = tf.compat.v1.Session(config=config)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, units=64, last: bool = False):
        super(ConvBlock, self).__init__()

        self.conv1 = Conv2D(units, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv2 = Conv2D(units, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.maxpool = MaxPooling2D()
        self.last = last

    def call(self, inputs, *_):
        x = self.conv1(inputs)
        x = self.conv2(x)
        if self.last is False:
            x = self.maxpool(x)
            return x
        else:
            return x


class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self, units=64, last=False):
        super(UpsampleBlock, self).__init__()

        self.upconv1 = Conv2D(units, 1, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.batch1 = BatchNormalization()
        self.upconv2 = Conv2D(int(units / 2), 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.batch2 = BatchNormalization()
        self.upsample = UpSampling2D((2, 2))
        self.last = last

    def call(self, inputs, *_):
        x = self.upconv1(inputs)
        x = self.batch1(x)
        x = self.upconv2(x)
        x = self.batch2(x)
        if self.last is True:
            return x
        else:
            return self.upsample(x)


class Craft(Model, ABC):
    def __init__(self, **config):
        super(Craft, self).__init__()

        self.loaded = False
        self.conv_block1 = ConvBlock(32)
        self.conv_block2 = ConvBlock(64)
        self.conv_block3 = ConvBlock(128)
        self.conv_block4 = ConvBlock(256)
        self.conv_block5 = ConvBlock(256)
        self.conv_block6 = ConvBlock(256, last=True)

        self.upsample_block1 = UpsampleBlock(128)
        self.upsample_block2 = UpsampleBlock(128)
        self.upsample_block3 = UpsampleBlock(64)
        self.upsample_block4 = UpsampleBlock(32)
        self.upsample_block5 = UpsampleBlock(32, True)

        self.conv_last1 = Conv2D(32, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_last2 = Conv2D(32, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_last3 = Conv2D(16, 3, 1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv_last4 = Conv2D(16, 1, 1, activation='relu')

        self.conv_fcn = Conv2D(1,1,1)

    @tf.function
    def call(self, x, training=None, mask=None):
        res0 = x
        res1 = self.conv_block1(x)
        res2 = self.conv_block2(res1)
        res3 = self.conv_block3(res2)
        res4 = self.conv_block4(res3)
        x = self.conv_block6(res4)

        x = tf.concat([x, res4], axis=3)
        x = self.upsample_block1(x)

        x = tf.concat([x, res3], axis=3)
        x = self.upsample_block2(x)

        x = tf.concat([x, res2], axis=3)
        x = self.upsample_block3(x)

        x = tf.concat([x, res1], axis=3)
        x = self.upsample_block4(x)

        x = tf.concat([x, res0], axis=3)
        x = self.upsample_block5(x)

        x = self.conv_last1(x)
        x = self.conv_last2(x)
        x = self.conv_last3(x)
        x = self.conv_last4(x)
        x = Dropout(0.5)(x)
        x = self.conv_fcn(x)

        x = tf.squeeze(x, axis=3)

        return x

    def load(self, model_id):
        print('loading detection model')
        self.load_weights('' + model_id).expect_partial()
