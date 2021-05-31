# import necessary layers
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tests.detection import decoding, iou


# Concatenate 클래스 선언
class Concatenate_(Layer):
    def __init__(self):
        super(Concatenate_, self).__init__()
        self.concat = Concatenate(axis=-1)

    def call(self, fir, sec):
        return self.concat([fir, sec])


# Maxpool2D 클래스 선언
class Max_Pool2D(Layer):
    def __init__(self):
        super(Max_Pool2D, self).__init__()
        self.mxpool = MaxPool2D(2)

    def call(self, inputs):
        return self.mxpool(inputs)


# Conv_layer 클래스 선언
class Conv_layer(Layer):
    def __init__(self, ft, ks):
        super(Conv_layer, self).__init__()
        self.filters = ft
        self.kernel_size = ks
        self.conv_layer = Conv2D(ft, ks, padding="same")
        self.batch = BatchNormalization()
        self.activation = Activation("relu")

    def call(self, inputs):
        x = self.conv_layer(inputs)
        x = self.batch(x)
        return self.activation(x)


# Stage_layer 클래스 선언
class Stage_layer(Layer):
    def __init__(self, stn):
        super(Stage_layer, self).__init__()
        self.conv_layer1 = Conv_layer(stn, 3)
        self.conv_layer2 = Conv_layer(stn, 3)
        self.mxpool = Max_Pool2D()

    def call(self, inputs):
        x = self.conv_layer1(inputs)
        x1 = self.conv_layer2(x)
        mx_1 = self.mxpool(x1)
        return mx_1, x1


# Encoding_layer 선언부
class Encoding_layer(Layer):
    def __init__(self):
        super(Encoding_layer, self).__init__()
        self.stage1 = Stage_layer(64)
        self.stage2 = Stage_layer(128)
        self.stage3 = Stage_layer(256)
        self.stage4 = Stage_layer(512)

    def call(self, inputs):
        mx_1, stg1 = self.stage1(inputs)
        mx_2, stg2 = self.stage2(mx_1)
        mx_3, stg3 = self.stage3(mx_2)
        mx_4, stg4 = self.stage4(mx_3)
        return mx_4, stg1, stg2, stg3, stg4


# Middle_layer 클래스 선언
class Middle_layer(Layer):
    def __init__(self):
        super(Middle_layer, self).__init__()
        self.conv_layer1 = Conv_layer(512, 3)
        self.conv_layer2 = Conv_layer(1024, 3)
        self.concat = Concatenate_()
        self.mxpool = Max_Pool2D()

    def call(self, inputs, cont):
        x = self.conv_layer1(inputs)
        x_cont1 = self.concat(x, cont)
        x1 = self.conv_layer2(x_cont1)
        return x1


# Conv_Transe_Layer 선언
class Conv_Trans(Layer):
    def __init__(self, ft):
        super(Conv_Trans, self).__init__()
        self.filters = ft
        self.conv_trans = Conv2DTranspose(
            ft, kernel_size=2, strides=(2, 2), padding="same"
        )
        self.batch = BatchNormalization()
        self.activation = Activation("relu")

    def call(self, inputs):
        x = self.conv_trans(inputs)
        x = self.batch(x)
        return self.activation(x)


# Upstage_layer 선언
class UpStage_layer(Layer):
    def __init__(self, stn):
        super(UpStage_layer, self).__init__()
        self.stagenum = stn

        if stn == 64:
            self.conv_trans = Conv_Trans(64)
            self.concat = Concatenate_()  # stg2랑 더해지겠지
            self.conv_layer1 = Conv_layer(64, 3)
            self.conv_layer2 = Conv_layer(64, 3)
        else:
            self.conv_trans = Conv_Trans(stn)
            self.concat = Concatenate_()  # stg4랑 더해지겠지
            self.conv_layer1 = Conv_layer(stn, 3)
            self.conv_layer2 = Conv_layer(stn / 2, 3)

    def call(self, inputs, cont):
        x = self.conv_trans(inputs)
        x_cont1 = self.concat(x, cont)
        x1 = self.conv_layer1(x_cont1)
        x2 = self.conv_layer2(x1)
        return x2


### Decoding_layer 선언부
class Decoding_layer(Layer):
    def __init__(self):
        super(Decoding_layer, self).__init__()
        self.upstage1 = UpStage_layer(512)
        self.upstage2 = UpStage_layer(256)
        self.upstage3 = UpStage_layer(128)
        self.upstage4 = UpStage_layer(64)

    def call(self, inputs, stg1, stg2, stg3, stg4):
        ups_1 = self.upstage1(inputs, stg4)
        ups_2 = self.upstage2(ups_1, stg3)
        ups_3 = self.upstage3(ups_2, stg2)
        ups_4 = self.upstage4(ups_3, stg1)
        return ups_4


### Detection_model


class Detection_model(Model):
    def __init__(self, **kwargs):
        super(Detection_model, self).__init__()
        self.encoding = Encoding_layer()
        self.middle = Middle_layer()
        self.decoding = Decoding_layer()
        self.fin_conv = Conv2D(1, 1, padding="same")
        self._build(**kwargs)

    def call(self, inputs, training=False):
        mx_4, stg1, stg2, stg3, stg4 = self.encoding(inputs)
        x = self.middle(mx_4, mx_4)
        x = self.decoding(x, stg1, stg2, stg3, stg4)
        return self.fin_conv(x)

    def _build(self, **kwargs):
        inputs = Input(shape=[1024, 1024, 1])
        outputs = self.call(inputs)
        super(Detection_model, self).__init__(inputs=inputs, outputs=outputs, **kwargs)


class Detection_callback(Callback):
    def __init__(self, train_x, val_x, train_y, val_y):
        super(Detection_callback, self).__init__()
        self.train_x = train_x
        self.val_x = val_x
        self.train_y = train_y
        self.val_y = val_y

    def on_train_begin(self, logs):
        self.losses = []

    def on_epoch_end(self, epoch, logs):
        # self.model.save_weights("data/train_weights/" + weight_name + "_last")
        self.losses.append(logs.get("loss"))
        if (epoch + 1) % 1 == 0:
            print("epoch : {}".format(epoch + 1))
            print(
                "loss : {}, val_loss : {}".format(
                    logs.get("loss"), logs.get("val_loss")
                )
            )
        if (epoch + 1) % 3 == 0:
            precision_total = 0
            recall_total = 0
            val_num = len(self.val_y)
            for i in range(val_num):
                print("train_x[{}].shape : {}".format(i, self.train_x[i].shape))
                train_temp = np.reshape(self.train_x[i], ((1,) + self.train_x[i].shape))
                print("add axis train_x[{}].shape : {}".format(i, train_temp.shape))
                pred_y = self.model.predict(train_temp)
                print("pred_y.shape : {}".format(pred_y.shape))
                pred_y = pred_y.reshape(pred_y.shape[1], pred_y.shape[2])
                print("change pred_y.shape : {}".format(pred_y.shape))
                print("val_y[{}].shape : {}".format(i, self.val_y[i].shape))
                true_y = self.val_y[i].reshape(
                    self.val_y[i].shape[0], self.val_y[i].shape[1]
                )
                print("change true_y.shape : {}".format(true_y.shape))
                predict_list = decoding.fun_decoding(pred_y)
                answer_list = decoding.fun_decoding(true_y)
                print(
                    "pred_list_num : {}, answer_list_num : {}".format(
                        len(predict_list), len(answer_list)
                    )
                )
                for a in range(len(predict_list)):
                    xmin = int(predict_list[a].get("xmin"))
                    xmax = int(predict_list[a].get("xmax"))
                    ymin = int(predict_list[a].get("ymin"))
                    ymax = int(predict_list[a].get("ymax"))
                    predict_img = cv2.rectangle(
                        self.train_x[i], (xmin, ymin), (xmax, ymax), (255, 0, 0), 1
                    )
                for a in range(len(answer_list)):
                    xmin = int(answer_list[a].get("xmin"))
                    xmax = int(answer_list[a].get("xmax"))
                    ymin = int(answer_list[a].get("ymin"))
                    ymax = int(answer_list[a].get("ymax"))
                    answer_img = cv2.rectangle(
                        self.val_y[i], (xmin, ymin), (xmax, ymax), (0, 255, 0), 1
                    )
                cv2.imshow("predict_img", predict_img)
                cv2.imshow("answer_img", answer_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                precision, recall = iou.TP_check(predict_list, answer_list)
                precision_total += precision
                recall_total += recall
            precision_mean = precision_total / val_num
            recall_mean = recall_total / val_num
            print("precision: {}, recall: {}".format(precision_mean, recall_mean))
