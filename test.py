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




#모델을 5epoch 마다 가중치 저장 - path 설정 및 추가
#

checkpoint_path = "./model/dt_model_save/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path) # Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                   save_weights_only=True,
                                   verbose=1) # Train the model with the new callback

model.detection_model()

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # Pass callback to training# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.


